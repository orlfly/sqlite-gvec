//! gvec SQLite extension — loadable into any SQLite via `.load libgvec.so`.
//!
//! Symbols registered on init:
//!   - `gvec_version()`           scalar TEXT starting with `v`
//!   - `gvec_dist(a, b, metric)`  scalar REAL
//!
//! For full Cypher execution including graph traversal and pattern matching,
//! use the `gvec` Rust binding crate, which provides the in-process engine
//! over `rusqlite`. This extension exposes only the scalar surface — the
//! full cypher interpreter requires a SQLite connection, which an extension
//! scalar function does not own. (This is a known limitation of the
//! `sqlite-loadable` model: extensions cannot safely run their own queries
//! on the host's connection from a scalar callback.)

use gvec_core::prelude::*;
use sqlite_loadable::prelude::*;
use sqlite_loadable::{api, define_scalar_function, FunctionFlags, Result};

pub const VERSION: &str = concat!("v", env!("CARGO_PKG_VERSION"));

#[sqlite_entrypoint]
pub fn sqlite3_gvec_init(db: *mut sqlite3) -> Result<()> {
    let _ = db;
    define_scalar_function(db, "gvec_version", 0, gvec_version, FunctionFlags::UTF8 | FunctionFlags::DETERMINISTIC)?;
    define_scalar_function(db, "gvec_dist", 3, gvec_dist, FunctionFlags::UTF8 | FunctionFlags::DETERMINISTIC)?;
    define_scalar_function(db, "gvec_query", 1, gvec_query_stub, FunctionFlags::UTF8)?;
    Ok(())
}

fn gvec_version(context: *mut sqlite3_context, _values: &[*mut sqlite3_value]) -> Result<()> {
    api::result_text(context, VERSION)?;
    Ok(())
}

fn gvec_dist(context: *mut sqlite3_context, values: &[*mut sqlite3_value]) -> Result<()> {
    let a = parse_vector_arg(values.get(0).copied());
    let b = parse_vector_arg(values.get(1).copied());
    let metric = api::value_text(values.get(2).expect("metric"))?.to_string();
    let dist = match Distance::parse(&metric) {
        Some(d) => d,
        None => { api::result_error(context, &format!("unknown metric: {metric}")); return Ok(()); }
    };
    match (a, b) {
        (Ok(va), Ok(vb)) => {
            let r = gvec_core::fusion::vector_distance(&va, &vb, dist);
            api::result_double(context, r as f64);
            Ok(())
        }
        (Err(e), _) | (_, Err(e)) => { api::result_error(context, &format!("{e}")); Ok(()) }
    }
}

/// `gvec_query(cypher)` is registered for compatibility with the
/// documentation, but in the loadable extension it returns a clear error
/// pointing users to the Rust binding. The full Cypher executor lives in
/// `gvec-core` and requires an owned `rusqlite::Connection` which a
/// loadable scalar function does not have. See `crates/gvec-bindings`.
fn gvec_query_stub(context: *mut sqlite3_context, _values: &[*mut sqlite3_value]) -> Result<()> {
    api::result_error(context, "gvec_query requires the gvec Rust binding crate (https://crates.io/crates/gvec); the loadable extension exposes scalar functions only");
    Ok(())
}

fn parse_vector_arg(v: Option<*mut sqlite3_value>) -> std::result::Result<VectorValue, gvec_core::Error> {
    let v = v.expect("vector arg");
    let s = sqlite_loadable::api::value_text_notnull(&v)
        .map_err(|e| gvec_core::Error::Vector(e.to_string()))?
        .to_string();
    let val: serde_json::Value = serde_json::from_str(&s).map_err(gvec_core::Error::Serde)?;
    gvec_core::fusion::vector_from_scalar(&val)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_is_v_prefixed() {
        assert!(VERSION.starts_with('v'));
    }
}