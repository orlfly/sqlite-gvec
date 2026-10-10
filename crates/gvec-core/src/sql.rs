//! `SqlExecutor` — the database-agnostic abstraction used by `gvec-core`.
//!
//! `gvec-core` does NOT depend on `rusqlite`. Callers supply an executor
//! implementation. The `gvec` (Rust binding) crate provides one based on
//! `rusqlite::Connection`; the `gvec_extension` crate provides one backed
//! by the host's `*mut sqlite3` (via the loadable extension FFI).

use crate::error::Result;
use serde_json::Value;

/// A single bound value.
#[derive(Debug, Clone)]
pub enum SqlValue {
    Null,
    Integer(i64),
    Real(f64),
    Text(String),
    Blob(Vec<u8>),
}

impl From<i64> for SqlValue {
    fn from(v: i64) -> Self {
        Self::Integer(v)
    }
}
impl From<f64> for SqlValue {
    fn from(v: f64) -> Self {
        Self::Real(v)
    }
}
impl From<String> for SqlValue {
    fn from(v: String) -> Self {
        Self::Text(v)
    }
}
impl From<&str> for SqlValue {
    fn from(v: &str) -> Self {
        Self::Text(v.to_string())
    }
}
impl From<Vec<u8>> for SqlValue {
    fn from(v: Vec<u8>) -> Self {
        Self::Blob(v)
    }
}
impl From<bool> for SqlValue {
    fn from(v: bool) -> Self {
        Self::Integer(v as i64)
    }
}
impl<T: Into<SqlValue>> From<Option<T>> for SqlValue {
    fn from(v: Option<T>) -> Self {
        match v {
            Some(x) => x.into(),
            None => Self::Null,
        }
    }
}

/// Object-safe executor trait. All query results are returned as JSON arrays
/// of arrays, matching the shape SQLite would yield via `json_each`.
pub trait SqlExecutor {
    /// Execute a non-query SQL statement. Returns the number of rows affected.
    fn execute(&self, sql: &str, params: &[SqlValue]) -> Result<usize>;

    /// Run a query and return at most one row, as a JSON array of column values.
    fn query_one_json(&self, sql: &str, params: &[SqlValue]) -> Result<Option<Vec<Value>>>;

    /// Run a query and return all rows, each as a JSON array of column values.
    fn query_all_json(&self, sql: &str, params: &[SqlValue]) -> Result<Vec<Vec<Value>>>;

    fn last_insert_rowid(&self) -> i64;
}

pub fn from_json(v: &Value) -> SqlValue {
    match v {
        Value::Null => SqlValue::Null,
        Value::Bool(b) => SqlValue::Integer(*b as i64),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                SqlValue::Integer(i)
            } else {
                SqlValue::Real(n.as_f64().unwrap_or(0.0))
            }
        }
        Value::String(s) => SqlValue::Text(s.clone()),
        Value::Array(_) | Value::Object(_) => SqlValue::Text(v.to_string()),
    }
}
