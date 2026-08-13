//! `gvec`: the in-process Rust binding for the gvec engine.
//!
//! Embeds SQLite via `rusqlite` and exposes graph + vector operations as a
//! pure Rust API. No FFI to the host — the entire engine lives in this crate.

use gvec_core::error::{Error, Result};
use gvec_core::sql::{SqlExecutor, SqlValue};
use gvec_core::storage::Storage;
use rusqlite::Connection;
use std::sync::{Arc, Mutex};

pub const VERSION: &str = concat!("v", env!("CARGO_PKG_VERSION"));

fn sqlite_err(e: rusqlite::Error) -> Error { Error::Storage(format!("rusqlite: {e}")) }

fn sqlvalue_to_rusqlite(p: &SqlValue) -> Box<dyn rusqlite::ToSql> {
    match p {
        SqlValue::Null => Box::new(rusqlite::types::Null),
        SqlValue::Integer(n) => Box::new(*n),
        SqlValue::Real(f) => Box::new(*f),
        SqlValue::Text(s) => Box::new(s.clone()),
        SqlValue::Blob(b) => Box::new(b.clone()),
    }
}

fn col_to_json(row: &rusqlite::Row<'_>, i: usize) -> serde_json::Value {
    let v: rusqlite::types::Value = row.get::<_, rusqlite::types::Value>(i).unwrap_or(rusqlite::types::Value::Null);
    match v {
        rusqlite::types::Value::Null => serde_json::Value::Null,
        rusqlite::types::Value::Integer(n) => serde_json::Value::from(n),
        rusqlite::types::Value::Real(f) => serde_json::json!(f),
        rusqlite::types::Value::Text(s) => serde_json::Value::String(s),
        rusqlite::types::Value::Blob(b) => serde_json::Value::String(format!("<{} bytes>", b.len())),
    }
}

/// Owned handle to a shared SQLite connection, wrapped so we can implement
/// `SqlExecutor` (orphan rule). Cheap to clone (`Arc<Mutex<_>>` inside).
#[derive(Clone)]
pub struct SharedConn(pub Arc<Mutex<Connection>>);

impl SqlExecutor for SharedConn {
    fn execute(&self, sql: &str, p: &[SqlValue]) -> Result<usize> {
        let conn = self.0.lock().map_err(|e| Error::Storage(format!("mutex: {e}")))?;
        let params: Vec<Box<dyn rusqlite::ToSql>> = p.iter().map(sqlvalue_to_rusqlite).collect();
        let refs: Vec<&dyn rusqlite::ToSql> = params.iter().map(|b| b.as_ref() as &dyn rusqlite::ToSql).collect();
        Ok(conn.execute(sql, &refs[..]).map_err(sqlite_err)?)
    }
    fn query_one_json(&self, sql: &str, p: &[SqlValue]) -> Result<Option<Vec<serde_json::Value>>> {
        let conn = self.0.lock().map_err(|e| Error::Storage(format!("mutex: {e}")))?;
        let mut stmt = conn.prepare(sql).map_err(sqlite_err)?;
        let params: Vec<Box<dyn rusqlite::ToSql>> = p.iter().map(sqlvalue_to_rusqlite).collect();
        let refs: Vec<&dyn rusqlite::ToSql> = params.iter().map(|b| b.as_ref() as &dyn rusqlite::ToSql).collect();
        let mut rows = stmt.query(&refs[..]).map_err(sqlite_err)?;
        if let Some(row) = rows.next().map_err(sqlite_err)? {
            let n = row.as_ref().column_count();
            let mut out = Vec::with_capacity(n);
            for i in 0..n { out.push(col_to_json(&row, i)); }
            return Ok(Some(out));
        }
        Ok(None)
    }
    fn query_all_json(&self, sql: &str, p: &[SqlValue]) -> Result<Vec<Vec<serde_json::Value>>> {
        let conn = self.0.lock().map_err(|e| Error::Storage(format!("mutex: {e}")))?;
        let mut stmt = conn.prepare(sql).map_err(sqlite_err)?;
        let params: Vec<Box<dyn rusqlite::ToSql>> = p.iter().map(sqlvalue_to_rusqlite).collect();
        let refs: Vec<&dyn rusqlite::ToSql> = params.iter().map(|b| b.as_ref() as &dyn rusqlite::ToSql).collect();
        let rows = stmt.query_map(&refs[..], |row| {
            let n = row.as_ref().column_count();
            let mut out: Vec<serde_json::Value> = Vec::with_capacity(n);
            for i in 0..n { out.push(col_to_json(row, i)); }
            Ok(out)
        }).map_err(sqlite_err)?;
        let mut out = Vec::new();
        for row in rows { out.push(row.map_err(sqlite_err)?); }
        Ok(out)
    }
    fn last_insert_rowid(&self) -> i64 {
        self.0.lock().map(|c| c.last_insert_rowid()).unwrap_or(0)
    }
}

/// A graph database backed by a shared `rusqlite::Connection`.
pub struct Database {
    pub conn: SharedConn,
}

impl Database {
    pub fn open(path: &str) -> Result<Self> {
        let conn = Connection::open(path).map_err(sqlite_err)?;
        conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA foreign_keys=ON;").map_err(sqlite_err)?;
        Ok(Self { conn: SharedConn(Arc::new(Mutex::new(conn))) })
    }

    pub fn open_in_memory() -> Result<Self> {
        let conn = Connection::open_in_memory().map_err(sqlite_err)?;
        conn.execute_batch("PRAGMA foreign_keys=ON;").map_err(sqlite_err)?;
        Ok(Self { conn: SharedConn(Arc::new(Mutex::new(conn))) })
    }

    /// Open or create a named graph. The `Graph` shares the connection with
    /// this `Database` via `Arc`, so both can coexist; the connection lives
    /// as long as either is alive.
    pub fn graph(&self, name: &str) -> Result<Graph> {
        let s = Storage::new(self.conn.clone(), name);
        s.ensure_schema()?;
        Ok(Graph { storage: s, name: name.to_string() })
    }
}

pub struct Graph {
    pub storage: Storage,
    pub name: String,
}

impl Graph {
    /// Run a Cypher-like query.
    pub fn run(&self, cypher: &str) -> Result<gvec_core::query::ResultSet> {
        let mut p = gvec_core::query::Parser::new(cypher).map_err(Error::from)?;
        let q = p.parse_query().map_err(Error::from)?;
        gvec_core::query::execute(&self.storage, &q).map_err(Error::from)
    }

    pub fn knn(&self, query: &gvec_core::vector::VectorValue, stored: &[(i64, gvec_core::vector::VectorValue)], k: usize, dist: gvec_core::vector::Distance) -> Vec<(i64, f32)> {
        gvec_core::vector::knn_search(query, stored, k, dist)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gvec_core::vector::{Distance, VectorValue};
    use serde_json::json;

    #[test]
    fn auto_load_and_match() {
        let db = Database::open_in_memory().unwrap();
        let g = db.graph("test").unwrap();
        g.run(r#"CREATE (a:Person {name: "Alice", age: 30})"#).unwrap();
        g.run(r#"CREATE (b:Person {name: "Bob", age: 25})"#).unwrap();
        let rs = g.run("MATCH (n:Person) WHERE n.age > 20 RETURN n.name AS name ORDER BY n.age ASC").unwrap();
        assert_eq!(rs.rows.len(), 2);
        assert_eq!(rs.rows[0][0], json!("Bob"));
        assert_eq!(rs.rows[1][0], json!("Alice"));
    }

    #[test]
    fn inline_property_filter() {
        let db = Database::open_in_memory().unwrap();
        let g = db.graph("test").unwrap();
        g.run(r#"CREATE (a:Person {name: "Alice"})"#).unwrap();
        g.run(r#"CREATE (b:Person {name: "Bob"})"#).unwrap();
        let rs = g.run(r#"MATCH (n:Person {name: "Alice"}) RETURN n.name AS name"#).unwrap();
        assert_eq!(rs.rows.len(), 1);
        assert_eq!(rs.rows[0][0], json!("Alice"));
    }

    #[test]
    fn variable_length_path() {
        let db = Database::open_in_memory().unwrap();
        let g = db.graph("test").unwrap();
        // Create A -> B -> C in one query so the chain is connected
        g.run(r#"CREATE (a:Person {name: "A"})-[:KNOWS]->(b:Person {name: "B"})-[:KNOWS]->(c:Person {name: "C"})"#).unwrap();
        // directed 1-hop
        let ab = g.run("MATCH (a:Person {name: \"A\"})-[:KNOWS]->(b) RETURN b.name AS name").unwrap();
        assert_eq!(ab.rows.len(), 1);
        assert_eq!(ab.rows[0][0], json!("B"));
        // variable-length 1..2 from A
        let reachable = g.run("MATCH (a:Person {name: \"A\"})-[:KNOWS*1..2]->(b) RETURN b.name AS name ORDER BY name ASC").unwrap();
        let names: Vec<serde_json::Value> = reachable.rows.iter().map(|r| r[0].clone()).collect();
        assert!(names.contains(&json!("B")), "must contain B: {:?}", names);
        assert!(names.contains(&json!("C")), "must contain C: {:?}", names);
    }

    #[test]
    fn knn_query() {
        let db = Database::open_in_memory().unwrap();
        let g = db.graph("test").unwrap();
        let stored = vec![
            (1, VectorValue::F32(vec![1.0, 0.0])),
            (2, VectorValue::F32(vec![3.0, 0.0])),
            (3, VectorValue::F32(vec![0.0, 0.0])),
        ];
        let q = VectorValue::F32(vec![0.0, 0.0]);
        let res = g.knn(&q, &stored, 2, Distance::L2);
        assert_eq!(res[0].0, 3);
        assert_eq!(res[1].0, 1);
    }

    /// Loads the compiled `libgvec_extension.so` into a fresh connection
    /// and verifies the entry point + scalar functions are registered.
    /// Skips if the extension .so is not built.
    #[test]
    fn load_extension_end_to_end() {
        use std::path::PathBuf;
        let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
        let workspace = PathBuf::from(manifest_dir).parent().unwrap().parent().unwrap().to_path_buf();
        let candidate = workspace.join("target").join("debug").join("libgvec_extension.so");
        if !candidate.exists() { eprintln!("skip: extension not built"); return; }
        let conn = Connection::open_in_memory().unwrap();
        unsafe { conn.load_extension_enable() }.unwrap();
        unsafe { conn.load_extension(candidate.to_str().unwrap(), Some("sqlite3_gvec_init")) }.unwrap();
        let v: String = conn.query_row("SELECT gvec_version()", [], |r| r.get(0)).unwrap();
        assert!(v.starts_with('v'), "got {v}");
        let d: f64 = conn.query_row("SELECT gvec_dist('[1.0, 0.0]', '[3.0, 0.0]', 'l2')", [], |r| r.get(0)).unwrap();
        assert!((d - 2.0).abs() < 1e-6);
        let err: String = conn.query_row("SELECT gvec_query('CREATE (n:Person {})')", [], |r| r.get(0)).unwrap_or_else(|e| e.to_string());
        assert!(err.contains("Rust binding crate"), "expected helpful error, got: {err}");
    }

    /// WAL-mode concurrency smoke: a writer creates nodes in a transaction
    /// while a reader concurrently runs MATCH queries and sees consistent
    /// snapshots.
    #[test]
    fn set_edge_property() {
        let db = Database::open_in_memory().unwrap();
        let g = db.graph("test").unwrap();
        g.run(r#"CREATE (a:Person {name: "A"})-[:KNOWS]->(b:Person {name: "B"})"#).unwrap();
        // SET on a relationship variable
        g.run("MATCH (a)-[r:KNOWS]->(b) SET r.weight = 0.9").unwrap();
        let rs = g.run("MATCH (a)-[r:KNOWS]->(b) RETURN r.weight AS w").unwrap();
        assert_eq!(rs.rows.len(), 1);
        assert_eq!(rs.rows[0][0], json!(0.9));
    }

    #[test]
    fn set_node_property() {
        let db = Database::open_in_memory().unwrap();
        let g = db.graph("test").unwrap();
        g.run(r#"CREATE (a:Person {name: "Alice", age: 30})"#).unwrap();
        g.run(r#"MATCH (n:Person {name: "Alice"}) SET n.age = 31"#).unwrap();
        let rs = g.run(r#"MATCH (n:Person {name: "Alice"}) RETURN n.age AS age"#).unwrap();
        assert_eq!(rs.rows[0][0], json!(31));
    }

    #[test]
    fn unlabeled_match() {
        let db = Database::open_in_memory().unwrap();
        let g = db.graph("test").unwrap();
        g.run(r#"CREATE (n {kind: "root"})"#).unwrap();
        let rs = g.run("MATCH (n) RETURN n.kind AS kind").unwrap();
        assert_eq!(rs.rows.len(), 1);
        assert_eq!(rs.rows[0][0], json!("root"));
    }

    #[test]
    fn multi_pattern_cross_product() {
        let db = Database::open_in_memory().unwrap();
        let g = db.graph("test").unwrap();
        g.run(r#"CREATE (a:Person {name: "A"})"#).unwrap();
        g.run(r#"CREATE (b:Person {name: "B"})"#).unwrap();
        let rs = g.run("MATCH (a:Person), (b:Person) RETURN a.name AS a_name, b.name AS b_name ORDER BY a_name ASC, b_name ASC").unwrap();
        assert_eq!(rs.rows.len(), 4, "cross product of 2×2 = 4: {:?}", rs.rows);
    }

    #[test]
    fn wal_concurrent_reader_writer() {
        use tempfile::tempdir;
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.db").to_path_buf();

        let writer = Database::open(path.to_str().unwrap()).unwrap();
        let g_w = writer.graph("shared").unwrap();
        g_w.run("CREATE (a:Person {name: \"A\", age: 1})").unwrap();

        let reader = Database::open(path.to_str().unwrap()).unwrap();
        let g_r = reader.graph("shared").unwrap();
        let rs = g_r.run("MATCH (n:Person) RETURN n.name AS name").unwrap();
        assert_eq!(rs.rows.len(), 1);

        g_w.run("CREATE (b:Person {name: \"B\", age: 2})").unwrap();
        let rs = g_r.run("MATCH (n:Person) RETURN n.name AS name ORDER BY n.age ASC").unwrap();
        assert_eq!(rs.rows.len(), 2);
    }
}