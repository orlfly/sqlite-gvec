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

fn sqlite_err(e: rusqlite::Error) -> Error {
    Error::Storage(format!("rusqlite: {e}"))
}

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
    let v: rusqlite::types::Value = row
        .get::<_, rusqlite::types::Value>(i)
        .unwrap_or(rusqlite::types::Value::Null);
    match v {
        rusqlite::types::Value::Null => serde_json::Value::Null,
        rusqlite::types::Value::Integer(n) => serde_json::Value::from(n),
        rusqlite::types::Value::Real(f) => serde_json::json!(f),
        rusqlite::types::Value::Text(s) => serde_json::Value::String(s),
        rusqlite::types::Value::Blob(b) => {
            serde_json::Value::String(format!("<{} bytes>", b.len()))
        }
    }
}

/// Owned handle to a shared SQLite connection, wrapped so we can implement
/// `SqlExecutor` (orphan rule). Cheap to clone (`Arc<Mutex<_>>` inside).
#[derive(Clone)]
pub struct SharedConn(pub Arc<Mutex<Connection>>);

impl SqlExecutor for SharedConn {
    fn execute(&self, sql: &str, p: &[SqlValue]) -> Result<usize> {
        let conn = self
            .0
            .lock()
            .map_err(|e| Error::Storage(format!("mutex: {e}")))?;
        let params: Vec<Box<dyn rusqlite::ToSql>> = p.iter().map(sqlvalue_to_rusqlite).collect();
        let refs: Vec<&dyn rusqlite::ToSql> = params
            .iter()
            .map(|b| b.as_ref() as &dyn rusqlite::ToSql)
            .collect();
        Ok(conn.execute(sql, &refs[..]).map_err(sqlite_err)?)
    }
    fn query_one_json(&self, sql: &str, p: &[SqlValue]) -> Result<Option<Vec<serde_json::Value>>> {
        let conn = self
            .0
            .lock()
            .map_err(|e| Error::Storage(format!("mutex: {e}")))?;
        let mut stmt = conn.prepare(sql).map_err(sqlite_err)?;
        let params: Vec<Box<dyn rusqlite::ToSql>> = p.iter().map(sqlvalue_to_rusqlite).collect();
        let refs: Vec<&dyn rusqlite::ToSql> = params
            .iter()
            .map(|b| b.as_ref() as &dyn rusqlite::ToSql)
            .collect();
        let mut rows = stmt.query(&refs[..]).map_err(sqlite_err)?;
        if let Some(row) = rows.next().map_err(sqlite_err)? {
            let n = row.as_ref().column_count();
            let mut out = Vec::with_capacity(n);
            for i in 0..n {
                out.push(col_to_json(&row, i));
            }
            return Ok(Some(out));
        }
        Ok(None)
    }
    fn query_all_json(&self, sql: &str, p: &[SqlValue]) -> Result<Vec<Vec<serde_json::Value>>> {
        let conn = self
            .0
            .lock()
            .map_err(|e| Error::Storage(format!("mutex: {e}")))?;
        let mut stmt = conn.prepare(sql).map_err(sqlite_err)?;
        let params: Vec<Box<dyn rusqlite::ToSql>> = p.iter().map(sqlvalue_to_rusqlite).collect();
        let refs: Vec<&dyn rusqlite::ToSql> = params
            .iter()
            .map(|b| b.as_ref() as &dyn rusqlite::ToSql)
            .collect();
        let rows = stmt
            .query_map(&refs[..], |row| {
                let n = row.as_ref().column_count();
                let mut out: Vec<serde_json::Value> = Vec::with_capacity(n);
                for i in 0..n {
                    out.push(col_to_json(row, i));
                }
                Ok(out)
            })
            .map_err(sqlite_err)?;
        let mut out = Vec::new();
        for row in rows {
            out.push(row.map_err(sqlite_err)?);
        }
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
        conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA foreign_keys=ON;")
            .map_err(sqlite_err)?;
        Ok(Self {
            conn: SharedConn(Arc::new(Mutex::new(conn))),
        })
    }

    pub fn open_in_memory() -> Result<Self> {
        let conn = Connection::open_in_memory().map_err(sqlite_err)?;
        conn.execute_batch("PRAGMA foreign_keys=ON;")
            .map_err(sqlite_err)?;
        Ok(Self {
            conn: SharedConn(Arc::new(Mutex::new(conn))),
        })
    }

    /// Open or create a named graph. The `Graph` shares the connection with
    /// this `Database` via `Arc`, so both can coexist; the connection lives
    /// as long as either is alive.
    pub fn graph(&self, name: &str) -> Result<Graph> {
        let s = Storage::new(self.conn.clone(), name);
        s.ensure_schema()?;
        Ok(Graph {
            storage: s,
            name: name.to_string(),
        })
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

    pub fn knn(
        &self,
        query: &gvec_core::vector::VectorValue,
        stored: &[(i64, gvec_core::vector::VectorValue)],
        k: usize,
        dist: gvec_core::vector::Distance,
    ) -> Vec<(i64, f32)> {
        gvec_core::vector::knn_search(query, stored, k, dist)
    }

    /// Create an FTS5 text index over `label.property`. Returns an error if the
    /// tokenizer name is not recognized.
    pub fn create_text_index(
        &self,
        label: &str,
        property: &str,
        tokenizer: Option<&str>,
    ) -> Result<()> {
        let t = match tokenizer {
            Some(name) => gvec_core::fts::Tokenizer::parse(name).ok_or_else(|| Error::Parse {
                message: format!("unknown tokenizer: {name}"),
                position: None,
            })?,
            None => gvec_core::fts::Tokenizer::Unicode61,
        };
        gvec_core::fts::create_text_index(
            self.storage.exec_ref(),
            &self.storage.prefix,
            label,
            property,
            &t,
        )
    }

    /// Drop an FTS5 text index.
    pub fn drop_text_index(&self, label: &str, property: &str) -> Result<()> {
        gvec_core::fts::drop_text_index(
            self.storage.exec_ref(),
            &self.storage.prefix,
            label,
            property,
        )
    }

    /// Run a BM25-ranked FTS search. Returns up to `k` `(rowid, score)` pairs.
    pub fn text_search(
        &self,
        label: &str,
        property: &str,
        query: &str,
        k: usize,
    ) -> Result<Vec<(i64, f32)>> {
        gvec_core::fts::text_search(
            self.storage.exec_ref(),
            &self.storage.prefix,
            label,
            property,
            query,
            k,
        )
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
        g.run(r#"CREATE (a:Person {name: "Alice", age: 30})"#)
            .unwrap();
        g.run(r#"CREATE (b:Person {name: "Bob", age: 25})"#)
            .unwrap();
        let rs = g
            .run("MATCH (n:Person) WHERE n.age > 20 RETURN n.name AS name ORDER BY n.age ASC")
            .unwrap();
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
        let rs = g
            .run(r#"MATCH (n:Person {name: "Alice"}) RETURN n.name AS name"#)
            .unwrap();
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
        let ab = g
            .run("MATCH (a:Person {name: \"A\"})-[:KNOWS]->(b) RETURN b.name AS name")
            .unwrap();
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

    #[test]
    fn fts_filters_rows_by_rowid_set() {
        // Regression: text_match must actually filter rows by the FTS rowid set,
        // not pass everything through. Previously Expr::TextMatch returned true,
        // so a query asking for "rust" would also return nodes with non-matching bodies.
        let db = Database::open_in_memory().unwrap();
        let g = db.graph("test").unwrap();
        g.run(r#"CREATE (a:Doc {title: "Rust guide", body: "the rust programming language"})"#)
            .unwrap();
        g.run(
            r#"CREATE (b:Doc {title: "Python guide", body: "python is dynamic and high-level"})"#,
        )
        .unwrap();
        g.run(r#"CREATE (c:Doc {title: "Rust async", body: "rust async runtime and tokio"})"#)
            .unwrap();
        g.create_text_index("Doc", "body", None).unwrap();
        let rs = g.run(r#"MATCH (n:Doc) WHERE text_match(n.body, "rust") RETURN n.title AS title ORDER BY title ASC"#).unwrap();
        assert_eq!(
            rs.rows.len(),
            2,
            "expected only the 2 rust rows, got: {:?}",
            rs.rows
        );
        let titles: Vec<String> = rs
            .rows
            .iter()
            .map(|r| r[0].as_str().unwrap_or("").to_string())
            .collect();
        assert!(titles.contains(&"Rust async".to_string()));
        assert!(titles.contains(&"Rust guide".to_string()));
        assert!(!titles.contains(&"Python guide".to_string()));
    }

    #[test]
    fn fts_rejects_unsafe_identifier() {
        // Identifier validation must reject anything that's not a plain SQL
        // identifier (alphanumeric + underscore, no quotes/dashes).
        let db = Database::open_in_memory().unwrap();
        let g = db.graph("test").unwrap();
        g.run(r#"CREATE (a:D {body: "x"})"#).unwrap();
        let r = g.create_text_index("D", "body\"; DROP TABLE D_nodes;--", None);
        assert!(
            r.is_err(),
            "expected identifier validation to reject injection payload"
        );
    }

    #[test]
    fn fts_create_and_search() {
        let db = Database::open_in_memory().unwrap();
        let g = db.graph("test").unwrap();
        g.run(r#"CREATE (a:Doc {title: "Rust database", body: "the rust database is fast"})"#)
            .unwrap();
        g.run(r#"CREATE (b:Doc {title: "Python", body: "python is dynamic"})"#)
            .unwrap();
        g.run(r#"CREATE (c:Doc {title: "Rust async", body: "rust async runtime"})"#)
            .unwrap();
        g.create_text_index("Doc", "body", None).unwrap();
        let res = g.text_search("Doc", "body", "rust", 10).unwrap();
        assert_eq!(res.len(), 2, "expected 2 rust hits");
        // higher score = more relevant
        let rowids: std::collections::HashSet<i64> = res.iter().map(|(id, _)| *id).collect();
        assert!(rowids.contains(&1));
        assert!(rowids.contains(&3));
    }

    #[test]
    fn fts_create_text_index_via_cypher() {
        let db = Database::open_in_memory().unwrap();
        let g = db.graph("test").unwrap();
        g.run(r#"CREATE (a:Doc {title: "Hello", body: "the rust programming language"})"#)
            .unwrap();
        g.run(r#"CREATE TEXT INDEX ON :Doc(body)"#).unwrap();
        let rs = g.run(r#"MATCH (n:Doc) WHERE text_match(n.body, "rust") RETURN n.title AS title, text_score(n) AS score ORDER BY score DESC"#).unwrap();
        assert_eq!(rs.rows.len(), 1);
        assert_eq!(rs.rows[0][0], json!("Hello"));
        assert!(rs.rows[0][1].as_f64().unwrap() > 0.0);
    }

    /// Calling `CREATE TEXT INDEX` twice for the same (label, property) must
    /// not fail with a rowid UNIQUE conflict: the backfill uses
    /// `INSERT OR REPLACE`, so the second call re-indexes in place and the
    /// index stays consistent with the current node state.
    #[test]
    fn fts_create_text_index_via_cypher_idempotent() {
        let db = Database::open_in_memory().unwrap();
        let g = db.graph("test").unwrap();
        g.run(r#"CREATE (a:Doc {title: "Hello", body: "the rust programming language"})"#)
            .unwrap();
        g.run(r#"CREATE (b:Doc {title: "World", body: "python is dynamic"})"#)
            .unwrap();

        // First call: creates the index.
        g.run(r#"CREATE TEXT INDEX ON :Doc(body)"#).unwrap();
        // Second call: must not fail and must leave the index usable.
        g.run(r#"CREATE TEXT INDEX ON :Doc(body)"#).unwrap();

        let res = g.text_search("Doc", "body", "rust", 10).unwrap();
        assert_eq!(res.len(), 1, "expected only the rust Doc after re-create");
        let rowids: std::collections::HashSet<i64> = res.iter().map(|(id, _)| *id).collect();
        assert!(rowids.contains(&1), "rowid 1 should match rust");

        // text_match via WHERE still works through Cypher after the second
        // CREATE TEXT INDEX.
        let rs = g
            .run(r#"MATCH (n:Doc) WHERE text_match(n.body, "rust") RETURN n.title AS title"#)
            .unwrap();
        assert_eq!(rs.rows.len(), 1);
        assert_eq!(rs.rows[0][0], json!("Hello"));
    }

    /// Calling `create_text_index` twice via the Rust binding must also be
    /// idempotent. This exercises the same INSERT OR REPLACE path directly,
    /// independent of the Cypher parser, and guards against the (prefix,
    /// label, property) UNIQUE conflict in any caller.
    #[test]
    fn fts_create_text_index_idempotent() {
        let db = Database::open_in_memory().unwrap();
        let g = db.graph("test").unwrap();
        g.run(r#"CREATE (a:Doc {body: "hello world"})"#).unwrap();
        g.run(r#"CREATE (b:Doc {body: "hello rust"})"#).unwrap();

        g.create_text_index("Doc", "body", None).unwrap();
        g.create_text_index("Doc", "body", None).unwrap();
        g.create_text_index("Doc", "body", Some("porter")).unwrap();

        let res = g.text_search("Doc", "body", "hello", 10).unwrap();
        assert_eq!(res.len(), 2, "expected 2 hello hits after re-index");
        let rowids: std::collections::HashSet<i64> = res.iter().map(|(id, _)| *id).collect();
        assert!(rowids.contains(&1));
        assert!(rowids.contains(&2));
    }

    /// `DROP TEXT INDEX` followed by `CREATE TEXT INDEX` for the same tuple
    /// must rebuild a working FTS5 index from scratch. This protects users
    /// who use DROP+CREATE as a deliberate re-index path.
    #[test]
    fn fts_drop_then_create_text_index_via_cypher() {
        let db = Database::open_in_memory().unwrap();
        let g = db.graph("test").unwrap();
        g.run(r#"CREATE (a:Doc {body: "rust first"})"#).unwrap();
        g.run(r#"CREATE TEXT INDEX ON :Doc(body)"#).unwrap();
        g.run("DROP TEXT INDEX ON :Doc(body)").unwrap();
        // Re-create with a different tokenizer; this also re-indexes the
        // existing node so search finds it again.
        g.run(r#"CREATE TEXT INDEX ON :Doc(body) USING fts5(porter)"#)
            .unwrap();
        let res = g.text_search("Doc", "body", "rust", 5).unwrap();
        assert_eq!(res.len(), 1);
    }

    #[test]
    fn fts_phrase_match() {
        let db = Database::open_in_memory().unwrap();
        let g = db.graph("test").unwrap();
        g.run(r#"CREATE (a:Doc {body: "the rust database is fast"})"#)
            .unwrap();
        g.run(r#"CREATE (b:Doc {body: "rust is the language database users pick"})"#)
            .unwrap();
        g.create_text_index("Doc", "body", None).unwrap();
        // Phrase match (quoted in FTS5)
        let res = g
            .text_search("Doc", "body", "\"rust database\"", 10)
            .unwrap();
        assert_eq!(
            res.len(),
            1,
            "phrase should match only node with the exact words in sequence"
        );
    }

    #[test]
    fn fts_drop_index() {
        let db = Database::open_in_memory().unwrap();
        let g = db.graph("test").unwrap();
        g.run(r#"CREATE (a:Doc {body: "hello"})"#).unwrap();
        g.create_text_index("Doc", "body", None).unwrap();
        g.drop_text_index("Doc", "body").unwrap();
        // After drop, text_search should fail (table doesn't exist)
        let r = g.text_search("Doc", "body", "hello", 5);
        assert!(r.is_err(), "expected error after dropping index");
    }

    #[test]
    fn fts_create_and_drop_via_cypher() {
        let db = Database::open_in_memory().unwrap();
        let g = db.graph("test").unwrap();
        g.run(r#"CREATE (a:Doc {body: "rust rocks"})"#).unwrap();
        g.run(r#"CREATE TEXT INDEX ON :Doc(body) USING fts5(porter)"#)
            .unwrap();
        let res = g.text_search("Doc", "body", "rust", 5).unwrap();
        assert_eq!(res.len(), 1);
        g.run("DROP TEXT INDEX ON :Doc(body)").unwrap();
        let r = g.text_search("Doc", "body", "rust", 5);
        assert!(r.is_err());
    }

    /// Loads the compiled `libgvec_extension.so` into a fresh connection
    /// and verifies the entry point + scalar functions are registered.
    /// Skips if the extension .so is not built.
    #[test]
    fn load_extension_end_to_end() {
        use std::path::PathBuf;
        let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
        let workspace = PathBuf::from(manifest_dir)
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .to_path_buf();
        let candidate = workspace
            .join("target")
            .join("debug")
            .join("libgvec_extension.so");
        if !candidate.exists() {
            eprintln!("skip: extension not built");
            return;
        }
        let conn = Connection::open_in_memory().unwrap();
        unsafe { conn.load_extension_enable() }.unwrap();
        unsafe { conn.load_extension(candidate.to_str().unwrap(), Some("sqlite3_gvec_init")) }
            .unwrap();
        let v: String = conn
            .query_row("SELECT gvec_version()", [], |r| r.get(0))
            .unwrap();
        assert!(v.starts_with('v'), "got {v}");
        let d: f64 = conn
            .query_row(
                "SELECT gvec_dist('[1.0, 0.0]', '[3.0, 0.0]', 'l2')",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert!((d - 2.0).abs() < 1e-6);
        let err: String = conn
            .query_row("SELECT gvec_query('CREATE (n:Person {})')", [], |r| {
                r.get(0)
            })
            .unwrap_or_else(|e| e.to_string());
        assert!(
            err.contains("Rust binding crate"),
            "expected helpful error, got: {err}"
        );
        let err: String = conn
            .query_row(
                "SELECT gvec_text_search('g', 'Doc', 'body', 'q', 5)",
                [],
                |r| r.get(0),
            )
            .unwrap_or_else(|e| e.to_string());
        assert!(
            err.contains("Rust binding crate"),
            "expected helpful error, got: {err}"
        );
    }

    /// WAL-mode concurrency smoke: a writer creates nodes in a transaction
    /// while a reader concurrently runs MATCH queries and sees consistent
    /// snapshots.
    #[test]
    fn set_edge_property() {
        let db = Database::open_in_memory().unwrap();
        let g = db.graph("test").unwrap();
        g.run(r#"CREATE (a:Person {name: "A"})-[:KNOWS]->(b:Person {name: "B"})"#)
            .unwrap();
        // SET on a relationship variable
        g.run("MATCH (a)-[r:KNOWS]->(b) SET r.weight = 0.9")
            .unwrap();
        let rs = g
            .run("MATCH (a)-[r:KNOWS]->(b) RETURN r.weight AS w")
            .unwrap();
        assert_eq!(rs.rows.len(), 1);
        assert_eq!(rs.rows[0][0], json!(0.9));
    }

    #[test]
    fn set_node_property() {
        let db = Database::open_in_memory().unwrap();
        let g = db.graph("test").unwrap();
        g.run(r#"CREATE (a:Person {name: "Alice", age: 30})"#)
            .unwrap();
        g.run(r#"MATCH (n:Person {name: "Alice"}) SET n.age = 31"#)
            .unwrap();
        let rs = g
            .run(r#"MATCH (n:Person {name: "Alice"}) RETURN n.age AS age"#)
            .unwrap();
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
        let rs = g_r
            .run("MATCH (n:Person) RETURN n.name AS name ORDER BY n.age ASC")
            .unwrap();
        assert_eq!(rs.rows.len(), 2);
    }
}
