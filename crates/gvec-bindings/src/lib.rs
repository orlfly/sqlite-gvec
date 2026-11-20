//! `gvec`: the in-process Rust binding for the gvec engine.
//!
//! Embeds SQLite via `rusqlite` and exposes graph + vector operations as a
//! pure Rust API. No FFI to the host — the entire engine lives in this crate.

pub use gvec_core::error::{Error, Result};
use gvec_core::sql::{SqlExecutor, SqlValue};
use gvec_core::storage::Storage;
pub use gvec_core::storage::VectorIndexMeta;
pub use gvec_core::vector::{Distance, ElementType, VectorBlob, VectorValue};
pub use gvec_core::vector_index::{
    FlatIndex, HnswIndex, HnswParams, IvfIndex, IvfParams, VectorIndex, VectorIndexAlgorithm,
    VectorIndexConfig,
};
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
        conn.execute(sql, &refs[..]).map_err(sqlite_err)
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
                out.push(col_to_json(row, i));
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
        let mut p = gvec_core::query::Parser::new(cypher)?;
        let q = p.parse_query()?;
        gvec_core::query::execute(&self.storage, &q)
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

    /// Create (or reconfigure and reindex) an ANN vector index over
    /// `label.property`.
    ///
    /// `algorithm` is `flat`, `hnsw`, or `ivf`. `metric` is `l2`, `cosine`, or
    /// `ip`. `options` is an optional JSON object with algorithm parameters
    /// (`m`, `ef_construction`, `ef_search`, `nlist`, `nprobe`).
    pub fn create_vector_index(
        &self,
        label: &str,
        property: &str,
        algorithm: &str,
        metric: Option<&str>,
        options: Option<serde_json::Value>,
    ) -> Result<VectorIndexMeta> {
        let mut opts: Vec<(String, serde_json::Value)> = Vec::new();
        if let Some(serde_json::Value::Object(map)) = options {
            for (k, v) in map {
                opts.push((k, v));
            }
        }
        if let Some(metric) = metric {
            opts.retain(|(k, _)| k != "metric");
            opts.push(("metric".into(), serde_json::Value::String(metric.into())));
        }
        self.storage
            .create_vector_index(label, property, algorithm, &opts)
    }

    pub fn drop_vector_index(&self, label: &str, property: &str) -> Result<()> {
        self.storage.drop_vector_index(label, property)
    }

    pub fn list_vector_indexes(&self) -> Result<Vec<VectorIndexMeta>> {
        self.storage.vector_indexes()
    }

    pub fn vector_count(&self, label: &str, property: &str) -> Result<usize> {
        self.storage.vector_count(label, property)
    }

    /// ANN search over `label.property`. Uses the configured index (HNSW by
    /// default) and falls back to an exact L2 scan when no index exists.
    pub fn vector_search(
        &self,
        label: &str,
        property: &str,
        query: &VectorValue,
        k: usize,
    ) -> Result<Vec<(i64, f32)>> {
        self.storage.vector_search(label, property, query, k)
    }

    /// Exact scan using the requested metric, useful for recall checks and
    /// when the user has not created an index.
    pub fn exact_vector_search(
        &self,
        label: &str,
        property: &str,
        query: &VectorValue,
        k: usize,
        metric: Distance,
    ) -> Result<Vec<(i64, f32)>> {
        self.storage
            .exact_vector_search(label, property, query, k, metric)
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

    // ---- Vector index (ANN) integration tests ----

    #[test]
    fn vector_index_knn_via_cypher() {
        let db = Database::open_in_memory().unwrap();
        let g = db.graph("test").unwrap();
        g.run(r#"CREATE (a:Doc {title: "origin", embedding: [0.0, 0.0]})"#)
            .unwrap();
        g.run(r#"CREATE (b:Doc {title: "near", embedding: [1.0, 0.0]})"#)
            .unwrap();
        g.run(r#"CREATE (c:Doc {title: "far", embedding: [5.0, 5.0]})"#)
            .unwrap();
        g.create_vector_index("Doc", "embedding", "hnsw", Some("l2"), None)
            .unwrap();

        let rs = g
            .run(
                r#"MATCH (n:Doc) WHERE vector_nn(n.embedding, [0.0, 0.0], 2)
                   RETURN n.title AS title, vector_distance(n) AS distance"#,
            )
            .unwrap();
        assert_eq!(rs.rows.len(), 2, "rows: {:?}", rs.rows);
        assert_eq!(rs.rows[0][0], json!("origin"));
        assert_eq!(rs.rows[1][0], json!("near"));
        let d0 = rs.rows[0][1].as_f64().unwrap();
        let d1 = rs.rows[1][1].as_f64().unwrap();
        assert!(d0 < d1, "expected ascending distance, got {d0} then {d1}");
    }

    #[test]
    fn vector_index_is_maintained_on_node_create_update_delete() {
        let db = Database::open_in_memory().unwrap();
        let g = db.graph("test").unwrap();
        // Create the index before any node exists: vectors must be picked up
        // by subsequent inserts.
        g.create_vector_index("Doc", "embedding", "flat", None, None)
            .unwrap();
        assert_eq!(g.vector_count("Doc", "embedding").unwrap(), 0);

        g.run(r#"CREATE (a:Doc {title: "a", embedding: [1.0, 0.0]})"#)
            .unwrap();
        g.run(r#"CREATE (b:Doc {title: "b", embedding: [2.0, 0.0]})"#)
            .unwrap();
        assert_eq!(g.vector_count("Doc", "embedding").unwrap(), 2);

        // SET updates the stored vector through Storage::update_node.
        g.run(r#"MATCH (n:Doc {title: "a"}) SET n.embedding = [9.0, 9.0]"#)
            .unwrap();
        let rs = g
            .vector_search("Doc", "embedding", &VectorValue::F32(vec![9.0, 9.0]), 1)
            .unwrap();
        assert_eq!(rs[0].0, 1, "updated node a should be nearest");

        // DELETE removes the vector row.
        g.run(r#"MATCH (n:Doc {title: "a"}) DELETE n"#).unwrap();
        assert_eq!(g.vector_count("Doc", "embedding").unwrap(), 1);
    }

    #[test]
    fn vector_index_persists_across_reopen() {
        use tempfile::tempdir;
        let dir = tempdir().unwrap();
        let path = dir.path().join("vectors.db");
        let path = path.to_str().unwrap().to_string();

        {
            let db = Database::open(&path).unwrap();
            let g = db.graph("persist").unwrap();
            g.run(r#"CREATE (a:Doc {title: "a", embedding: [0.0, 0.0]})"#)
                .unwrap();
            g.run(r#"CREATE (b:Doc {title: "b", embedding: [4.0, 4.0]})"#)
                .unwrap();
            g.create_vector_index("Doc", "embedding", "hnsw", Some("l2"), None)
                .unwrap();
            assert_eq!(g.vector_count("Doc", "embedding").unwrap(), 2);
        }

        let db = Database::open(&path).unwrap();
        let g = db.graph("persist").unwrap();
        let indexes = g.list_vector_indexes().unwrap();
        assert_eq!(indexes.len(), 1);
        assert_eq!(indexes[0].algorithm, "hnsw");
        assert_eq!(g.vector_count("Doc", "embedding").unwrap(), 2);

        let rs = g
            .run(
                r#"MATCH (n:Doc) WHERE vector_nn(n.embedding, [0.1, 0.1], 1)
                   RETURN n.title AS title"#,
            )
            .unwrap();
        assert_eq!(rs.rows.len(), 1);
        assert_eq!(rs.rows[0][0], json!("a"));
    }

    #[test]
    fn vector_index_ddl_ivf_and_drop() {
        let db = Database::open_in_memory().unwrap();
        let g = db.graph("test").unwrap();
        for i in 0..8 {
            g.run(&format!(
                r#"CREATE (n:Doc {{i: {i}, embedding: [{i}.0, {}]}})"#,
                (i * 2) as f32
            ))
            .unwrap();
        }
        g.run(
            r#"CREATE VECTOR INDEX ON :Doc(embedding)
               USING IVF(nlist = 4, nprobe = 2, metric = "cosine")"#,
        )
        .unwrap();
        let indexes = g.list_vector_indexes().unwrap();
        assert_eq!(indexes.len(), 1);
        assert_eq!(indexes[0].algorithm, "ivf");
        assert_eq!(indexes[0].metric, "cosine");
        assert_eq!(indexes[0].dim, Some(2));
        assert_eq!(g.vector_count("Doc", "embedding").unwrap(), 8);

        // DDL-created indexes are used by vector_nn.
        let rs = g
            .run(
                r#"MATCH (n:Doc) WHERE vector_nn(n.embedding, [0.0, 0.0], 2)
                   RETURN n.i AS i"#,
            )
            .unwrap();
        assert_eq!(rs.rows.len(), 2);

        g.run("DROP VECTOR INDEX ON :Doc(embedding)").unwrap();
        assert!(g.list_vector_indexes().unwrap().is_empty());
        // Without an index, vector_nn falls back to an exact scan and still
        // works for small data sets.
        let rs = g
            .run(
                r#"MATCH (n:Doc) WHERE vector_nn(n.embedding, [0.0, 0.0], 1)
                   RETURN n.i AS i"#,
            )
            .unwrap();
        assert_eq!(rs.rows.len(), 1);
    }

    #[test]
    fn vector_index_hybrid_text_and_vector() {
        let db = Database::open_in_memory().unwrap();
        let g = db.graph("test").unwrap();
        g.run(r#"CREATE (a:Doc {title: "rust db", body: "the rust database is fast", embedding: [1.0, 0.0]})"#).unwrap();
        g.run(r#"CREATE (b:Doc {title: "rust async", body: "rust async runtime", embedding: [0.9, 0.1]})"#).unwrap();
        g.run(
            r#"CREATE (c:Doc {title: "python", body: "python is dynamic", embedding: [0.0, 1.0]})"#,
        )
        .unwrap();
        g.create_text_index("Doc", "body", None).unwrap();
        g.create_vector_index("Doc", "embedding", "hnsw", Some("l2"), None)
            .unwrap();

        let rs = g
            .run(
                r#"MATCH (n:Doc)
                   WHERE text_match(n.body, "rust") AND vector_nn(n.embedding, [1.0, 0.0], 3)
                   RETURN n.title AS title, text_score(n) AS score, vector_distance(n) AS distance
                   ORDER BY distance ASC"#,
            )
            .unwrap();
        assert_eq!(rs.rows.len(), 2, "rows: {:?}", rs.rows);
        let titles: Vec<&str> = rs.rows.iter().map(|r| r[0].as_str().unwrap()).collect();
        assert_eq!(titles, vec!["rust db", "rust async"]);
        assert!(rs.rows[0][1].as_f64().unwrap() > 0.0);
        assert!(rs.rows[0][2].as_f64().unwrap() <= rs.rows[1][2].as_f64().unwrap());
    }

    #[test]
    fn vector_index_rejects_dimension_mismatch() {
        let db = Database::open_in_memory().unwrap();
        let g = db.graph("test").unwrap();
        g.run(r#"CREATE (a:Doc {title: "a", embedding: [1.0, 0.0]})"#)
            .unwrap();
        g.run(r#"CREATE (b:Doc {title: "b", embedding: [0.0, 1.0]})"#)
            .unwrap();
        g.create_vector_index("Doc", "embedding", "hnsw", None, None)
            .unwrap();

        let err = g
            .run(r#"CREATE (c:Doc {title: "c", embedding: [1.0, 2.0, 3.0]})"#)
            .unwrap_err();
        assert!(format!("{err}").contains("dimension"), "got: {err}");

        // The invalid node must not have been written.
        let rs = g.run("MATCH (n:Doc) RETURN n.title AS title").unwrap();
        assert_eq!(rs.rows.len(), 2);
        assert_eq!(g.vector_count("Doc", "embedding").unwrap(), 2);
    }

    #[test]
    fn vector_index_supports_int8_vectors() {
        let db = Database::open_in_memory().unwrap();
        let g = db.graph("test").unwrap();
        let create = |values: Vec<i64>| {
            g.storage
                .create_node(
                    &["Doc".to_string()],
                    &json!({"v": {"type": "int8", "values": values}}),
                )
                .unwrap()
        };
        create(vec![1, 2, 3]);
        create(vec![10, 10, 10]);
        create(vec![1, 2, 4]);
        g.create_vector_index("Doc", "v", "hnsw", Some("l2"), None)
            .unwrap();

        let results = g
            .vector_search("Doc", "v", &VectorValue::I8(vec![1, 2, 3]), 2)
            .unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, 1);
    }

    #[test]
    fn vector_index_cosine_metric() {
        let db = Database::open_in_memory().unwrap();
        let g = db.graph("test").unwrap();
        g.run(r#"CREATE (a:Doc {title: "east", embedding: [1.0, 0.0]})"#)
            .unwrap();
        g.run(r#"CREATE (b:Doc {title: "north", embedding: [0.0, 1.0]})"#)
            .unwrap();
        g.run(r#"CREATE (c:Doc {title: "west", embedding: [-1.0, 0.0]})"#)
            .unwrap();
        g.create_vector_index("Doc", "embedding", "hnsw", Some("cosine"), None)
            .unwrap();
        let rs = g
            .run(
                r#"MATCH (n:Doc) WHERE vector_nn(n.embedding, [2.0, 0.0], 2)
                   RETURN n.title AS title"#,
            )
            .unwrap();
        assert_eq!(rs.rows[0][0], json!("east"));
        assert_eq!(rs.rows[1][0], json!("north"));
    }

    #[test]
    fn vector_index_recreate_is_idempotent_and_rebuilds() {
        let db = Database::open_in_memory().unwrap();
        let g = db.graph("test").unwrap();
        g.run(r#"CREATE (a:Doc {title: "a", embedding: [0.0, 0.0]})"#)
            .unwrap();
        g.run(r#"CREATE (b:Doc {title: "b", embedding: [1.0, 0.0]})"#)
            .unwrap();
        g.run(r#"CREATE VECTOR INDEX ON :Doc(embedding)"#).unwrap();
        // Re-running DDL must not duplicate metadata or vectors.
        g.run(r#"CREATE VECTOR INDEX ON :Doc(embedding)"#).unwrap();
        assert_eq!(g.list_vector_indexes().unwrap().len(), 1);
        assert_eq!(g.vector_count("Doc", "embedding").unwrap(), 2);

        // A node added between the two calls is picked up by the reindex.
        g.run(r#"CREATE (c:Doc {title: "c", embedding: [2.0, 0.0]})"#)
            .unwrap();
        g.run(r#"CREATE VECTOR INDEX ON :Doc(embedding)"#).unwrap();
        assert_eq!(g.vector_count("Doc", "embedding").unwrap(), 3);
    }

    #[test]
    fn vector_index_default_ddl_algorithm_is_hnsw_l2() {
        let db = Database::open_in_memory().unwrap();
        let g = db.graph("test").unwrap();
        g.run(r#"CREATE (a:Doc {embedding: [1.0, 0.0]})"#).unwrap();
        g.run(r#"CREATE VECTOR INDEX ON :Doc(embedding)"#).unwrap();
        let indexes = g.list_vector_indexes().unwrap();
        assert_eq!(indexes[0].algorithm, "hnsw");
        assert_eq!(indexes[0].metric, "l2");
    }

    #[test]
    fn array_literals_in_return() {
        let db = Database::open_in_memory().unwrap();
        let g = db.graph("test").unwrap();
        g.run(r#"CREATE (a:Doc {title: "a", score: 0.5})"#).unwrap();
        let rs = g
            .run("MATCH (n:Doc) RETURN [n.title, n.score] AS pair")
            .unwrap();
        assert_eq!(rs.rows[0][0], json!(["a", 0.5]));
    }

    #[test]
    fn explicit_order_by_overrides_distance_order() {
        let db = Database::open_in_memory().unwrap();
        let g = db.graph("test").unwrap();
        g.run(r#"CREATE (a:Doc {title: "z", embedding: [0.0, 0.0]})"#)
            .unwrap();
        g.run(r#"CREATE (b:Doc {title: "a", embedding: [1.0, 0.0]})"#)
            .unwrap();
        g.create_vector_index("Doc", "embedding", "flat", None, None)
            .unwrap();
        let rs = g
            .run(
                r#"MATCH (n:Doc) WHERE vector_nn(n.embedding, [0.0, 0.0], 2)
                   RETURN n.title AS title ORDER BY title ASC"#,
            )
            .unwrap();
        assert_eq!(rs.rows[0][0], json!("a"));
        assert_eq!(rs.rows[1][0], json!("z"));
    }

    #[test]
    fn text_match_with_zero_hits_returns_no_rows() {
        let db = Database::open_in_memory().unwrap();
        let g = db.graph("test").unwrap();
        g.run(r#"CREATE (a:Doc {body: "rust database"})"#).unwrap();
        g.run(r#"CREATE (b:Doc {body: "python runtime"})"#).unwrap();
        g.create_text_index("Doc", "body", None).unwrap();
        let rs = g
            .run(
                r#"MATCH (n:Doc) WHERE text_match(n.body, "nonexistentterm")
                   RETURN n.body AS body"#,
            )
            .unwrap();
        assert!(rs.rows.is_empty(), "expected no rows, got {:?}", rs.rows);
    }

    #[test]
    fn fts_update_and_delete_sync() {
        let db = Database::open_in_memory().unwrap();
        let g = db.graph("test").unwrap();
        g.run(r#"CREATE (a:Doc {title: "a", body: "rust database"})"#)
            .unwrap();
        g.run(r#"CREATE (b:Doc {title: "b", body: "python runtime"})"#)
            .unwrap();
        g.create_text_index("Doc", "body", None).unwrap();

        // Updating a node must remove the old terms and index the new ones.
        g.run(r#"MATCH (n:Doc) WHERE n.title = "a" SET n.body = "python tooling""#)
            .unwrap();
        assert!(g.text_search("Doc", "body", "rust", 10).unwrap().is_empty());
        assert_eq!(
            g.text_search("Doc", "body", "tooling", 10).unwrap().len(),
            1
        );

        // Deleting a node must remove it from the index.
        g.run(r#"MATCH (n:Doc) WHERE n.title = "b" DELETE n"#)
            .unwrap();
        assert!(
            g.text_search("Doc", "body", "runtime", 10)
                .unwrap()
                .is_empty(),
            "deleted node must not remain searchable"
        );
    }

    #[test]
    fn fts_insert_after_index_sync() {
        let db = Database::open_in_memory().unwrap();
        let g = db.graph("test").unwrap();
        g.run(r#"CREATE (a:Doc {body: "rust database"})"#).unwrap();
        g.create_text_index("Doc", "body", None).unwrap();

        // The AFTER INSERT trigger must index nodes created after the index.
        g.run(r#"CREATE (b:Doc {body: "python runtime"})"#).unwrap();
        assert_eq!(g.text_search("Doc", "body", "python", 10).unwrap().len(), 1);

        // Nodes without the indexed property are skipped, not indexed as NULL.
        g.run(r#"CREATE (c:Doc {title: "no body"})"#).unwrap();
        assert_eq!(g.text_search("Doc", "body", "python", 10).unwrap().len(), 1);
    }

    #[test]
    fn fts_label_changes_sync() {
        use gvec_core::ids::NodeId;

        let db = Database::open_in_memory().unwrap();
        let g = db.graph("test").unwrap();
        let id = g
            .storage
            .create_node(&["Doc".into()], &json!({"body": "rust database"}))
            .unwrap();
        g.create_text_index("Doc", "body", None).unwrap();
        assert_eq!(g.text_search("Doc", "body", "rust", 10).unwrap().len(), 1);

        // Removing the indexed label must remove the row from FTS.
        g.storage
            .set_node_labels(NodeId(id.0), &["Other".into()])
            .unwrap();
        assert!(
            g.text_search("Doc", "body", "rust", 10).unwrap().is_empty(),
            "row with a removed label must not remain searchable"
        );

        // Adding the label back must index the row again.
        g.storage
            .set_node_labels(NodeId(id.0), &["Doc".into()])
            .unwrap();
        assert_eq!(
            g.text_search("Doc", "body", "rust", 10).unwrap().len(),
            1,
            "row must be searchable after the label returns"
        );
    }

    #[test]
    fn fts_tokenizer_reconfiguration_is_honored() {
        let db = Database::open_in_memory().unwrap();
        let g = db.graph("test").unwrap();
        g.run(r#"CREATE (a:Doc {body: "running quickly"})"#)
            .unwrap();
        g.create_text_index("Doc", "body", Some("unicode61"))
            .unwrap();
        assert!(
            g.text_search("Doc", "body", "run", 10).unwrap().is_empty(),
            "unicode61 must not stem 'running' to 'run'"
        );

        // Re-creating with porter must switch the tokenizer, not silently keep
        // the previous one.
        g.create_text_index("Doc", "body", Some("porter")).unwrap();
        assert_eq!(
            g.text_search("Doc", "body", "run", 10).unwrap().len(),
            1,
            "porter should stem 'running' to 'run'"
        );
    }

    #[test]
    fn fts_bm25_repeated_term_ranks_higher() {
        let db = Database::open_in_memory().unwrap();
        let g = db.graph("test").unwrap();
        g.run(r#"CREATE (a:Doc {body: "rust rust rust database engine"})"#)
            .unwrap();
        g.run(r#"CREATE (b:Doc {body: "rust database engine tools"})"#)
            .unwrap();
        g.create_text_index("Doc", "body", None).unwrap();
        let res = g.text_search("Doc", "body", "rust", 10).unwrap();
        assert_eq!(res.len(), 2);
        assert_eq!(
            res[0].0, 1,
            "the document with more occurrences should rank first"
        );
        assert!(res[0].1 > res[1].1, "scores must be descending");
    }

    #[test]
    fn fts_text_match_is_not_capped_at_50() {
        let db = Database::open_in_memory().unwrap();
        let g = db.graph("test").unwrap();
        for i in 0..75 {
            g.storage
                .create_node(
                    &["Doc".into()],
                    &json!({"body": format!("common term {i}")}),
                )
                .unwrap();
        }
        g.create_text_index("Doc", "body", None).unwrap();
        let rs = g
            .run(
                r#"MATCH (n:Doc) WHERE text_match(n.body, "common")
                   RETURN n.body AS body"#,
            )
            .unwrap();
        assert_eq!(rs.rows.len(), 75, "text_match must not silently cap hits");

        // The query's own LIMIT still applies.
        let rs = g
            .run(
                r#"MATCH (n:Doc) WHERE text_match(n.body, "common")
                   RETURN n.body AS body LIMIT 10"#,
            )
            .unwrap();
        assert_eq!(rs.rows.len(), 10);
    }

    #[test]
    fn fts_empty_query_returns_no_rows() {
        let db = Database::open_in_memory().unwrap();
        let g = db.graph("test").unwrap();
        g.run(r#"CREATE (a:Doc {body: "rust database"})"#).unwrap();
        g.create_text_index("Doc", "body", None).unwrap();
        assert!(g.text_search("Doc", "body", "", 10).unwrap().is_empty());
        assert!(g.text_search("Doc", "body", "   ", 10).unwrap().is_empty());
        let rs = g
            .run(r#"MATCH (n:Doc) WHERE text_match(n.body, "") RETURN n.body AS body"#)
            .unwrap();
        assert!(rs.rows.is_empty());
    }

    #[test]
    fn fts_boolean_prefix_and_phrase_syntax() {
        let db = Database::open_in_memory().unwrap();
        let g = db.graph("test").unwrap();
        g.run(r#"CREATE (a:Doc {body: "rust database engine"})"#)
            .unwrap();
        g.run(r#"CREATE (b:Doc {body: "rust async runtime"})"#)
            .unwrap();
        g.run(r#"CREATE (c:Doc {body: "python database engine"})"#)
            .unwrap();
        g.create_text_index("Doc", "body", None).unwrap();

        // AND, OR, NOT, prefix, and phrase are passed through to FTS5.
        assert_eq!(
            g.text_search("Doc", "body", "rust AND database", 10)
                .unwrap()
                .len(),
            1
        );
        assert_eq!(
            g.text_search("Doc", "body", "rust OR python", 10)
                .unwrap()
                .len(),
            3
        );
        assert_eq!(
            g.text_search("Doc", "body", "database NOT python", 10)
                .unwrap()
                .len(),
            1
        );
        assert_eq!(g.text_search("Doc", "body", "data*", 10).unwrap().len(), 2);
        assert_eq!(
            g.text_search("Doc", "body", "\"rust database\"", 10)
                .unwrap()
                .len(),
            1
        );
    }

    #[test]
    fn fts_multiple_text_match_is_rejected() {
        let db = Database::open_in_memory().unwrap();
        let g = db.graph("test").unwrap();
        g.run(r#"CREATE (a:Doc {title: "guide", body: "rust database"})"#)
            .unwrap();
        g.create_text_index("Doc", "body", None).unwrap();
        g.create_text_index("Doc", "title", None).unwrap();
        let err = g
            .run(
                r#"MATCH (n:Doc)
                   WHERE text_match(n.body, "rust") AND text_match(n.title, "guide")
                   RETURN n.title AS title"#,
            )
            .unwrap_err();
        assert!(
            err.to_string().contains("multiple text_match"),
            "expected an explicit limitation, got: {err}"
        );

        let err = g
            .run(
                r#"MATCH (n:Doc)
                   WHERE text_match(n.body, "rust") OR n.title = "guide"
                   RETURN n.title AS title"#,
            )
            .unwrap_err();
        assert!(
            err.to_string().contains("top-level conjunct"),
            "expected an explicit limitation, got: {err}"
        );
    }

    #[test]
    fn fts_trigram_substring_and_cjk() {
        let db = Database::open_in_memory().unwrap();
        let g = db.graph("test").unwrap();
        g.run(r#"CREATE (a:Doc {body: "embedded vector database"})"#)
            .unwrap();
        g.run(r#"CREATE (b:Doc {body: "图数据库支持全文检索"})"#)
            .unwrap();
        g.create_text_index("Doc", "body", Some("trigram")).unwrap();

        let res = g.text_search("Doc", "body", "vector data", 10).unwrap();
        assert_eq!(res.len(), 1, "trigram must match interior substrings");

        let cjk = g.text_search("Doc", "body", "全文检索", 10).unwrap();
        assert_eq!(cjk.len(), 1, "trigram must match CJK substrings");
    }
}
