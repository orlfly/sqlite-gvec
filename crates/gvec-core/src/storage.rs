use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

use serde_json::Value;

use crate::edge::Edge;
use crate::error::{Error, Result};
use crate::ids::{EdgeId, NodeId};
use crate::node::Node;
use crate::property::resolve;
use crate::sql::{SqlExecutor, SqlValue};
use crate::vector::{Distance, VectorBlob, VectorValue};
use crate::vector_index::{VectorIndex, VectorIndexAlgorithm, VectorIndexConfig};

/// Metadata for a persisted vector index over `(label, property)`.
#[derive(Debug, Clone, PartialEq)]
pub struct VectorIndexMeta {
    pub id: i64,
    pub label: String,
    pub property: String,
    pub algorithm: String,
    pub metric: String,
    pub params: Value,
    pub dim: Option<usize>,
    pub element_type: Option<String>,
}

impl VectorIndexMeta {
    pub fn config(&self) -> Result<VectorIndexConfig> {
        let algorithm = VectorIndexAlgorithm::parse(&self.algorithm).ok_or_else(|| {
            Error::Vector(format!(
                "unknown vector index algorithm '{}'",
                self.algorithm
            ))
        })?;
        let metric = Distance::parse(&self.metric)
            .ok_or_else(|| Error::Vector(format!("unknown distance metric '{}'", self.metric)))?;
        Ok(VectorIndexConfig {
            algorithm,
            metric,
            params: self.params.clone(),
        })
    }
}

/// Storage facade. Holds the shadow-table prefix and an owned `SqlExecutor`.
/// Owned (not borrowed) so callers don't need lifetime gymnastics.
pub struct Storage {
    /// The database executor (private; bindings use the public methods on `Storage`).
    pub(crate) exec: Box<dyn SqlExecutor>,
    pub prefix: String,
    /// Cached list of vector indexes, invalidated by vector-index DDL.
    vector_meta_cache: Mutex<Option<Vec<VectorIndexMeta>>>,
    /// Lazily built ANN indexes, keyed by `(label, property)`. Invalidated on
    /// any vector write so the next query rebuilds from persisted vectors.
    vector_index_cache: Mutex<HashMap<(String, String), Arc<VectorIndex>>>,
}

impl Storage {
    /// Borrow the underlying executor. Used by `gvec-bindings` to access
    /// FTS5 functionality.
    pub fn exec_ref(&self) -> &dyn SqlExecutor {
        &*self.exec
    }
}

impl Storage {
    pub fn new(exec: impl SqlExecutor + 'static, prefix: impl Into<String>) -> Self {
        Self {
            exec: Box::new(exec),
            prefix: prefix.into(),
            vector_meta_cache: Mutex::new(None),
            vector_index_cache: Mutex::new(HashMap::new()),
        }
    }

    /// Construct from an already-boxed executor.
    pub fn with_boxed(exec: Box<dyn SqlExecutor>, prefix: impl Into<String>) -> Self {
        Self {
            exec,
            prefix: prefix.into(),
            vector_meta_cache: Mutex::new(None),
            vector_index_cache: Mutex::new(HashMap::new()),
        }
    }

    fn tbl(&self, name: &str) -> String {
        format!("{}_{}", self.prefix, name)
    }

    pub fn ensure_schema(&self) -> Result<()> {
        // Issue one statement at a time; `SqlExecutor::execute` runs a single
        // statement only (rusqlite's `execute` does not handle `;`-separated
        // batches).
        let vector_rows_existed = self
            .exec
            .query_one_json(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?1",
                &[SqlValue::Text(self.tbl("vector_rows"))],
            )?
            .is_some();
        self.exec.execute(
            &format!(
                "CREATE TABLE IF NOT EXISTS {nodes} (
                id    INTEGER PRIMARY KEY,
                labels TEXT NOT NULL,
                properties TEXT NOT NULL
            )",
                nodes = self.tbl("nodes"),
            ),
            &[],
        )?;
        self.exec.execute(
            &format!(
                "CREATE TABLE IF NOT EXISTS {edges} (
                id    INTEGER PRIMARY KEY,
                source INTEGER NOT NULL,
                target INTEGER NOT NULL,
                edge_type TEXT,
                properties TEXT NOT NULL,
                FOREIGN KEY(source) REFERENCES {nodes}(id) ON DELETE CASCADE,
                FOREIGN KEY(target) REFERENCES {nodes}(id) ON DELETE CASCADE
            )",
                edges = self.tbl("edges"),
                nodes = self.tbl("nodes"),
            ),
            &[],
        )?;
        self.exec.execute(
            &format!(
                "CREATE INDEX IF NOT EXISTS {}_edges_source ON {}(source)",
                self.prefix,
                self.tbl("edges")
            ),
            &[],
        )?;
        self.exec.execute(
            &format!(
                "CREATE INDEX IF NOT EXISTS {}_edges_target ON {}(target)",
                self.prefix,
                self.tbl("edges")
            ),
            &[],
        )?;
        self.exec.execute(
            &format!(
                "CREATE TABLE IF NOT EXISTS {chunks} (
                chunk_id INTEGER PRIMARY KEY,
                size INTEGER NOT NULL,
                validity BLOB,
                rowids BLOB
            )",
                chunks = self.tbl("chunks"),
            ),
            &[],
        )?;
        self.exec.execute(
            &format!(
                "CREATE TABLE IF NOT EXISTS {rowids} (
                rowid INTEGER PRIMARY KEY,
                id INTEGER,
                chunk_id INTEGER,
                chunk_offset INTEGER
            )",
                rowids = self.tbl("rowids"),
            ),
            &[],
        )?;
        self.exec.execute(
            &format!(
                "CREATE TABLE IF NOT EXISTS {indexes} (
                id          INTEGER PRIMARY KEY,
                label       TEXT NOT NULL,
                property    TEXT NOT NULL,
                algorithm   TEXT NOT NULL,
                metric      TEXT NOT NULL,
                params      TEXT NOT NULL,
                dim         INTEGER,
                element_type TEXT,
                UNIQUE(label, property)
            )",
                indexes = self.tbl("vector_indexes"),
            ),
            &[],
        )?;
        self.exec.execute(
            &format!(
                "CREATE TABLE IF NOT EXISTS {vectors} (
                index_id INTEGER NOT NULL,
                node_id  INTEGER NOT NULL,
                vector   TEXT NOT NULL,
                PRIMARY KEY (index_id, node_id)
            )",
                vectors = self.tbl("vectors"),
            ),
            &[],
        )?;
        self.exec.execute(
            &format!(
                "CREATE TABLE IF NOT EXISTS {rows} (
                node_id INTEGER NOT NULL,
                property TEXT NOT NULL,
                vector TEXT NOT NULL,
                dim INTEGER NOT NULL,
                element_type TEXT NOT NULL,
                PRIMARY KEY (node_id, property),
                FOREIGN KEY(node_id) REFERENCES {nodes}(id) ON DELETE CASCADE
            )",
                rows = self.tbl("vector_rows"),
                nodes = self.tbl("nodes"),
            ),
            &[],
        )?;
        self.exec.execute(
            &format!(
                "CREATE INDEX IF NOT EXISTS {}_vector_rows_property
                 ON {}(property)",
                self.prefix,
                self.tbl("vector_rows")
            ),
            &[],
        )?;
        if !vector_rows_existed {
            self.backfill_vector_rows_from_indexes()?;
        }
        crate::tag::ensure_schema(self.exec.as_ref(), &self.prefix)?;
        Ok(())
    }

    /// One-time migration for databases created before persistent vector rows
    /// existed: copy vectors from the per-index table into `vector_rows`.
    fn backfill_vector_rows_from_indexes(&self) -> Result<()> {
        for meta in self.vector_indexes()? {
            let rows = self.exec.query_all_json(
                &format!(
                    "SELECT node_id, vector FROM {} WHERE index_id = ?1",
                    self.tbl("vectors")
                ),
                &[SqlValue::from(meta.id)],
            )?;
            for row in rows {
                let node_id = NodeId(json_to_i64(&row[0])?);
                let blob = VectorBlob::from_hex(&json_to_string(&row[1])?)?;
                self.insert_vector_row_raw(node_id, &meta.property, &blob.decode()?)?;
            }
        }
        Ok(())
    }

    pub fn drop_all(&self) -> Result<()> {
        let names = [
            "nodes",
            "edges",
            "chunks",
            "rowids",
            "auxiliary",
            "vectors",
            "vector_rows",
            "vector_indexes",
            "tags",
            "tag_indexes",
        ];
        for n in names {
            let _ = self
                .exec
                .execute(&format!("DROP TABLE IF EXISTS {}", self.tbl(n)), &[]);
        }
        self.invalidate_vector_meta_cache();
        self.invalidate_all_indexes();
        Ok(())
    }

    // ---- Node CRUD ----

    pub fn create_node(&self, labels: &[String], properties: &Value) -> Result<NodeId> {
        // Validate and stage vector-index rows before writing, so a malformed
        // vector does not leave a half-written node behind.
        let prepared = self.prepare_node_vectors(labels, properties)?;
        let vector_rows = self.prepare_node_vector_rows(labels, properties)?;
        self.exec.execute(
            &format!(
                "INSERT INTO {} (labels, properties) VALUES (?1, ?2)",
                self.tbl("nodes")
            ),
            &[
                SqlValue::Text(serde_json::to_string(labels)?),
                SqlValue::Text(serde_json::to_string(properties)?),
            ],
        )?;
        let id = NodeId(self.exec.last_insert_rowid());
        self.apply_node_vectors(id, &prepared)?;
        self.replace_node_vector_rows(id, &vector_rows, &HashSet::new())?;
        Ok(id)
    }

    pub fn get_node(&self, id: NodeId) -> Result<Node> {
        let row = self.exec.query_one_json(
            &format!(
                "SELECT id, labels, properties FROM {} WHERE id = ?1",
                self.tbl("nodes")
            ),
            &[SqlValue::from(id.0)],
        )?;
        match row {
            None => Err(Error::Storage(format!("node {} not found", id.0))),
            Some(row) => {
                let id = json_to_i64(&row[0])?;
                let labels = json_to_string(&row[1])?;
                let properties = json_to_string(&row[2])?;
                Ok(Node {
                    id: NodeId(id),
                    labels: serde_json::from_str(&labels)?,
                    properties: serde_json::from_str(&properties)?,
                })
            }
        }
    }

    /// List nodes by label. An empty label selects ALL nodes (so `MATCH (n)`
    /// without a label returns unlabeled nodes too, whose `labels` is `"[]"`).
    pub fn list_nodes_by_label(&self, label: &str) -> Result<Vec<Node>> {
        let (where_sql, params): (&str, Vec<SqlValue>) = if label.is_empty() {
            ("", Vec::new())
        } else {
            (
                "WHERE labels LIKE ?1",
                vec![SqlValue::Text(format!("%\"{}\"%", label))],
            )
        };
        let rows = self.exec.query_all_json(
            &format!(
                "SELECT id, labels, properties FROM {} {}",
                self.tbl("nodes"),
                where_sql
            ),
            &params,
        )?;
        let mut out = Vec::with_capacity(rows.len());
        for row in rows {
            let id = json_to_i64(&row[0])?;
            let labels = json_to_string(&row[1])?;
            let properties = json_to_string(&row[2])?;
            out.push(Node {
                id: NodeId(id),
                labels: serde_json::from_str(&labels)?,
                properties: serde_json::from_str(&properties)?,
            });
        }
        Ok(out)
    }

    /// List nodes by label and top-level scalar properties, pushing the
    /// property comparisons into SQLite's JSON1 extractor. The caller still
    /// performs the exact Rust-side match, so this is only a prefilter. It
    /// avoids materializing and JSON-decoding every node of a label when the
    /// pattern has a selective inline property such as `{id: 42}`.
    pub fn list_nodes_by_label_and_properties(
        &self,
        label: &str,
        properties: &[(String, Value)],
    ) -> Result<Vec<Node>> {
        let mut clauses: Vec<String> = Vec::new();
        let mut params: Vec<SqlValue> = Vec::new();
        if !label.is_empty() {
            clauses.push("labels LIKE ?1".to_string());
            params.push(SqlValue::Text(format!("%\"{}\"%", label)));
        }
        let mut next = params.len() + 1;
        for (key, value) in properties {
            // Only top-level keys with a scalar value can be narrowed safely.
            // The Rust filter handles nested paths, arrays, objects, and keys
            // whose JSON path would need escaping.
            if key.is_empty()
                || key.contains('.')
                || key.contains('"')
                || key.contains('\\')
                || value.is_array()
                || value.is_object()
            {
                continue;
            }
            let path = SqlValue::Text(format!("$.\"{key}\""));
            if value.is_null() {
                clauses.push(format!("json_extract(properties, ?{next}) IS NULL"));
                params.push(path);
                next += 1;
            } else {
                clauses.push(format!("json_extract(properties, ?{next}) = ?{}", next + 1));
                params.push(path);
                params.push(crate::sql::from_json(value));
                next += 2;
            }
        }
        if clauses.is_empty() {
            return self.list_nodes_by_label(label);
        }
        let rows = self.exec.query_all_json(
            &format!(
                "SELECT id, labels, properties FROM {} WHERE {}",
                self.tbl("nodes"),
                clauses.join(" AND ")
            ),
            &params,
        )?;
        let mut out = Vec::with_capacity(rows.len());
        for row in rows {
            let id = json_to_i64(&row[0])?;
            let labels = json_to_string(&row[1])?;
            let properties = json_to_string(&row[2])?;
            out.push(Node {
                id: NodeId(id),
                labels: serde_json::from_str(&labels)?,
                properties: serde_json::from_str(&properties)?,
            });
        }
        Ok(out)
    }

    /// Fetch nodes by id, in chunks that stay below SQLite's variable limit.
    pub fn list_nodes_by_ids(&self, ids: &std::collections::HashSet<i64>) -> Result<Vec<Node>> {
        if ids.is_empty() {
            return Ok(Vec::new());
        }
        let ids: Vec<i64> = ids.iter().copied().collect();
        let mut out = Vec::with_capacity(ids.len());
        for chunk in ids.chunks(500) {
            let placeholders = (1..=chunk.len())
                .map(|i| format!("?{i}"))
                .collect::<Vec<_>>()
                .join(", ");
            let params: Vec<SqlValue> = chunk.iter().map(|id| SqlValue::from(*id)).collect();
            let rows = self.exec.query_all_json(
                &format!(
                    "SELECT id, labels, properties FROM {} WHERE id IN ({})",
                    self.tbl("nodes"),
                    placeholders
                ),
                &params,
            )?;
            for row in rows {
                let id = json_to_i64(&row[0])?;
                let labels = json_to_string(&row[1])?;
                let properties = json_to_string(&row[2])?;
                out.push(Node {
                    id: NodeId(id),
                    labels: serde_json::from_str(&labels)?,
                    properties: serde_json::from_str(&properties)?,
                });
            }
        }
        out.sort_by_key(|n| n.id.0);
        Ok(out)
    }

    // ---- Tag indexes ----

    /// Create or rebuild an exact tag index over `label.property`.
    pub fn create_tag_index(
        &self,
        label: &str,
        property: &str,
    ) -> Result<crate::tag::TagIndexMeta> {
        crate::tag::create_tag_index(self.exec.as_ref(), &self.prefix, label, property)
    }

    /// Drop a tag index and its stored tags.
    pub fn drop_tag_index(&self, label: &str, property: &str) -> Result<()> {
        crate::tag::drop_tag_index(self.exec.as_ref(), &self.prefix, label, property)
    }

    /// List all tag indexes for this graph.
    pub fn tag_indexes(&self) -> Result<Vec<crate::tag::TagIndexMeta>> {
        crate::tag::list_tag_indexes(self.exec.as_ref(), &self.prefix)
    }

    /// Find a tag index by label and property.
    pub fn find_tag_index(
        &self,
        label: &str,
        property: &str,
    ) -> Result<Option<crate::tag::TagIndexMeta>> {
        crate::tag::find_tag_index(self.exec.as_ref(), &self.prefix, label, property)
    }

    /// Number of distinct nodes in a tag index. Returns zero if it does not exist.
    pub fn tag_count(&self, label: &str, property: &str) -> Result<usize> {
        let Some(meta) = self.find_tag_index(label, property)? else {
            return Ok(0);
        };
        crate::tag::tag_count(self.exec.as_ref(), &self.prefix, meta.id)
    }

    /// Fetch every node id carrying `tag`. Falls back to a property scan when
    /// no tag index exists.
    pub fn tag_search(&self, label: &str, property: &str, tag: &str) -> Result<Vec<i64>> {
        self.tag_search_limit(label, property, tag, None)
    }

    /// Fetch up to `k` node ids carrying `tag`. `None` means no limit.
    pub fn tag_search_limit(
        &self,
        label: &str,
        property: &str,
        tag: &str,
        k: Option<usize>,
    ) -> Result<Vec<i64>> {
        if k == Some(0) {
            return Ok(Vec::new());
        }
        if let Some(meta) = self.find_tag_index(label, property)? {
            return crate::tag::tag_search_ids(self.exec.as_ref(), &self.prefix, meta.id, tag, k);
        }
        let mut ids = self.exact_tag_search(label, property, tag)?;
        if let Some(k) = k {
            ids.truncate(k);
        }
        Ok(ids)
    }

    /// Exact tag scan used when no tag index is defined.
    pub fn exact_tag_search(&self, label: &str, property: &str, tag: &str) -> Result<Vec<i64>> {
        let mut out = Vec::new();
        for node in self.list_nodes_by_label(label)? {
            let Some(value) = crate::property::resolve(&node.properties, property) else {
                continue;
            };
            if crate::tag::value_has_tag(value, tag) {
                out.push(node.id.0);
            }
        }
        out.sort_unstable();
        Ok(out)
    }

    pub fn set_node_labels(&self, id: NodeId, labels: &[String]) -> Result<()> {
        let node = self.get_node(id)?;
        self.update_node(id, labels, &node.properties)
    }

    /// Update a node's labels and properties, keeping matching vector indexes
    /// in sync. Rows derived from the previous JSON state are replaced with the
    /// vectors implied by the new state; explicit sidecar rows for properties
    /// that never appeared in JSON are preserved.
    pub fn update_node(&self, id: NodeId, labels: &[String], properties: &Value) -> Result<()> {
        let previous = self.get_node(id)?;
        let previous_rows =
            self.prepare_node_vector_rows(&previous.labels, &previous.properties)?;
        let previous_properties: HashSet<String> = previous_rows
            .into_iter()
            .map(|(property, _)| property)
            .collect();
        self.prepare_node_vectors(labels, properties)?;
        let vector_rows = self.prepare_node_vector_rows(labels, properties)?;
        let n = self.exec.execute(
            &format!(
                "UPDATE {} SET labels = ?1, properties = ?2 WHERE id = ?3",
                self.tbl("nodes")
            ),
            &[
                SqlValue::Text(serde_json::to_string(labels)?),
                SqlValue::Text(serde_json::to_string(properties)?),
                SqlValue::from(id.0),
            ],
        )?;
        if n == 0 {
            return Err(Error::Storage(format!("node {} not found", id.0)));
        }
        self.clear_node_index_vectors(id)?;
        self.replace_node_vector_rows(id, &vector_rows, &previous_properties)?;
        self.reindex_node_vector_rows(id)?;
        Ok(())
    }

    pub fn delete_node(&self, id: NodeId) -> Result<()> {
        self.clear_node_vectors(id)?;
        let n = self.exec.execute(
            &format!("DELETE FROM {} WHERE id = ?1", self.tbl("nodes")),
            &[SqlValue::from(id.0)],
        )?;
        if n == 0 {
            return Err(Error::Storage(format!("node {} not found", id.0)));
        }
        Ok(())
    }

    // ---- Edge CRUD ----

    pub fn create_edge(
        &self,
        source: NodeId,
        target: NodeId,
        edge_type: Option<&str>,
        properties: &Value,
    ) -> Result<EdgeId> {
        let s_exists = self
            .exec
            .query_one_json(
                &format!("SELECT 1 FROM {} WHERE id = ?1", self.tbl("nodes")),
                &[SqlValue::from(source.0)],
            )?
            .is_some();
        if !s_exists {
            return Err(Error::Storage(format!(
                "source node {} not found",
                source.0
            )));
        }
        let t_exists = self
            .exec
            .query_one_json(
                &format!("SELECT 1 FROM {} WHERE id = ?1", self.tbl("nodes")),
                &[SqlValue::from(target.0)],
            )?
            .is_some();
        if !t_exists {
            return Err(Error::Storage(format!(
                "target node {} not found",
                target.0
            )));
        }

        let type_val: SqlValue = match edge_type {
            Some(s) => SqlValue::Text(s.to_string()),
            None => SqlValue::Null,
        };
        self.exec.execute(
            &format!(
                "INSERT INTO {} (source, target, edge_type, properties) VALUES (?1, ?2, ?3, ?4)",
                self.tbl("edges")
            ),
            &[
                SqlValue::from(source.0),
                SqlValue::from(target.0),
                type_val,
                SqlValue::Text(serde_json::to_string(properties)?),
            ],
        )?;
        Ok(EdgeId(self.exec.last_insert_rowid()))
    }

    pub fn list_edges(&self, source: Option<NodeId>, target: Option<NodeId>) -> Result<Vec<Edge>> {
        let mut sql = format!(
            "SELECT id, source, target, edge_type, properties FROM {}",
            self.tbl("edges")
        );
        let params: Vec<SqlValue> = match (source, target) {
            (Some(s), Some(t)) => {
                sql.push_str(" WHERE source = ?1 AND target = ?2");
                vec![SqlValue::from(s.0), SqlValue::from(t.0)]
            }
            (Some(s), None) => {
                sql.push_str(" WHERE source = ?1");
                vec![SqlValue::from(s.0)]
            }
            (None, Some(t)) => {
                sql.push_str(" WHERE target = ?1");
                vec![SqlValue::from(t.0)]
            }
            (None, None) => vec![],
        };
        let rows = self.exec.query_all_json(&sql, &params)?;
        let mut out = Vec::with_capacity(rows.len());
        for row in rows {
            let id = json_to_i64(&row[0])?;
            let source = json_to_i64(&row[1])?;
            let target = json_to_i64(&row[2])?;
            let edge_type = match &row[3] {
                Value::Null => None,
                v => Some(json_to_string(v)?),
            };
            let properties = json_to_string(&row[4])?;
            out.push(Edge {
                id: EdgeId(id),
                source: NodeId(source),
                target: NodeId(target),
                edge_type,
                properties: serde_json::from_str(&properties)?,
            });
        }
        Ok(out)
    }

    pub fn delete_edge(&self, id: EdgeId) -> Result<()> {
        let n = self.exec.execute(
            &format!("DELETE FROM {} WHERE id = ?1", self.tbl("edges")),
            &[SqlValue::from(id.0)],
        )?;
        if n == 0 {
            return Err(Error::Storage(format!("edge {} not found", id.0)));
        }
        Ok(())
    }

    // ---- Vector index management ----

    /// List every persisted vector index attached to this graph.
    pub fn vector_indexes(&self) -> Result<Vec<VectorIndexMeta>> {
        {
            let guard = self
                .vector_meta_cache
                .lock()
                .map_err(|e| Error::Storage(format!("mutex: {e}")))?;
            if let Some(cached) = guard.as_ref() {
                return Ok(cached.clone());
            }
        }
        let rows = self.exec.query_all_json(
            &format!(
                "SELECT id, label, property, algorithm, metric, params, dim, element_type
                 FROM {} ORDER BY id",
                self.tbl("vector_indexes")
            ),
            &[],
        )?;
        let mut out = Vec::with_capacity(rows.len());
        for row in rows {
            let id = json_to_i64(&row[0])?;
            let params = json_to_string(&row[5])?;
            out.push(VectorIndexMeta {
                id,
                label: json_to_string(&row[1])?,
                property: json_to_string(&row[2])?,
                algorithm: json_to_string(&row[3])?,
                metric: json_to_string(&row[4])?,
                params: serde_json::from_str(&params)?,
                dim: match &row[6] {
                    Value::Null => None,
                    v => Some(json_to_i64(v)? as usize),
                },
                element_type: match &row[7] {
                    Value::Null => None,
                    v => Some(json_to_string(v)?),
                },
            });
        }
        let mut guard = self
            .vector_meta_cache
            .lock()
            .map_err(|e| Error::Storage(format!("mutex: {e}")))?;
        *guard = Some(out.clone());
        Ok(out)
    }

    pub fn find_vector_index(
        &self,
        label: &str,
        property: &str,
    ) -> Result<Option<VectorIndexMeta>> {
        Ok(self
            .vector_indexes()?
            .into_iter()
            .find(|m| m.label == label && m.property == property))
    }

    /// Create (or reconfigure) a vector index over `label.property` and
    /// backfill it from existing node properties.
    ///
    /// `options` are free-form key/value pairs (`metric`, `m`,
    /// `ef_construction`, `nlist`, ...). The `metric` option is also stored as
    /// the index metric.
    pub fn create_vector_index(
        &self,
        label: &str,
        property: &str,
        algorithm: &str,
        options: &[(String, Value)],
    ) -> Result<VectorIndexMeta> {
        validate_ident("label", label)?;
        validate_ident("property", property)?;
        let algorithm = VectorIndexAlgorithm::parse(algorithm).ok_or_else(|| {
            Error::Vector(format!(
                "unknown vector index algorithm '{algorithm}'; supported: flat, hnsw, ivf"
            ))
        })?;

        let mut params = serde_json::Map::new();
        for (k, v) in options {
            params.insert(k.clone(), v.clone());
        }
        let metric = match params.get("metric") {
            Some(Value::String(s)) => s.clone(),
            Some(other) => {
                return Err(Error::Vector(format!(
                    "metric option must be a string, got {other}"
                )))
            }
            None => "l2".to_string(),
        };
        if Distance::parse(&metric).is_none() {
            return Err(Error::Vector(format!(
                "unknown distance metric '{metric}'; supported: l2, cosine, ip"
            )));
        }
        params.insert("metric".into(), Value::String(metric.clone()));

        // Collect and validate the current node vectors *before* mutating any
        // index state, so a dimension/type mismatch cannot leave a partially
        // rebuilt index behind.
        let rows = self.exec.query_all_json(
            &format!(
                "SELECT id, properties FROM {} WHERE labels LIKE ?1",
                self.tbl("nodes")
            ),
            &[SqlValue::Text(format!("%\"{}\"%", label))],
        )?;
        let mut expected_dim: Option<usize> = None;
        let mut expected_type: Option<&'static str> = None;
        let mut prepared: Vec<(NodeId, VectorValue)> = Vec::new();
        for row in rows {
            let node_id = NodeId(json_to_i64(&row[0])?);
            let properties: Value = serde_json::from_str(&json_to_string(&row[1])?)?;
            let Some(value) = resolve(&properties, property) else {
                continue;
            };
            if value.is_null() {
                continue;
            }
            let vector = VectorValue::from_json(value).map_err(|e| {
                Error::Vector(format!("node {} property {}: {e}", node_id.0, property))
            })?;
            match expected_dim {
                None => expected_dim = Some(vector.dim()),
                Some(dim) if dim != vector.dim() => {
                    return Err(Error::Vector(format!(
                        "node {} property {} has dimension {}, index expects {dim}",
                        node_id.0,
                        property,
                        vector.dim()
                    )))
                }
                _ => {}
            }
            match expected_type {
                None => expected_type = Some(vector.element_type().as_str()),
                Some(ty) if ty != vector.element_type().as_str() => {
                    return Err(Error::Vector(format!(
                        "node {} property {} has type {}, index expects {ty}",
                        node_id.0,
                        property,
                        vector.element_type().as_str()
                    )))
                }
                _ => {}
            }
            prepared.push((node_id, vector));
        }

        // Explicit rows keep vectors outside node JSON; include them when
        // building or rebuilding an index. JSON-derived vectors are already in
        // `prepared`, so skip duplicates.
        let mut prepared_ids: HashSet<i64> = prepared.iter().map(|(id, _)| id.0).collect();
        for row in self.vector_rows(Some(label), Some(property))? {
            if !prepared_ids.insert(row.node_id) {
                continue;
            }
            let vector = row.vector;
            match expected_dim {
                None => expected_dim = Some(vector.dim()),
                Some(dim) if dim != vector.dim() => {
                    return Err(Error::Vector(format!(
                        "node {} property {} has dimension {}, index expects {dim}",
                        row.node_id,
                        property,
                        vector.dim()
                    )))
                }
                _ => {}
            }
            match expected_type {
                None => expected_type = Some(vector.element_type().as_str()),
                Some(ty) if ty != vector.element_type().as_str() => {
                    return Err(Error::Vector(format!(
                        "node {} property {} has type {}, index expects {ty}",
                        row.node_id,
                        property,
                        vector.element_type().as_str()
                    )))
                }
                _ => {}
            }
            prepared.push((NodeId(row.node_id), vector));
        }

        self.exec.execute(
            &format!(
                "INSERT INTO {tbl} (label, property, algorithm, metric, params, dim, element_type)
                 VALUES (?1, ?2, ?3, ?4, ?5, NULL, NULL)
                 ON CONFLICT(label, property) DO UPDATE SET
                   algorithm = excluded.algorithm,
                   metric = excluded.metric,
                   params = excluded.params,
                   dim = NULL,
                   element_type = NULL",
                tbl = self.tbl("vector_indexes")
            ),
            &[
                SqlValue::Text(label.to_string()),
                SqlValue::Text(property.to_string()),
                SqlValue::Text(algorithm.as_str().to_string()),
                SqlValue::Text(metric),
                SqlValue::Text(Value::Object(params).to_string()),
            ],
        )?;
        self.invalidate_vector_meta_cache();

        let meta = self
            .find_vector_index(label, property)?
            .ok_or_else(|| Error::Storage("vector index disappeared after create".into()))?;

        // Reindex from scratch so re-running CREATE VECTOR INDEX is idempotent
        // and picks up every current node.
        self.exec.execute(
            &format!("DELETE FROM {} WHERE index_id = ?1", self.tbl("vectors")),
            &[SqlValue::from(meta.id)],
        )?;

        for (node_id, vector) in prepared {
            self.insert_vector_row_raw(node_id, property, &vector)?;
            self.upsert_index_vector(meta.id, node_id, &vector)?;
        }

        self.find_vector_index(label, property)?
            .ok_or_else(|| Error::Storage("vector index disappeared after backfill".into()))
    }

    /// Drop a vector index and all its stored vectors.
    pub fn drop_vector_index(&self, label: &str, property: &str) -> Result<()> {
        let Some(meta) = self.find_vector_index(label, property)? else {
            return Ok(());
        };
        self.exec.execute(
            &format!("DELETE FROM {} WHERE index_id = ?1", self.tbl("vectors")),
            &[SqlValue::from(meta.id)],
        )?;
        self.exec.execute(
            &format!("DELETE FROM {} WHERE id = ?1", self.tbl("vector_indexes")),
            &[SqlValue::from(meta.id)],
        )?;
        self.invalidate_vector_meta_cache();
        self.invalidate_all_indexes();
        Ok(())
    }

    /// Number of vectors currently stored for an index.
    pub fn vector_count(&self, label: &str, property: &str) -> Result<usize> {
        let Some(meta) = self.find_vector_index(label, property)? else {
            return Ok(0);
        };
        let row = self.exec.query_one_json(
            &format!(
                "SELECT COUNT(*) FROM {} WHERE index_id = ?1",
                self.tbl("vectors")
            ),
            &[SqlValue::from(meta.id)],
        )?;
        Ok(row
            .and_then(|r| r.first().and_then(|v| v.as_i64()))
            .unwrap_or(0) as usize)
    }

    /// Count persisted vector rows, optionally filtered by node label and
    /// property. Unlike `vector_count`, this does not require a vector index.
    pub fn vector_row_count(&self, label: Option<&str>, property: Option<&str>) -> Result<usize> {
        let mut sql = format!("SELECT COUNT(*) FROM {}", self.tbl("vector_rows"));
        let mut clauses = Vec::new();
        let mut params = Vec::new();
        if let Some(label) = label.filter(|l| !l.is_empty()) {
            sql.push_str(&format!(
                " JOIN {} n ON n.id = {}.node_id",
                self.tbl("nodes"),
                self.tbl("vector_rows")
            ));
            clauses.push(format!("n.labels LIKE ?{}", params.len() + 1));
            params.push(SqlValue::Text(format!("%\"{}\"%", label)));
        }
        if let Some(property) = property {
            clauses.push(format!(
                "{}.property = ?{}",
                self.tbl("vector_rows"),
                params.len() + 1
            ));
            params.push(SqlValue::Text(property.to_string()));
        }
        if !clauses.is_empty() {
            sql.push_str(" WHERE ");
            sql.push_str(&clauses.join(" AND "));
        }
        let row = self.exec.query_one_json(&sql, &params)?;
        Ok(row
            .and_then(|r| r.first().and_then(|v| v.as_i64()))
            .unwrap_or(0) as usize)
    }

    /// List persisted vector rows, optionally filtered by node label and
    /// property. The label filter joins the node table so multi-label nodes are
    /// handled by the same `labels LIKE` semantics as pattern matching.
    pub fn vector_rows(
        &self,
        label: Option<&str>,
        property: Option<&str>,
    ) -> Result<Vec<crate::vector::VectorRow>> {
        let mut sql = format!(
            "SELECT vr.node_id, vr.property, vr.vector FROM {} vr",
            self.tbl("vector_rows")
        );
        let mut clauses = Vec::new();
        let mut params = Vec::new();
        if let Some(label) = label.filter(|l| !l.is_empty()) {
            sql.push_str(&format!(
                " JOIN {} n ON n.id = vr.node_id",
                self.tbl("nodes")
            ));
            clauses.push(format!("n.labels LIKE ?{}", params.len() + 1));
            params.push(SqlValue::Text(format!("%\"{}\"%", label)));
        }
        if let Some(property) = property {
            clauses.push(format!("vr.property = ?{}", params.len() + 1));
            params.push(SqlValue::Text(property.to_string()));
        }
        if !clauses.is_empty() {
            sql.push_str(" WHERE ");
            sql.push_str(&clauses.join(" AND "));
        }
        sql.push_str(" ORDER BY vr.node_id, vr.property");
        let rows = self.exec.query_all_json(&sql, &params)?;
        let mut out = Vec::with_capacity(rows.len());
        for row in rows {
            let node_id = json_to_i64(&row[0])?;
            let property = json_to_string(&row[1])?;
            let blob = VectorBlob::from_hex(&json_to_string(&row[2])?)?;
            out.push(crate::vector::VectorRow {
                node_id,
                property,
                vector: blob.decode()?,
            });
        }
        Ok(out)
    }

    /// Fetch one persisted vector row by node id and property.
    pub fn get_vector_row(
        &self,
        node_id: NodeId,
        property: &str,
    ) -> Result<Option<crate::vector::VectorRow>> {
        let row = self.exec.query_one_json(
            &format!(
                "SELECT node_id, property, vector FROM {} WHERE node_id = ?1 AND property = ?2",
                self.tbl("vector_rows")
            ),
            &[
                SqlValue::from(node_id.0),
                SqlValue::Text(property.to_string()),
            ],
        )?;
        let Some(row) = row else {
            return Ok(None);
        };
        let blob = VectorBlob::from_hex(&json_to_string(&row[2])?)?;
        Ok(Some(crate::vector::VectorRow {
            node_id: json_to_i64(&row[0])?,
            property: json_to_string(&row[1])?,
            vector: blob.decode()?,
        }))
    }

    /// Explicitly upsert a vector row. Node property writes populate rows
    /// automatically; this API is for callers that keep vectors out of the
    /// node's JSON properties (for example, a sidecar embedding table).
    pub fn upsert_vector_row(
        &self,
        node_id: NodeId,
        property: &str,
        vector: &VectorValue,
    ) -> Result<()> {
        if property.is_empty() {
            return Err(Error::Vector("vector property must not be empty".into()));
        }
        if vector.dim() == 0 {
            return Err(Error::Vector("vector must not be empty".into()));
        }
        let node = self.get_node(node_id)?;
        let mut matching = Vec::new();
        for meta in self.vector_indexes()? {
            if meta.property != property || !node.labels.iter().any(|l| l == &meta.label) {
                continue;
            }
            if let Some(dim) = meta.dim {
                if dim != vector.dim() {
                    return Err(Error::Vector(format!(
                        "node {} property {} has dimension {}, index expects {dim}",
                        node_id.0,
                        property,
                        vector.dim()
                    )));
                }
            }
            if let Some(ty) = meta.element_type.as_deref() {
                if ty != vector.element_type().as_str() {
                    return Err(Error::Vector(format!(
                        "node {} property {} has type {}, index expects {ty}",
                        node_id.0,
                        property,
                        vector.element_type().as_str()
                    )));
                }
            }
            matching.push(meta);
        }
        self.insert_vector_row_raw(node_id, property, vector)?;
        for meta in matching {
            self.upsert_index_vector(meta.id, node_id, vector)?;
        }
        self.invalidate_all_indexes();
        Ok(())
    }

    fn insert_vector_row_raw(
        &self,
        node_id: NodeId,
        property: &str,
        vector: &VectorValue,
    ) -> Result<()> {
        let encoded = VectorBlob::encode(vector).to_hex();
        self.exec.execute(
            &format!(
                "INSERT OR REPLACE INTO {} (node_id, property, vector, dim, element_type)
                 VALUES (?1, ?2, ?3, ?4, ?5)",
                self.tbl("vector_rows")
            ),
            &[
                SqlValue::from(node_id.0),
                SqlValue::Text(property.to_string()),
                SqlValue::Text(encoded),
                SqlValue::from(vector.dim() as i64),
                SqlValue::Text(vector.element_type().as_str().to_string()),
            ],
        )?;
        Ok(())
    }

    fn replace_node_vector_rows(
        &self,
        node_id: NodeId,
        rows: &[(String, VectorValue)],
        remove: &HashSet<String>,
    ) -> Result<()> {
        for property in remove {
            self.exec.execute(
                &format!(
                    "DELETE FROM {} WHERE node_id = ?1 AND property = ?2",
                    self.tbl("vector_rows")
                ),
                &[SqlValue::from(node_id.0), SqlValue::Text(property.clone())],
            )?;
        }
        for (property, vector) in rows {
            self.insert_vector_row_raw(node_id, property, vector)?;
        }
        self.invalidate_all_indexes();
        Ok(())
    }

    /// Re-attach every persisted row for `node_id` to the indexes whose label
    /// and property match. This keeps explicit sidecar rows indexed after an
    /// unrelated property or label update, which clears the node's index rows
    /// first.
    fn reindex_node_vector_rows(&self, node_id: NodeId) -> Result<()> {
        let node = self.get_node(node_id)?;
        let indexes = self.vector_indexes()?;
        if indexes.is_empty() {
            return Ok(());
        }
        let rows = self.exec.query_all_json(
            &format!(
                "SELECT property, vector FROM {} WHERE node_id = ?1",
                self.tbl("vector_rows")
            ),
            &[SqlValue::from(node_id.0)],
        )?;
        for row in rows {
            let property = json_to_string(&row[0])?;
            let vector = VectorBlob::from_hex(&json_to_string(&row[1])?)?.decode()?;
            for meta in &indexes {
                if meta.property != property
                    || !node.labels.iter().any(|label| label == &meta.label)
                {
                    continue;
                }
                if let Some(dim) = meta.dim {
                    if dim != vector.dim() {
                        return Err(Error::Vector(format!(
                            "node {} property {} has dimension {}, index expects {dim}",
                            node_id.0,
                            property,
                            vector.dim()
                        )));
                    }
                }
                if let Some(ty) = meta.element_type.as_deref() {
                    if ty != vector.element_type().as_str() {
                        return Err(Error::Vector(format!(
                            "node {} property {} has type {}, index expects {ty}",
                            node_id.0,
                            property,
                            vector.element_type().as_str()
                        )));
                    }
                }
                self.upsert_index_vector(meta.id, node_id, &vector)?;
            }
        }
        Ok(())
    }

    /// ANN search against the index on `label.property`. If no index exists,
    /// falls back to an exact scan of the label's nodes using L2 distance.
    pub fn vector_search(
        &self,
        label: &str,
        property: &str,
        query: &VectorValue,
        k: usize,
    ) -> Result<Vec<(i64, f32)>> {
        if k == 0 {
            return Ok(Vec::new());
        }
        let Some(meta) = self.find_vector_index(label, property)? else {
            return self.exact_vector_search(label, property, query, k, Distance::L2);
        };
        if let Some(dim) = meta.dim {
            if dim != query.dim() {
                return Err(Error::Vector(format!(
                    "query vector dimension {} does not match index dimension {dim}",
                    query.dim()
                )));
            }
        }
        if let Some(ty) = &meta.element_type {
            if ty != query.element_type().as_str() {
                return Err(Error::Vector(format!(
                    "query vector type {} does not match index type {ty}",
                    query.element_type().as_str()
                )));
            }
        }

        let key = (label.to_string(), property.to_string());
        {
            let guard = self
                .vector_index_cache
                .lock()
                .map_err(|e| Error::Storage(format!("mutex: {e}")))?;
            if let Some(index) = guard.get(&key) {
                return Ok(index.search(query, k));
            }
        }

        let config = meta.config()?;
        let rows = self.exec.query_all_json(
            &format!(
                "SELECT node_id, vector FROM {} WHERE index_id = ?1",
                self.tbl("vectors")
            ),
            &[SqlValue::from(meta.id)],
        )?;
        let mut vectors = Vec::with_capacity(rows.len());
        for row in rows {
            let node_id = json_to_i64(&row[0])?;
            let encoded = json_to_string(&row[1])?;
            let blob = VectorBlob::from_hex(&encoded)?;
            vectors.push((node_id, blob.decode()?));
        }
        let index = VectorIndex::build(&config, vectors)?;
        let results = index.search(query, k);
        let mut guard = self
            .vector_index_cache
            .lock()
            .map_err(|e| Error::Storage(format!("mutex: {e}")))?;
        guard.insert(key, Arc::new(index));
        Ok(results)
    }

    /// Exact scan fallback used when no index is defined. If `label` is empty,
    /// every node is scanned.
    pub fn exact_vector_search(
        &self,
        label: &str,
        property: &str,
        query: &VectorValue,
        k: usize,
        metric: Distance,
    ) -> Result<Vec<(i64, f32)>> {
        let rows = self.vector_rows((!label.is_empty()).then_some(label), Some(property))?;
        let mut stored = Vec::new();
        for row in &rows {
            if row.vector.dim() != query.dim() || row.vector.element_type() != query.element_type()
            {
                continue;
            }
            stored.push((row.node_id, row.vector.clone()));
        }
        if !stored.is_empty() {
            return Ok(crate::vector::knn_search(query, &stored, k, metric));
        }
        // Legacy/backfill fallback: rows are populated on node writes, but an
        // older database may not have them yet.
        let nodes = self.list_nodes_by_label(label)?;
        for node in nodes {
            let Some(value) = resolve(&node.properties, property) else {
                continue;
            };
            if value.is_null() {
                continue;
            }
            let vector = VectorValue::from_json(value).map_err(|e| {
                Error::Vector(format!("node {} property {}: {e}", node.id.0, property))
            })?;
            if vector.dim() != query.dim() || vector.element_type() != query.element_type() {
                continue;
            }
            stored.push((node.id.0, vector));
        }
        Ok(crate::vector::knn_search(query, &stored, k, metric))
    }

    /// Build the persistent vector rows implied by a node's current state.
    ///
    /// Top-level arrays of numbers and typed vector objects are detected
    /// automatically. Hex strings and nested property paths are also picked up
    /// when the label/property is declared by a vector index, avoiding false
    /// positives on ordinary string identifiers.
    fn prepare_node_vector_rows(
        &self,
        labels: &[String],
        properties: &Value,
    ) -> Result<Vec<(String, VectorValue)>> {
        let mut detected: HashMap<String, VectorValue> = HashMap::new();
        if let Some(map) = properties.as_object() {
            for (property, value) in map {
                if let Some(vector) = detect_vector_value(value) {
                    detected.insert(property.clone(), vector);
                }
            }
        }
        for meta in self.vector_indexes()? {
            if !labels.iter().any(|l| l == &meta.label) {
                continue;
            }
            let Some(value) = resolve(properties, &meta.property) else {
                continue;
            };
            if value.is_null() {
                continue;
            }
            if let Ok(vector) = VectorValue::from_json(value) {
                detected.insert(meta.property.clone(), vector);
            }
        }
        let mut rows: Vec<(String, VectorValue)> = detected.into_iter().collect();
        rows.sort_by(|a, b| a.0.cmp(&b.0));
        Ok(rows)
    }

    fn prepare_node_vectors(
        &self,
        labels: &[String],
        properties: &Value,
    ) -> Result<Vec<(i64, VectorValue)>> {
        let mut out = Vec::new();
        for meta in self.vector_indexes()? {
            if !labels.iter().any(|l| l == &meta.label) {
                continue;
            }
            let Some(value) = resolve(properties, &meta.property) else {
                continue;
            };
            if value.is_null() {
                continue;
            }
            let vector = VectorValue::from_json(value).map_err(|e| {
                Error::Vector(format!(
                    "node property {}.{}: {e}",
                    meta.label, meta.property
                ))
            })?;
            if let Some(dim) = meta.dim {
                if dim != vector.dim() {
                    return Err(Error::Vector(format!(
                        "vector for {}.{} has dimension {}, index expects {dim}",
                        meta.label,
                        meta.property,
                        vector.dim()
                    )));
                }
            }
            if let Some(ty) = &meta.element_type {
                if ty != vector.element_type().as_str() {
                    return Err(Error::Vector(format!(
                        "vector for {}.{} has type {}, index expects {ty}",
                        meta.label,
                        meta.property,
                        vector.element_type().as_str()
                    )));
                }
            }
            out.push((meta.id, vector));
        }
        Ok(out)
    }

    fn apply_node_vectors(&self, node_id: NodeId, prepared: &[(i64, VectorValue)]) -> Result<()> {
        for (index_id, vector) in prepared {
            self.upsert_index_vector(*index_id, node_id, vector)?;
        }
        Ok(())
    }

    fn upsert_index_vector(
        &self,
        index_id: i64,
        node_id: NodeId,
        vector: &VectorValue,
    ) -> Result<()> {
        let encoded = VectorBlob::encode(vector).to_hex();
        self.exec.execute(
            &format!(
                "INSERT OR REPLACE INTO {} (index_id, node_id, vector) VALUES (?1, ?2, ?3)",
                self.tbl("vectors")
            ),
            &[
                SqlValue::from(index_id),
                SqlValue::from(node_id.0),
                SqlValue::Text(encoded),
            ],
        )?;
        self.exec.execute(
            &format!(
                "UPDATE {} SET dim = COALESCE(dim, ?1), element_type = COALESCE(element_type, ?2)
                 WHERE id = ?3",
                self.tbl("vector_indexes")
            ),
            &[
                SqlValue::from(vector.dim() as i64),
                SqlValue::Text(vector.element_type().as_str().to_string()),
                SqlValue::from(index_id),
            ],
        )?;
        self.invalidate_vector_meta_cache();
        self.invalidate_all_indexes();
        Ok(())
    }

    fn clear_node_vectors(&self, node_id: NodeId) -> Result<()> {
        self.clear_node_index_vectors(node_id)?;
        self.exec.execute(
            &format!("DELETE FROM {} WHERE node_id = ?1", self.tbl("vector_rows")),
            &[SqlValue::from(node_id.0)],
        )?;
        self.invalidate_all_indexes();
        Ok(())
    }

    fn clear_node_index_vectors(&self, node_id: NodeId) -> Result<()> {
        self.exec.execute(
            &format!("DELETE FROM {} WHERE node_id = ?1", self.tbl("vectors")),
            &[SqlValue::from(node_id.0)],
        )?;
        self.invalidate_all_indexes();
        Ok(())
    }

    fn invalidate_vector_meta_cache(&self) {
        if let Ok(mut guard) = self.vector_meta_cache.lock() {
            *guard = None;
        }
    }

    fn invalidate_all_indexes(&self) {
        if let Ok(mut guard) = self.vector_index_cache.lock() {
            guard.clear();
        }
    }

    pub fn count_nodes(&self) -> Result<i64> {
        let row = self
            .exec
            .query_one_json(&format!("SELECT COUNT(*) FROM {}", self.tbl("nodes")), &[])?;
        Ok(row
            .and_then(|r| r.first().and_then(|v| v.as_i64()))
            .unwrap_or(0))
    }

    pub fn count_edges(&self) -> Result<i64> {
        let row = self
            .exec
            .query_one_json(&format!("SELECT COUNT(*) FROM {}", self.tbl("edges")), &[])?;
        Ok(row
            .and_then(|r| r.first().and_then(|v| v.as_i64()))
            .unwrap_or(0))
    }

    pub fn dump(&self) -> Result<HashMap<String, Value>> {
        let mut out = HashMap::new();
        let n = self.count_nodes()?;
        let e = self.count_edges()?;
        out.insert("nodes".into(), Value::from(n));
        out.insert("edges".into(), Value::from(e));
        Ok(out)
    }
}

/// Best-effort detection of vector-valued properties without confusing ordinary
/// string arrays (tags, aliases) or scalar identifiers for vectors.
fn detect_vector_value(value: &Value) -> Option<VectorValue> {
    match value {
        Value::Array(items)
            if !items.is_empty() && items.iter().all(|item| item.is_number() || item.is_null()) =>
        {
            VectorValue::from_json(value).ok()
        }
        Value::Object(map) if map.contains_key("type") => VectorValue::from_json(value).ok(),
        _ => None,
    }
}

fn json_to_i64(v: &Value) -> Result<i64> {
    match v {
        Value::Number(n) => n
            .as_i64()
            .ok_or_else(|| Error::Storage(format!("expected integer, got {v}"))),
        Value::String(s) => s.parse::<i64>().map_err(|e| Error::Storage(e.to_string())),
        Value::Null => Err(Error::Storage("expected integer, got NULL".into())),
        _ => Err(Error::Storage(format!("expected integer, got {v}"))),
    }
}
fn json_to_string(v: &Value) -> Result<String> {
    match v {
        Value::String(s) => Ok(s.clone()),
        Value::Null => Err(Error::Storage("expected string, got NULL".into())),
        other => Ok(other.to_string()),
    }
}

/// Validate a label/property name before it is interpolated into DDL. The
/// query language restricts these to plain identifiers, matching the FTS
/// surface.
pub(crate) fn validate_ident(kind: &str, s: &str) -> Result<()> {
    if s.is_empty() {
        return Err(Error::Parse {
            message: format!("{kind} must not be empty"),
            position: None,
        });
    }
    let mut chars = s.chars();
    let first = chars.next().unwrap();
    if !(first.is_ascii_alphabetic() || first == '_') {
        return Err(Error::Parse {
            message: format!("{kind} '{s}' must start with a letter or underscore"),
            position: None,
        });
    }
    for c in chars {
        if !(c.is_ascii_alphanumeric() || c == '_') {
            return Err(Error::Parse {
                message: format!("{kind} '{s}' contains invalid character '{c}'"),
                position: None,
            });
        }
    }
    Ok(())
}

#[derive(Debug, Clone, Default)]
pub struct StorageSnapshot {
    pub nodes: Vec<Node>,
    pub edges: Vec<Edge>,
}

impl StorageSnapshot {
    pub fn empty() -> Self {
        Self::default()
    }
}
