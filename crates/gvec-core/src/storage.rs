use std::collections::HashMap;
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
            "vector_indexes",
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

    pub fn set_node_labels(&self, id: NodeId, labels: &[String]) -> Result<()> {
        let node = self.get_node(id)?;
        self.update_node(id, labels, &node.properties)
    }

    /// Update a node's labels and properties, keeping matching vector indexes
    /// in sync. All existing vector rows for the node are replaced atomically
    /// with the vectors implied by the new state.
    pub fn update_node(&self, id: NodeId, labels: &[String], properties: &Value) -> Result<()> {
        let prepared = self.prepare_node_vectors(labels, properties)?;
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
        self.clear_node_vectors(id)?;
        self.apply_node_vectors(id, &prepared)?;
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
        let nodes = self.list_nodes_by_label(label)?;
        let mut stored = Vec::new();
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
            if vector.dim() != query.dim() {
                continue;
            }
            stored.push((node.id.0, vector));
        }
        Ok(crate::vector::knn_search(query, &stored, k, metric))
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
fn validate_ident(kind: &str, s: &str) -> Result<()> {
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
