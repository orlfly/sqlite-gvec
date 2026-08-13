use std::collections::HashMap;

use serde_json::Value;

use crate::edge::Edge;
use crate::error::{Error, Result};
use crate::ids::{EdgeId, NodeId};
use crate::node::Node;
use crate::sql::{SqlExecutor, SqlValue};

/// Storage facade. Holds the shadow-table prefix and an owned `SqlExecutor`.
/// Owned (not borrowed) so callers don't need lifetime gymnastics.
pub struct Storage {
    /// The database executor (private; bindings use the public methods on `Storage`).
    pub(crate) exec: Box<dyn SqlExecutor>,
    pub prefix: String,
}

impl Storage {
    /// Borrow the underlying executor. Used by `gvec-bindings` to access
    /// FTS5 functionality.
    pub fn exec_ref(&self) -> &dyn SqlExecutor { &*self.exec }
}

impl Storage {
    pub fn new(exec: impl SqlExecutor + 'static, prefix: impl Into<String>) -> Self {
        Self { exec: Box::new(exec), prefix: prefix.into() }
    }

    /// Construct from an already-boxed executor.
    pub fn with_boxed(exec: Box<dyn SqlExecutor>, prefix: impl Into<String>) -> Self {
        Self { exec, prefix: prefix.into() }
    }

    fn tbl(&self, name: &str) -> String {
        format!("{}_{}", self.prefix, name)
    }

    pub fn ensure_schema(&self) -> Result<()> {
        // Issue one statement at a time; `SqlExecutor::execute` runs a single
        // statement only (rusqlite's `execute` does not handle `;`-separated
        // batches).
        self.exec.execute(&format!(
            "CREATE TABLE IF NOT EXISTS {nodes} (
                id    INTEGER PRIMARY KEY,
                labels TEXT NOT NULL,
                properties TEXT NOT NULL
            )",
            nodes = self.tbl("nodes"),
        ), &[])?;
        self.exec.execute(&format!(
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
        ), &[])?;
        self.exec.execute(&format!(
            "CREATE TABLE IF NOT EXISTS {chunks} (
                chunk_id INTEGER PRIMARY KEY,
                size INTEGER NOT NULL,
                validity BLOB,
                rowids BLOB
            )",
            chunks = self.tbl("chunks"),
        ), &[])?;
        self.exec.execute(&format!(
            "CREATE TABLE IF NOT EXISTS {rowids} (
                rowid INTEGER PRIMARY KEY,
                id INTEGER,
                chunk_id INTEGER,
                chunk_offset INTEGER
            )",
            rowids = self.tbl("rowids"),
        ), &[])?;
        Ok(())
    }

    pub fn drop_all(&self) -> Result<()> {
        let names = ["nodes", "edges", "chunks", "rowids", "auxiliary"];
        for n in names {
            let _ = self.exec.execute(&format!("DROP TABLE IF EXISTS {}", self.tbl(n)), &[]);
        }
        Ok(())
    }

    // ---- Node CRUD ----

    pub fn create_node(&self, labels: &[String], properties: &Value) -> Result<NodeId> {
        self.exec.execute(
            &format!("INSERT INTO {} (labels, properties) VALUES (?1, ?2)", self.tbl("nodes")),
            &[SqlValue::Text(serde_json::to_string(labels)?), SqlValue::Text(serde_json::to_string(properties)?)],
        )?;
        Ok(NodeId(self.exec.last_insert_rowid()))
    }

    pub fn get_node(&self, id: NodeId) -> Result<Node> {
        let row = self.exec.query_one_json(
            &format!("SELECT id, labels, properties FROM {} WHERE id = ?1", self.tbl("nodes")),
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
            ("WHERE labels LIKE ?1", vec![SqlValue::Text(format!("%\"{}\"%", label))])
        };
        let rows = self.exec.query_all_json(
            &format!("SELECT id, labels, properties FROM {} {}", self.tbl("nodes"), where_sql),
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

    pub fn set_node_labels(&self, id: NodeId, labels: &[String]) -> Result<()> {
        let n = self.exec.execute(
            &format!("UPDATE {} SET labels = ?1 WHERE id = ?2", self.tbl("nodes")),
            &[SqlValue::Text(serde_json::to_string(labels)?), SqlValue::from(id.0)],
        )?;
        if n == 0 { return Err(Error::Storage(format!("node {} not found", id.0))); }
        Ok(())
    }

    pub fn delete_node(&self, id: NodeId) -> Result<()> {
        let n = self.exec.execute(
            &format!("DELETE FROM {} WHERE id = ?1", self.tbl("nodes")),
            &[SqlValue::from(id.0)],
        )?;
        if n == 0 { return Err(Error::Storage(format!("node {} not found", id.0))); }
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
        let s_exists = self.exec.query_one_json(
            &format!("SELECT 1 FROM {} WHERE id = ?1", self.tbl("nodes")),
            &[SqlValue::from(source.0)],
        )?.is_some();
        if !s_exists { return Err(Error::Storage(format!("source node {} not found", source.0))); }
        let t_exists = self.exec.query_one_json(
            &format!("SELECT 1 FROM {} WHERE id = ?1", self.tbl("nodes")),
            &[SqlValue::from(target.0)],
        )?.is_some();
        if !t_exists { return Err(Error::Storage(format!("target node {} not found", target.0))); }

        let type_val: SqlValue = match edge_type { Some(s) => SqlValue::Text(s.to_string()), None => SqlValue::Null };
        self.exec.execute(
            &format!("INSERT INTO {} (source, target, edge_type, properties) VALUES (?1, ?2, ?3, ?4)", self.tbl("edges")),
            &[SqlValue::from(source.0), SqlValue::from(target.0), type_val, SqlValue::Text(serde_json::to_string(properties)?)],
        )?;
        Ok(EdgeId(self.exec.last_insert_rowid()))
    }

    pub fn list_edges(&self, source: Option<NodeId>, target: Option<NodeId>) -> Result<Vec<Edge>> {
        let mut sql = format!("SELECT id, source, target, edge_type, properties FROM {}", self.tbl("edges"));
        let params: Vec<SqlValue> = match (source, target) {
            (Some(s), Some(t)) => { sql.push_str(" WHERE source = ?1 AND target = ?2"); vec![SqlValue::from(s.0), SqlValue::from(t.0)] }
            (Some(s), None) => { sql.push_str(" WHERE source = ?1"); vec![SqlValue::from(s.0)] }
            (None, Some(t)) => { sql.push_str(" WHERE target = ?1"); vec![SqlValue::from(t.0)] }
            (None, None) => vec![],
        };
        let rows = self.exec.query_all_json(&sql, &params)?;
        let mut out = Vec::with_capacity(rows.len());
        for row in rows {
            let id = json_to_i64(&row[0])?;
            let source = json_to_i64(&row[1])?;
            let target = json_to_i64(&row[2])?;
            let edge_type = match &row[3] { Value::Null => None, v => Some(json_to_string(v)?) };
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
        if n == 0 { return Err(Error::Storage(format!("edge {} not found", id.0))); }
        Ok(())
    }

    pub fn count_nodes(&self) -> Result<i64> {
        let row = self.exec.query_one_json(
            &format!("SELECT COUNT(*) FROM {}", self.tbl("nodes")),
            &[],
        )?;
        Ok(row.and_then(|r| r.first().and_then(|v| v.as_i64())).unwrap_or(0))
    }

    pub fn count_edges(&self) -> Result<i64> {
        let row = self.exec.query_one_json(
            &format!("SELECT COUNT(*) FROM {}", self.tbl("edges")),
            &[],
        )?;
        Ok(row.and_then(|r| r.first().and_then(|v| v.as_i64())).unwrap_or(0))
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
    match v { Value::Number(n) => n.as_i64().ok_or_else(|| Error::Storage(format!("expected integer, got {v}"))), Value::String(s) => s.parse::<i64>().map_err(|e| Error::Storage(e.to_string())), Value::Null => Err(Error::Storage("expected integer, got NULL".into())), _ => Err(Error::Storage(format!("expected integer, got {v}"))) }
}
fn json_to_string(v: &Value) -> Result<String> {
    match v { Value::String(s) => Ok(s.clone()), Value::Null => Err(Error::Storage("expected string, got NULL".into())), other => Ok(other.to_string()) }
}

#[derive(Debug, Clone, Default)]
pub struct StorageSnapshot {
    pub nodes: Vec<Node>,
    pub edges: Vec<Edge>,
}

impl StorageSnapshot { pub fn empty() -> Self { Self::default() } }