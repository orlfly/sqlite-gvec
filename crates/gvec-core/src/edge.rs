use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::ids::{EdgeId, NodeId};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Edge {
    pub id: EdgeId,
    pub source: NodeId,
    pub target: NodeId,
    pub edge_type: Option<String>,
    pub properties: Value,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EdgeRecord {
    pub id: i64,
    pub source: i64,
    pub target: i64,
    pub edge_type: Option<String>,
    pub properties: Value,
}

impl From<&Edge> for EdgeRecord {
    fn from(e: &Edge) -> Self {
        Self {
            id: e.id.0,
            source: e.source.0,
            target: e.target.0,
            edge_type: e.edge_type.clone(),
            properties: e.properties.clone(),
        }
    }
}
