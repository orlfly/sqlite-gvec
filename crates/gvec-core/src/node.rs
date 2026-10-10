use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::ids::NodeId;

/// In-memory node record. Labels are stored as a JSON array; properties as a JSON object.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Node {
    pub id: NodeId,
    pub labels: Vec<String>,
    pub properties: Value,
}

/// Lightweight view of a node used inside query results (no internal id handling).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NodeRecord {
    pub id: i64,
    pub labels: Vec<String>,
    pub properties: Value,
}

impl From<&Node> for NodeRecord {
    fn from(n: &Node) -> Self {
        Self {
            id: n.id.0,
            labels: n.labels.clone(),
            properties: n.properties.clone(),
        }
    }
}
