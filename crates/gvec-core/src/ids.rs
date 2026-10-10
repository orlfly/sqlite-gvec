/// Stable newtype wrappers around integer ids so we don't mix node and edge ids accidentally.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct NodeId(pub i64);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct EdgeId(pub i64);

impl From<i64> for NodeId {
    fn from(v: i64) -> Self {
        Self(v)
    }
}
impl From<NodeId> for i64 {
    fn from(v: NodeId) -> Self {
        v.0
    }
}
impl From<i64> for EdgeId {
    fn from(v: i64) -> Self {
        Self(v)
    }
}
impl From<EdgeId> for i64 {
    fn from(v: EdgeId) -> Self {
        v.0
    }
}

impl std::fmt::Display for NodeId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "n#{}", self.0)
    }
}
impl std::fmt::Display for EdgeId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "e#{}", self.0)
    }
}

use serde::{Deserialize, Serialize};
