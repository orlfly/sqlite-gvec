//! `gvec-core`: pure-Rust core for the gvec graph + vector engine.
//!
//! Holds storage (node/edge/property), vector encoding, the Cypher-like
//! query pipeline, and graph/vector fusion. No SQLite FFI lives here —
//! SQLite integration is the job of `gvec-extension` (via `sqlite-loadable`)
//! and `gvec-bindings` (via `rusqlite`). The `SqlExecutor` trait here is the
//! boundary those crates implement.

pub mod edge;
pub mod error;
pub mod fts;
pub mod fusion;
pub mod ids;
pub mod node;
pub mod property;
pub mod query;
pub mod sql;
pub mod storage;
pub mod vector;

pub mod prelude {
    pub use crate::edge::{Edge, EdgeRecord};
    pub use crate::error::{Error, Result};
    pub use crate::fts::{fts_table_name, Tokenizer};
    pub use crate::ids::{EdgeId, NodeId};
    pub use crate::node::{Node, NodeRecord};
    pub use crate::sql::{SqlExecutor, SqlValue};
    pub use crate::storage::{Storage, StorageSnapshot};
    pub use crate::vector::{Distance, ElementType, VectorBlob, VectorValue};
}

pub use edge::{Edge, EdgeRecord};
pub use error::{Error, Result};
pub use fts::{fts_table_name, Tokenizer};
pub use ids::{EdgeId, NodeId};
pub use node::{Node, NodeRecord};
pub use sql::{SqlExecutor, SqlValue};
pub use storage::{Storage, StorageSnapshot};
pub use vector::{Distance, ElementType, VectorBlob, VectorValue};
