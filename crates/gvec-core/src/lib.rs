//! `gvec-core`: pure-Rust core for the gvec graph + vector engine.
//!
//! Holds storage (node/edge/property), vector encoding, the Cypher-like
//! query pipeline, and graph/vector fusion. No SQLite FFI lives here —
//! SQLite integration is the job of `gvec-extension` (via `sqlite-loadable`)
//! and `gvec-bindings` (via `rusqlite`). The `SqlExecutor` trait here is the
//! boundary those crates implement.

pub mod error;
pub mod sql;
pub mod ids;
pub mod property;
pub mod node;
pub mod edge;
pub mod vector;
pub mod storage;
pub mod query;
pub mod fusion;
pub mod fts;

pub mod prelude {
    pub use crate::error::{Error, Result};
    pub use crate::ids::{EdgeId, NodeId};
    pub use crate::node::{Node, NodeRecord};
    pub use crate::edge::{Edge, EdgeRecord};
    pub use crate::vector::{Distance, ElementType, VectorBlob, VectorValue};
    pub use crate::storage::{Storage, StorageSnapshot};
    pub use crate::sql::{SqlExecutor, SqlValue};
    pub use crate::fts::{Tokenizer, fts_table_name};
}

pub use error::{Error, Result};
pub use ids::{EdgeId, NodeId};
pub use node::{Node, NodeRecord};
pub use edge::{Edge, EdgeRecord};
pub use vector::{Distance, ElementType, VectorBlob, VectorValue};
pub use storage::{Storage, StorageSnapshot};
pub use sql::{SqlExecutor, SqlValue};
pub use fts::{Tokenizer, fts_table_name};