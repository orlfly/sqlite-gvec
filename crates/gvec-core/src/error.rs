use std::result;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("storage error: {0}")]
    Storage(String),

    #[error("vector error: {0}")]
    Vector(String),

    #[error("query error: {0}")]
    Query(String),

    #[error("parse error at {position:?}: {message}")]
    Parse {
        message: String,
        position: Option<(usize, usize)>,
    },

    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    #[error("serde error: {0}")]
    Serde(#[from] serde_json::Error),
}

pub type Result<T> = result::Result<T, Error>;
