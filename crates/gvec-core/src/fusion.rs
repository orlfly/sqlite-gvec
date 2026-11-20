//! Graph-Vector fusion: `vector_nn(...)` predicates and KNN-anchored traversal.
//!
//! Re-exports the KNN helpers from `vector` and provides a small helper to
//! register an external "vector anchor" (a JSON array of f32 values, or a
//! blob hex string) into a query execution. Used by the bindings and
//! extension to wire `MATCH ... WHERE vector_nn(...)` predicates.

use crate::error::Result;
use crate::storage::Storage;
use crate::vector::{distance_metric as distance, knn_search, Distance, VectorBlob, VectorValue};
use serde_json::Value;

/// Build a `VectorValue` from a SQL scalar argument (JSON array or hex BLOB).
pub fn vector_from_scalar(v: &Value) -> Result<VectorValue> {
    VectorValue::from_json(v)
}

/// Extract a node's persisted vector row (or fall back to its JSON property).
pub fn node_vector(storage: &Storage, rowid: i64, property: &str) -> Result<Option<VectorValue>> {
    if let Some(row) = storage.get_vector_row(crate::ids::NodeId(rowid), property)? {
        return Ok(Some(row.vector));
    }
    let node = storage.get_node(crate::ids::NodeId(rowid))?;
    match crate::property::resolve(&node.properties, property) {
        Some(value) if !value.is_null() => Ok(Some(VectorValue::from_json(value)?)),
        _ => Ok(None),
    }
}

/// Convenience: KNN across an in-memory collection, used by tests and the
/// bindings. Distances default to L2.
pub fn knn(
    query: &VectorValue,
    stored: &[(i64, VectorValue)],
    k: usize,
    d: Distance,
) -> Vec<(i64, f32)> {
    knn_search(query, stored, k, d)
}

/// Distance between two vectors, exposed for `vector_dist(a, b, metric)`.
pub fn vector_distance(a: &VectorValue, b: &VectorValue, d: Distance) -> f32 {
    distance(a, b, d)
}

/// Re-export vector blob encoder.
pub fn encode_blob(v: &VectorValue) -> VectorBlob {
    VectorBlob::encode(v)
}

/// Decode a vector blob (used by bindings to read back BLOB storage).
pub fn decode_blob(b: &VectorBlob) -> Result<VectorValue> {
    b.decode()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn parse_json_array() {
        let v = vector_from_scalar(&json!([1.0, 2.0, 3.0])).unwrap();
        assert_eq!(v, VectorValue::F32(vec![1.0, 2.0, 3.0]));
    }

    #[test]
    fn parse_hex_blob() {
        let v = vector_from_scalar(&json!("0000803f0000004000004040")).unwrap();
        assert_eq!(v.dim(), 3);
    }

    #[test]
    fn knn_distance_parity() {
        let q = VectorValue::F32(vec![0.0, 0.0]);
        let stored = vec![
            (1, VectorValue::F32(vec![1.0, 0.0])),
            (2, VectorValue::F32(vec![3.0, 0.0])),
        ];
        let res = knn(&q, &stored, 2, Distance::L2);
        assert_eq!(res[0].0, 1);
        assert_eq!(res[1].0, 2);
    }
}
