//! Graph-Vector fusion: `vector_nn(...)` predicates and KNN-anchored traversal.
//!
//! Re-exports the KNN helpers from `vector` and provides a small helper to
//! register an external "vector anchor" (a JSON array of f32 values, or a
//! blob hex string) into a query execution. Used by the bindings and
//! extension to wire `MATCH ... WHERE vector_nn(...)` predicates.

use crate::error::{Error, Result};
use crate::storage::Storage;
use crate::vector::{distance_metric as distance, knn_search, Distance, VectorBlob, VectorValue};
use serde_json::Value;

/// Build a `VectorValue` from a SQL scalar argument (JSON array or hex BLOB).
pub fn vector_from_scalar(v: &Value) -> Result<VectorValue> {
    match v {
        Value::Array(xs) => {
            // f32 by default
            let mut out = Vec::with_capacity(xs.len());
            for x in xs {
                match x {
                    Value::Number(n) => out.push(n.as_f64().unwrap_or(0.0) as f32),
                    _ => return Err(Error::Vector("expected number".into())),
                }
            }
            Ok(VectorValue::F32(out))
        }
        Value::String(s) => {
            // hex blob → bytes
            let bytes = hex_decode(s).map_err(|e| Error::Vector(e.to_string()))?;
            // Heuristic: 4 bytes per f32
            if bytes.len() % 4 == 0 {
                let mut out = Vec::with_capacity(bytes.len() / 4);
                for chunk in bytes.chunks_exact(4) {
                    let arr: [u8; 4] = chunk.try_into().unwrap();
                    out.push(f32::from_le_bytes(arr));
                }
                Ok(VectorValue::F32(out))
            } else {
                Ok(VectorValue::I8(bytes.iter().map(|b| *b as i8).collect()))
            }
        }
        _ => Err(Error::Vector("expected JSON array or hex string".into())),
    }
}

fn hex_decode(s: &str) -> std::result::Result<Vec<u8>, String> {
    let s = s.trim_start_matches("0x");
    if s.len() % 2 != 0 {
        return Err("odd length".into());
    }
    let mut out = Vec::with_capacity(s.len() / 2);
    let bytes = s.as_bytes();
    for i in 0..s.len() / 2 {
        let hi = hex_nibble(bytes[i * 2])?;
        let lo = hex_nibble(bytes[i * 2 + 1])?;
        out.push((hi << 4) | lo);
    }
    Ok(out)
}
fn hex_nibble(b: u8) -> std::result::Result<u8, String> {
    match b {
        b'0'..=b'9' => Ok(b - b'0'),
        b'a'..=b'f' => Ok(b - b'a' + 10),
        b'A'..=b'F' => Ok(b - b'A' + 10),
        _ => Err(format!("invalid hex char {:?}", b as char)),
    }
}

/// Extract a node's stored vector (if any) given a property name. Currently a
/// stub: vectors live in the virtual table, so we always return None here.
pub fn node_vector(
    _storage: &Storage,
    _rowid: i64,
    _property: &str,
) -> Result<Option<VectorValue>> {
    Ok(None)
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
