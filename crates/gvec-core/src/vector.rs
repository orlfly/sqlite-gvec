use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::error::{Error, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ElementType {
    Float32,
    Int8,
    Binary,
}

impl ElementType {
    pub fn as_str(&self) -> &'static str {
        match self {
            ElementType::Float32 => "float32",
            ElementType::Int8 => "int8",
            ElementType::Binary => "binary",
        }
    }
}

/// Logical vector value carried in memory. Binary is bit-packed into bytes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum VectorValue {
    F32(Vec<f32>),
    I8(Vec<i8>),
    Bits(Vec<u8>),
}

impl VectorValue {
    pub fn dim(&self) -> usize {
        match self {
            VectorValue::F32(v) => v.len(),
            VectorValue::I8(v) => v.len(),
            VectorValue::Bits(v) => v.len() * 8,
        }
    }

    pub fn element_type(&self) -> ElementType {
        match self {
            VectorValue::F32(_) => ElementType::Float32,
            VectorValue::I8(_) => ElementType::Int8,
            VectorValue::Bits(_) => ElementType::Binary,
        }
    }

    /// Parse a vector from the JSON shapes accepted by the query language and
    /// storage layer:
    ///
    /// - `[1.0, 2.0, ...]` -> `Float32`
    /// - `"0000803f00000040"` (hex) -> `Float32` when 4-byte aligned, else `Int8`
    /// - `{"type": "float32"|"f32", "values": [...]}`
    /// - `{"type": "int8", "values": [...]}`
    /// - `{"type": "binary", "bits": [...]}` or `{"type":"binary","hex":"..."}`
    pub fn from_json(v: &Value) -> Result<Self> {
        match v {
            Value::Array(xs) => {
                let mut out = Vec::with_capacity(xs.len());
                for x in xs {
                    match x {
                        Value::Number(n) => out.push(n.as_f64().unwrap_or(0.0) as f32),
                        Value::Null => out.push(0.0),
                        _ => return Err(Error::Vector("expected a JSON array of numbers".into())),
                    }
                }
                Ok(VectorValue::F32(out))
            }
            Value::String(s) => {
                if let Ok(bytes) = hex_decode(s) {
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
                } else {
                    Err(Error::Vector(format!(
                        "cannot parse vector from string '{s}' (expected hex)"
                    )))
                }
            }
            Value::Object(map) => {
                let ty = map
                    .get("type")
                    .and_then(|t| t.as_str())
                    .unwrap_or("float32")
                    .to_ascii_lowercase();
                match ty.as_str() {
                    "float32" | "f32" | "float" => {
                        let xs =
                            map.get("values")
                                .or_else(|| map.get("vector"))
                                .ok_or_else(|| {
                                    Error::Vector("float vector object requires 'values'".into())
                                })?;
                        let mut out = Vec::new();
                        for x in xs
                            .as_array()
                            .ok_or_else(|| Error::Vector("'values' must be an array".into()))?
                        {
                            out.push(x.as_f64().unwrap_or(0.0) as f32);
                        }
                        Ok(VectorValue::F32(out))
                    }
                    "int8" | "i8" => {
                        let xs = map.get("values").ok_or_else(|| {
                            Error::Vector("int8 vector object requires 'values'".into())
                        })?;
                        let mut out = Vec::new();
                        for x in xs
                            .as_array()
                            .ok_or_else(|| Error::Vector("'values' must be an array".into()))?
                        {
                            let n = x
                                .as_i64()
                                .ok_or_else(|| Error::Vector("expected integer".into()))?;
                            out.push(n.clamp(-128, 127) as i8);
                        }
                        Ok(VectorValue::I8(out))
                    }
                    "binary" | "bits" => {
                        if let Some(h) = map.get("hex").and_then(|h| h.as_str()) {
                            return Ok(VectorValue::Bits(hex_decode(h)?));
                        }
                        let xs = map.get("bits").ok_or_else(|| {
                            Error::Vector("binary vector object requires 'bits' or 'hex'".into())
                        })?;
                        let mut bits = Vec::new();
                        for x in xs
                            .as_array()
                            .ok_or_else(|| Error::Vector("'bits' must be an array".into()))?
                        {
                            bits.push(if x.as_i64().unwrap_or(0) != 0 {
                                1u8
                            } else {
                                0u8
                            });
                        }
                        let mut packed = vec![0u8; bits.len().div_ceil(8)];
                        for (i, b) in bits.iter().enumerate() {
                            if *b != 0 {
                                packed[i / 8] |= 1 << (i % 8);
                            }
                        }
                        Ok(VectorValue::Bits(packed))
                    }
                    other => Err(Error::Vector(format!("unknown vector type '{other}'"))),
                }
            }
            _ => Err(Error::Vector(
                "expected a JSON array, hex string, or vector object".into(),
            )),
        }
    }

    pub fn to_json(&self) -> Value {
        match self {
            VectorValue::F32(xs) => json!(xs),
            VectorValue::I8(xs) => json!(xs),
            VectorValue::Bits(xs) => json!(xs),
        }
    }
}

/// Stored, serialised vector: a header (type + dimension) followed by raw bytes.
#[derive(Debug, Clone, PartialEq)]
pub struct VectorBlob {
    pub ty: ElementType,
    pub dim: usize,
    pub bytes: Vec<u8>,
}

impl VectorBlob {
    /// Encode a `VectorValue` into a `VectorBlob` (header: 1 byte ty + 4 bytes dim LE).
    pub fn encode(v: &VectorValue) -> Self {
        let ty = v.element_type();
        let dim = v.dim();
        let bytes = match v {
            VectorValue::F32(xs) => xs.iter().flat_map(|x| x.to_le_bytes()).collect(),
            VectorValue::I8(xs) => xs.iter().map(|x| *x as u8).collect(),
            VectorValue::Bits(xs) => xs.clone(),
        };
        Self { ty, dim, bytes }
    }

    pub fn decode(&self) -> Result<VectorValue> {
        match self.ty {
            ElementType::Float32 => {
                if self.bytes.len() != self.dim * 4 {
                    return Err(Error::Vector(format!(
                        "float32 blob length {} != dim*4 ({})",
                        self.bytes.len(),
                        self.dim * 4
                    )));
                }
                let mut out = Vec::with_capacity(self.dim);
                for chunk in self.bytes.chunks_exact(4) {
                    let arr: [u8; 4] = chunk.try_into().unwrap();
                    out.push(f32::from_le_bytes(arr));
                }
                Ok(VectorValue::F32(out))
            }
            ElementType::Int8 => {
                if self.bytes.len() != self.dim {
                    return Err(Error::Vector(format!(
                        "int8 blob length {} != dim ({})",
                        self.bytes.len(),
                        self.dim
                    )));
                }
                Ok(VectorValue::I8(
                    self.bytes.iter().map(|b| *b as i8).collect(),
                ))
            }
            ElementType::Binary => {
                if self.bytes.len() * 8 != self.dim {
                    return Err(Error::Vector(format!(
                        "binary blob length {} bits {} != dim {}",
                        self.bytes.len(),
                        self.bytes.len() * 8,
                        self.dim
                    )));
                }
                Ok(VectorValue::Bits(self.bytes.clone()))
            }
        }
    }

    /// Hex encoding of the header + payload. Hex is used for SQLite storage
    /// so vectors survive the JSON-shaped `SqlExecutor` result surface.
    pub fn to_hex(&self) -> String {
        let mut bytes = Vec::with_capacity(5 + self.bytes.len());
        bytes.push(match self.ty {
            ElementType::Float32 => 1,
            ElementType::Int8 => 2,
            ElementType::Binary => 3,
        });
        bytes.extend_from_slice(&(self.dim as u32).to_le_bytes());
        bytes.extend_from_slice(&self.bytes);
        const HEX: &[u8; 16] = b"0123456789abcdef";
        let mut out = String::with_capacity(bytes.len() * 2);
        for b in bytes {
            out.push(HEX[(b >> 4) as usize] as char);
            out.push(HEX[(b & 0x0f) as usize] as char);
        }
        out
    }

    pub fn from_hex(s: &str) -> Result<Self> {
        let bytes = hex_decode(s)?;
        if bytes.len() < 5 {
            return Err(Error::Vector("vector blob is too short".into()));
        }
        let ty = match bytes[0] {
            1 => ElementType::Float32,
            2 => ElementType::Int8,
            3 => ElementType::Binary,
            other => return Err(Error::Vector(format!("unknown vector blob type {other}"))),
        };
        let dim = u32::from_le_bytes(bytes[1..5].try_into().unwrap()) as usize;
        Ok(Self {
            ty,
            dim,
            bytes: bytes[5..].to_vec(),
        })
    }
}

fn hex_decode(s: &str) -> Result<Vec<u8>> {
    let s = s.trim_start_matches("0x");
    if !s.len().is_multiple_of(2) {
        return Err(Error::Vector("hex string has odd length".into()));
    }
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(s.len() / 2);
    for i in 0..s.len() / 2 {
        let hi = hex_nibble(bytes[i * 2])?;
        let lo = hex_nibble(bytes[i * 2 + 1])?;
        out.push((hi << 4) | lo);
    }
    Ok(out)
}

fn hex_nibble(b: u8) -> Result<u8> {
    match b {
        b'0'..=b'9' => Ok(b - b'0'),
        b'a'..=b'f' => Ok(b - b'a' + 10),
        b'A'..=b'F' => Ok(b - b'A' + 10),
        _ => Err(Error::Vector(format!(
            "invalid hex character {:?}",
            b as char
        ))),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Distance {
    L2,
    Cosine,
    InnerProduct,
}

impl Distance {
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "l2" | "" | "euclidean" => Some(Distance::L2),
            "cosine" => Some(Distance::Cosine),
            "ip" | "inner_product" | "dot" => Some(Distance::InnerProduct),
            _ => None,
        }
    }
}

pub fn distance_metric(a: &VectorValue, b: &VectorValue, d: Distance) -> f32 {
    if a.element_type() != b.element_type() {
        return f32::INFINITY;
    }
    match (a, b) {
        (VectorValue::F32(x), VectorValue::F32(y)) => match d {
            Distance::L2 => l2_f32(x, y),
            Distance::Cosine => cosine_f32(x, y),
            Distance::InnerProduct => -dot_f32(x, y),
        },
        (VectorValue::I8(x), VectorValue::I8(y)) => match d {
            Distance::L2 => l2_i8(x, y),
            Distance::Cosine => cosine_i8(x, y),
            Distance::InnerProduct => -dot_i8(x, y),
        },
        (VectorValue::Bits(x), VectorValue::Bits(y)) => match d {
            Distance::L2 => hamming(x, y) as f32,
            Distance::Cosine => hamming(x, y) as f32 / (x.len() as u32 * 8).max(1) as f32,
            Distance::InnerProduct => 0.0 - (x.len() as u32 * 8) as f32 + hamming(x, y) as f32,
        },
        _ => f32::INFINITY,
    }
}

/// Convert an ascending distance into a descending similarity score.
///
/// - L2 / Hamming: `1 / (1 + d)`, in `(0, 1]`.
/// - Cosine: `1 - d`, i.e. cosine similarity (can be negative).
/// - Inner product: `-d`, i.e. the raw dot product (larger is better).
pub fn similarity_score(distance: f32, metric: Distance) -> f32 {
    match metric {
        Distance::L2 => 1.0 / (1.0 + distance.max(0.0)),
        Distance::Cosine => 1.0 - distance,
        Distance::InnerProduct => -distance,
    }
}

/// A persisted vector attached to `(node_id, property)`.
#[derive(Debug, Clone, PartialEq)]
pub struct VectorRow {
    pub node_id: i64,
    pub property: String,
    pub vector: VectorValue,
}

fn dot_f32(x: &[f32], y: &[f32]) -> f32 {
    x.iter().zip(y).map(|(a, b)| a * b).sum()
}
fn l2_f32(x: &[f32], y: &[f32]) -> f32 {
    x.iter()
        .zip(y)
        .map(|(a, b)| {
            let d = a - b;
            d * d
        })
        .sum::<f32>()
        .sqrt()
}
fn cosine_f32(x: &[f32], y: &[f32]) -> f32 {
    let mut dot = 0f32;
    let mut ax = 0f32;
    let mut ay = 0f32;
    for (a, b) in x.iter().zip(y) {
        dot += a * b;
        ax += a * a;
        ay += b * b;
    }
    if ax == 0.0 || ay == 0.0 {
        return 1.0;
    }
    1.0 - dot / (ax.sqrt() * ay.sqrt())
}
fn dot_i8(x: &[i8], y: &[i8]) -> f32 {
    x.iter()
        .zip(y)
        .map(|(a, b)| (*a as f32) * (*b as f32))
        .sum()
}
fn l2_i8(x: &[i8], y: &[i8]) -> f32 {
    x.iter()
        .zip(y)
        .map(|(a, b)| {
            let d = *a as f32 - *b as f32;
            d * d
        })
        .sum::<f32>()
        .sqrt()
}
fn cosine_i8(x: &[i8], y: &[i8]) -> f32 {
    let mut dot = 0f32;
    let mut ax = 0f32;
    let mut ay = 0f32;
    for (a, b) in x.iter().zip(y) {
        let a = *a as f32;
        let b = *b as f32;
        dot += a * b;
        ax += a * a;
        ay += b * b;
    }
    if ax == 0.0 || ay == 0.0 {
        return 1.0;
    }
    1.0 - dot / (ax.sqrt() * ay.sqrt())
}
fn hamming(x: &[u8], y: &[u8]) -> u32 {
    x.iter().zip(y).map(|(a, b)| (a ^ b).count_ones()).sum()
}

/// Brute-force KNN: returns up to `k` `(rowid, distance)` pairs sorted ascending by distance.
pub fn knn_search(
    query: &VectorValue,
    stored: &[(i64, VectorValue)],
    k: usize,
    distance: Distance,
) -> Vec<(i64, f32)> {
    let mut scored: Vec<(i64, f32)> = stored
        .iter()
        .map(|(rowid, v)| (*rowid, distance_metric(v, query, distance)))
        .collect();
    scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(k);
    scored
}

/// Convenience alias used by query layer.
pub type Vector = VectorValue;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn float_round_trip() {
        let v = VectorValue::F32(vec![1.0, -2.5, 3.125]);
        let blob = VectorBlob::encode(&v);
        assert_eq!(blob.ty, ElementType::Float32);
        assert_eq!(blob.dim, 3);
        assert_eq!(blob.decode().unwrap(), v);
    }

    #[test]
    fn int8_round_trip() {
        let v = VectorValue::I8(vec![1, -2, 3]);
        let blob = VectorBlob::encode(&v);
        assert_eq!(blob.decode().unwrap(), v);
    }

    #[test]
    fn binary_round_trip() {
        let v = VectorValue::Bits(vec![0b1010_1100, 0b0001_1111]);
        let blob = VectorBlob::encode(&v);
        assert_eq!(blob.decode().unwrap(), v);
    }

    #[test]
    fn l2_matches_expectation() {
        let a = VectorValue::F32(vec![0.0, 0.0]);
        let b = VectorValue::F32(vec![3.0, 4.0]);
        assert!((distance_metric(&a, &b, Distance::L2) - 5.0).abs() < 1e-6);
    }

    #[test]
    fn knn_returns_k_sorted() {
        let stored = vec![
            (1, VectorValue::F32(vec![1.0, 0.0])),
            (2, VectorValue::F32(vec![3.0, 0.0])),
            (3, VectorValue::F32(vec![0.0, 0.0])),
        ];
        let q = VectorValue::F32(vec![0.0, 0.0]);
        let res = knn_search(&q, &stored, 2, Distance::L2);
        assert_eq!(res.len(), 2);
        assert_eq!(res[0].0, 3);
        assert_eq!(res[1].0, 1);
        assert!(res[0].1 < res[1].1);
    }

    #[test]
    fn similarity_score_is_descending() {
        assert!((similarity_score(0.0, Distance::L2) - 1.0).abs() < 1e-6);
        assert!((similarity_score(3.0, Distance::L2) - 0.25).abs() < 1e-6);
        assert!((similarity_score(0.25, Distance::Cosine) - 0.75).abs() < 1e-6);
        assert!((similarity_score(-2.5, Distance::InnerProduct) - 2.5).abs() < 1e-6);
        // Larger distances must never produce larger L2 scores.
        assert!(similarity_score(10.0, Distance::L2) < similarity_score(1.0, Distance::L2));
    }
}
