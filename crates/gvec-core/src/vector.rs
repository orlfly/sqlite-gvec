use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ElementType {
    Float32,
    Int8,
    Binary,
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
                Ok(VectorValue::I8(self.bytes.iter().map(|b| *b as i8).collect()))
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
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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

fn dot_f32(x: &[f32], y: &[f32]) -> f32 {
    x.iter().zip(y).map(|(a, b)| a * b).sum()
}
fn l2_f32(x: &[f32], y: &[f32]) -> f32 {
    x.iter().zip(y).map(|(a, b)| {
        let d = a - b;
        d * d
    }).sum::<f32>().sqrt()
}
fn cosine_f32(x: &[f32], y: &[f32]) -> f32 {
    let mut dot = 0f32; let mut ax = 0f32; let mut ay = 0f32;
    for (a, b) in x.iter().zip(y) {
        dot += a * b; ax += a * a; ay += b * b;
    }
    if ax == 0.0 || ay == 0.0 { return 1.0; }
    1.0 - dot / (ax.sqrt() * ay.sqrt())
}
fn dot_i8(x: &[i8], y: &[i8]) -> f32 {
    x.iter().zip(y).map(|(a, b)| (*a as f32) * (*b as f32)).sum()
}
fn l2_i8(x: &[i8], y: &[i8]) -> f32 {
    x.iter().zip(y).map(|(a, b)| {
        let d = *a as f32 - *b as f32;
        d * d
    }).sum::<f32>().sqrt()
}
fn cosine_i8(x: &[i8], y: &[i8]) -> f32 {
    let mut dot = 0f32; let mut ax = 0f32; let mut ay = 0f32;
    for (a, b) in x.iter().zip(y) {
        let a = *a as f32; let b = *b as f32;
        dot += a * b; ax += a * a; ay += b * b;
    }
    if ax == 0.0 || ay == 0.0 { return 1.0; }
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
}