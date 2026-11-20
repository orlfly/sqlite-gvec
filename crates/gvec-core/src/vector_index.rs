//! Approximate nearest-neighbour indexes: flat (exact), HNSW, and IVF.
//!
//! The implementations are dependency-free and operate on the logical
//! [`VectorValue`] representation, so the same index types work for float32,
//! int8, and binary vectors.
//!
//! Design notes:
//!
//! - **HNSW** follows the hierarchical navigable-small-world construction
//!   popularised by Malkov & Yashunin and used by Qdrant. Levels are assigned
//!   deterministically from a splitmix64 hash of the vector id, so rebuilding
//!   an index from the same persisted vectors yields the same graph.
//! - **IVF** follows the inverted-file design used by Milvus (`IVF_FLAT`):
//!   k-means (farthest-first initialisation) partitions vectors into `nlist`
//!   lists, and a query probes `nprobe` nearest centroids. Centroids are
//!   recomputed as means for float vectors and as medoids for int8/binary
//!   vectors, keeping the implementation metric-agnostic.
//! - **Flat** is an exact brute-force scan used as a fallback and as a recall
//!   oracle in tests.
//!
//! The index objects themselves are in-memory. Persistence is handled by the
//! storage layer (vector blobs + index metadata); an index is rebuilt lazily
//! and cached after a restart.

use std::cmp::{Ordering, Reverse};
use std::collections::{BinaryHeap, HashMap, HashSet};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::{Error, Result};
use crate::vector::{distance_metric, Distance, VectorValue};

// ===========================================================================
// Configuration
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VectorIndexAlgorithm {
    Flat,
    Hnsw,
    Ivf,
}

impl VectorIndexAlgorithm {
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "flat" | "exact" | "bruteforce" | "brute_force" => Some(Self::Flat),
            "hnsw" => Some(Self::Hnsw),
            "ivf" | "ivf_flat" => Some(Self::Ivf),
            _ => None,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Flat => "flat",
            Self::Hnsw => "hnsw",
            Self::Ivf => "ivf",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorIndexConfig {
    pub algorithm: VectorIndexAlgorithm,
    pub metric: Distance,
    /// Raw options as stored in the index metadata (m, ef_construction, ...).
    pub params: Value,
}

#[derive(Debug, Clone, Copy)]
pub struct HnswParams {
    /// Maximum neighbours per node on layers above 0.
    pub m: usize,
    /// Maximum neighbours per node on layer 0 (conventionally 2*m).
    pub m0: usize,
    pub ef_construction: usize,
    pub ef_search: usize,
}

impl Default for HnswParams {
    fn default() -> Self {
        Self {
            m: 16,
            m0: 32,
            ef_construction: 200,
            ef_search: 64,
        }
    }
}

impl HnswParams {
    pub fn from_json(v: &Value) -> Self {
        let mut p = Self::default();
        if let Some(n) = v.get("m").and_then(|x| x.as_u64()) {
            p.m = (n as usize).max(2);
            p.m0 = p.m * 2;
        }
        if let Some(n) = v.get("m0").and_then(|x| x.as_u64()) {
            p.m0 = (n as usize).max(p.m);
        }
        if let Some(n) = v.get("ef_construction").and_then(|x| x.as_u64()) {
            p.ef_construction = (n as usize).max(p.m);
        }
        if let Some(n) = v.get("ef_search").and_then(|x| x.as_u64()) {
            p.ef_search = (n as usize).max(1);
        }
        p
    }
}

#[derive(Debug, Clone, Copy)]
pub struct IvfParams {
    /// Number of inverted lists (clusters).
    pub nlist: usize,
    /// Number of lists probed per query.
    pub nprobe: usize,
}

impl Default for IvfParams {
    fn default() -> Self {
        Self {
            nlist: 16,
            nprobe: 4,
        }
    }
}

impl IvfParams {
    pub fn from_json(v: &Value) -> Self {
        let mut p = Self::default();
        if let Some(n) = v.get("nlist").and_then(|x| x.as_u64()) {
            p.nlist = (n as usize).max(1);
        }
        if let Some(n) = v.get("nprobe").and_then(|x| x.as_u64()) {
            p.nprobe = (n as usize).max(1);
        }
        if p.nprobe > p.nlist {
            p.nprobe = p.nlist;
        }
        p
    }
}

// ===========================================================================
// Small shared helpers
// ===========================================================================

#[derive(Debug, Clone, Copy)]
struct Scored {
    distance: f32,
    slot: usize,
}

impl PartialEq for Scored {
    fn eq(&self, other: &Self) -> bool {
        self.distance.total_cmp(&other.distance) == Ordering::Equal && self.slot == other.slot
    }
}
impl Eq for Scored {}
impl PartialOrd for Scored {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Scored {
    fn cmp(&self, other: &Self) -> Ordering {
        self.distance
            .total_cmp(&other.distance)
            .then_with(|| self.slot.cmp(&other.slot))
    }
}

fn validate_vectors(vectors: &[(i64, VectorValue)]) -> Result<usize> {
    let mut dim = None;
    let mut ty = None;
    for (_, v) in vectors {
        match dim {
            None => dim = Some(v.dim()),
            Some(d) if d != v.dim() => {
                return Err(Error::Vector(format!(
                    "vector index dimension mismatch: expected {d}, got {}",
                    v.dim()
                )))
            }
            _ => {}
        }
        match ty {
            None => ty = Some(v.element_type()),
            Some(t) if t != v.element_type() => {
                return Err(Error::Vector(
                    "vector index cannot mix element types".into(),
                ))
            }
            _ => {}
        }
    }
    Ok(dim.unwrap_or(0))
}

fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn level_for_id(id: i64, m: usize) -> usize {
    // P(level >= l) = (1 / m)^l. The multiplier mL = 1 / ln(m).
    let m = m.max(2) as f64;
    let mut h = splitmix64(id as u64);
    let mut level = 0usize;
    let threshold = 1.0 / m;
    loop {
        // Map the next 53 bits into [0, 1); treat 0 as a very small value.
        let u = ((h >> 11) as f64 + 1.0) / ((1u64 << 53) as f64);
        if u < threshold {
            level += 1;
            h = splitmix64(h);
            if level >= 32 {
                break;
            }
        } else {
            break;
        }
    }
    level
}

// ===========================================================================
// Flat index
// ===========================================================================

#[derive(Debug, Clone)]
pub struct FlatIndex {
    metric: Distance,
    vectors: Vec<(i64, VectorValue)>,
    dim: usize,
}

impl FlatIndex {
    pub fn build(metric: Distance, vectors: Vec<(i64, VectorValue)>) -> Result<Self> {
        let dim = validate_vectors(&vectors)?;
        Ok(Self {
            metric,
            vectors,
            dim,
        })
    }

    pub fn len(&self) -> usize {
        self.vectors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn search(&self, query: &VectorValue, k: usize) -> Vec<(i64, f32)> {
        crate::vector::knn_search(query, &self.vectors, k, self.metric)
    }
}

// ===========================================================================
// HNSW index
// ===========================================================================

#[derive(Debug, Clone)]
struct HnswNode {
    id: i64,
    vector: VectorValue,
    /// `neighbors[level]` holds slot indexes into `HnswIndex::nodes`.
    neighbors: Vec<Vec<usize>>,
}

#[derive(Debug, Clone)]
pub struct HnswIndex {
    metric: Distance,
    params: HnswParams,
    nodes: Vec<HnswNode>,
    id_to_slot: HashMap<i64, usize>,
    entry: Option<usize>,
    max_level: usize,
    dim: usize,
}

impl HnswIndex {
    pub fn build(
        metric: Distance,
        params: HnswParams,
        vectors: Vec<(i64, VectorValue)>,
    ) -> Result<Self> {
        let dim = validate_vectors(&vectors)?;
        let mut index = Self {
            metric,
            params,
            nodes: Vec::with_capacity(vectors.len()),
            id_to_slot: HashMap::with_capacity(vectors.len()),
            entry: None,
            max_level: 0,
            dim,
        };
        for (id, vector) in vectors {
            if index.id_to_slot.contains_key(&id) {
                continue;
            }
            index.insert(id, vector);
        }
        Ok(index)
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn params(&self) -> HnswParams {
        self.params
    }

    fn distance(&self, slot: usize, query: &VectorValue) -> f32 {
        distance_metric(&self.nodes[slot].vector, query, self.metric)
    }

    fn insert(&mut self, id: i64, vector: VectorValue) {
        let level = level_for_id(id, self.params.m);
        let slot = self.nodes.len();
        self.nodes.push(HnswNode {
            id,
            vector,
            neighbors: vec![Vec::new(); level + 1],
        });
        self.id_to_slot.insert(id, slot);

        let Some(entry) = self.entry else {
            self.entry = Some(slot);
            self.max_level = level;
            return;
        };

        let query = self.nodes[slot].vector.clone();
        let mut ep = entry;
        let mut cur_level = self.max_level;

        // Greedy descent through layers above the new node's level.
        while cur_level > level {
            let found = self.search_layer(&query, &[ep], 1, cur_level);
            if let Some(best) = found.first() {
                ep = best.slot;
            }
            cur_level -= 1;
        }

        let mut entries = vec![ep];
        let top = level.min(self.max_level);
        for lc in (0..=top).rev() {
            let candidates = self.search_layer(&query, &entries, self.params.ef_construction, lc);
            let max_neighbors = if lc == 0 {
                self.params.m0
            } else {
                self.params.m
            };
            let selected = self.select_neighbors(&candidates, max_neighbors, slot);

            self.nodes[slot].neighbors[lc] = selected.clone();
            for &neighbor in &selected {
                self.nodes[neighbor].neighbors[lc].push(slot);
                self.prune_neighbors(neighbor, lc, max_neighbors);
            }

            if candidates.is_empty() {
                entries = vec![ep];
            } else {
                entries = candidates.iter().map(|c| c.slot).collect();
            }
        }

        if level > self.max_level {
            self.max_level = level;
            self.entry = Some(slot);
        }
    }

    fn prune_neighbors(&mut self, slot: usize, level: usize, max_neighbors: usize) {
        if self.nodes[slot].neighbors[level].len() <= max_neighbors {
            return;
        }
        let base = self.nodes[slot].vector.clone();
        let mut neighbors = std::mem::take(&mut self.nodes[slot].neighbors[level]);
        let mut candidates: Vec<Scored> = neighbors
            .drain(..)
            .map(|neighbor| Scored {
                distance: distance_metric(&self.nodes[neighbor].vector, &base, self.metric),
                slot: neighbor,
            })
            .collect();
        candidates.sort_by(|a, b| a.distance.total_cmp(&b.distance));
        self.nodes[slot].neighbors[level] = self.select_neighbors(&candidates, max_neighbors, slot);
    }

    /// HNSW neighbour-selection heuristic: keep a candidate only if it is
    /// closer to the base vector than to every neighbour already selected.
    /// This preserves graph diversity and is the standard Algorithm 4 from
    /// the HNSW paper.
    fn select_neighbors(
        &self,
        candidates: &[Scored],
        max_neighbors: usize,
        exclude: usize,
    ) -> Vec<usize> {
        let mut selected: Vec<usize> = Vec::with_capacity(max_neighbors);
        for candidate in candidates {
            if candidate.slot == exclude || selected.contains(&candidate.slot) {
                continue;
            }
            let mut keep = true;
            for &neighbor in &selected {
                let distance_to_selected = distance_metric(
                    &self.nodes[candidate.slot].vector,
                    &self.nodes[neighbor].vector,
                    self.metric,
                );
                if distance_to_selected < candidate.distance {
                    keep = false;
                    break;
                }
            }
            if keep {
                selected.push(candidate.slot);
                if selected.len() >= max_neighbors {
                    break;
                }
            }
        }
        selected
    }

    fn search_layer(
        &self,
        query: &VectorValue,
        entries: &[usize],
        ef: usize,
        level: usize,
    ) -> Vec<Scored> {
        let ef = ef.max(1);
        let mut visited: HashSet<usize> = HashSet::new();
        // Nearest-first candidate queue.
        let mut candidates: BinaryHeap<Reverse<Scored>> = BinaryHeap::new();
        // Farthest-first bounded result set.
        let mut results: BinaryHeap<Scored> = BinaryHeap::new();

        for &e in entries {
            if e >= self.nodes.len() || !visited.insert(e) {
                continue;
            }
            let s = Scored {
                distance: self.distance(e, query),
                slot: e,
            };
            candidates.push(Reverse(s));
            results.push(s);
        }
        while results.len() > ef {
            results.pop();
        }

        while let Some(Reverse(candidate)) = candidates.pop() {
            let worst = results.peek().map(|s| s.distance).unwrap_or(f32::INFINITY);
            if results.len() >= ef && candidate.distance > worst {
                break;
            }
            let Some(neighbors) = self.nodes[candidate.slot].neighbors.get(level) else {
                continue;
            };
            for &neighbor in neighbors {
                if neighbor >= self.nodes.len() || !visited.insert(neighbor) {
                    continue;
                }
                let distance = self.distance(neighbor, query);
                let worst = results.peek().map(|s| s.distance).unwrap_or(f32::INFINITY);
                if results.len() < ef || distance < worst {
                    let scored = Scored {
                        distance,
                        slot: neighbor,
                    };
                    candidates.push(Reverse(scored));
                    results.push(scored);
                    if results.len() > ef {
                        results.pop();
                    }
                }
            }
        }

        let mut out = results.into_vec();
        out.sort_by(|a, b| a.distance.total_cmp(&b.distance));
        out
    }

    pub fn search(&self, query: &VectorValue, k: usize) -> Vec<(i64, f32)> {
        self.search_with_ef(query, k, self.params.ef_search)
    }

    /// Search with an explicit `ef_search`, useful for tuning recall without
    /// rebuilding the graph.
    pub fn search_with_ef(
        &self,
        query: &VectorValue,
        k: usize,
        ef_search: usize,
    ) -> Vec<(i64, f32)> {
        if k == 0 || self.entry.is_none() {
            return Vec::new();
        }
        let mut ep = self.entry.unwrap();
        let mut cur_level = self.max_level;
        while cur_level > 0 {
            let found = self.search_layer(query, &[ep], 1, cur_level);
            if let Some(best) = found.first() {
                ep = best.slot;
            }
            cur_level -= 1;
        }
        let ef = k.max(ef_search).max(1);
        let found = self.search_layer(query, &[ep], ef, 0);
        found
            .into_iter()
            .take(k)
            .map(|s| (self.nodes[s.slot].id, s.distance))
            .collect()
    }
}

// ===========================================================================
// IVF index
// ===========================================================================

#[derive(Debug, Clone)]
pub struct IvfIndex {
    metric: Distance,
    params: IvfParams,
    vectors: Vec<(i64, VectorValue)>,
    centroids: Vec<VectorValue>,
    lists: Vec<Vec<usize>>,
    dim: usize,
}

impl IvfIndex {
    pub fn build(
        metric: Distance,
        params: IvfParams,
        vectors: Vec<(i64, VectorValue)>,
    ) -> Result<Self> {
        let dim = validate_vectors(&vectors)?;
        let nlist = params
            .nlist
            .min(vectors.len())
            .max(usize::from(!vectors.is_empty()));
        let mut index = Self {
            metric,
            params: IvfParams {
                nlist,
                nprobe: params.nprobe.min(nlist).max(1),
            },
            vectors,
            centroids: Vec::new(),
            lists: vec![Vec::new(); nlist],
            dim,
        };
        index.train();
        Ok(index)
    }

    pub fn len(&self) -> usize {
        self.vectors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn params(&self) -> IvfParams {
        self.params
    }

    fn train(&mut self) {
        let n = self.vectors.len();
        if n == 0 {
            return;
        }
        let k = self.params.nlist.max(1).min(n);
        self.centroids = self.init_centroids(k);
        self.lists = vec![Vec::new(); k];

        let mut assignments = vec![usize::MAX; n];
        for _ in 0..10 {
            let mut changed = false;
            self.lists.iter_mut().for_each(Vec::clear);
            for (i, (_, vector)) in self.vectors.iter().enumerate() {
                let cluster = self.nearest_centroid(vector);
                if assignments[i] != cluster {
                    assignments[i] = cluster;
                    changed = true;
                }
                self.lists[cluster].push(i);
            }
            self.update_centroids();
            if !changed {
                break;
            }
        }
    }

    fn init_centroids(&self, k: usize) -> Vec<VectorValue> {
        let mut chosen = vec![0usize];
        while chosen.len() < k {
            let mut best = 0usize;
            let mut best_distance = -1.0f32;
            for (i, (_, vector)) in self.vectors.iter().enumerate() {
                if chosen.contains(&i) {
                    continue;
                }
                let nearest = chosen
                    .iter()
                    .map(|&c| distance_metric(vector, &self.vectors[c].1, self.metric))
                    .fold(f32::INFINITY, f32::min);
                if nearest > best_distance {
                    best_distance = nearest;
                    best = i;
                }
            }
            chosen.push(best);
        }
        chosen
            .into_iter()
            .map(|i| self.vectors[i].1.clone())
            .collect()
    }

    fn nearest_centroid(&self, vector: &VectorValue) -> usize {
        let mut best = 0usize;
        let mut best_distance = f32::INFINITY;
        for (i, centroid) in self.centroids.iter().enumerate() {
            let d = distance_metric(vector, centroid, self.metric);
            if d < best_distance {
                best_distance = d;
                best = i;
            }
        }
        best
    }

    fn update_centroids(&mut self) {
        for (cluster, list) in self.lists.iter().enumerate() {
            if list.is_empty() {
                continue;
            }
            let centroid = match &self.centroids[cluster] {
                VectorValue::F32(_) => {
                    let mut sum = vec![0f64; self.dim];
                    let mut count = 0f64;
                    for &i in list {
                        if let VectorValue::F32(xs) = &self.vectors[i].1 {
                            for (s, x) in sum.iter_mut().zip(xs) {
                                *s += *x as f64;
                            }
                            count += 1.0;
                        }
                    }
                    VectorValue::F32(
                        sum.into_iter()
                            .map(|s| (s / count.max(1.0)) as f32)
                            .collect(),
                    )
                }
                // Medoid update: choose the member closest to every other
                // member. Works for int8 and binary vectors without inventing
                // arithmetic that does not exist for them.
                _ => {
                    let mut best = list[0];
                    let mut best_sum = f32::INFINITY;
                    for &candidate in list {
                        let sum: f32 = list
                            .iter()
                            .map(|&other| {
                                distance_metric(
                                    &self.vectors[candidate].1,
                                    &self.vectors[other].1,
                                    self.metric,
                                )
                            })
                            .sum();
                        if sum < best_sum {
                            best_sum = sum;
                            best = candidate;
                        }
                    }
                    self.vectors[best].1.clone()
                }
            };
            self.centroids[cluster] = centroid;
        }
    }

    pub fn search(&self, query: &VectorValue, k: usize) -> Vec<(i64, f32)> {
        if k == 0 || self.vectors.is_empty() {
            return Vec::new();
        }
        let mut centroid_order: Vec<(usize, f32)> = self
            .centroids
            .iter()
            .enumerate()
            .map(|(i, c)| (i, distance_metric(query, c, self.metric)))
            .collect();
        centroid_order.sort_by(|a, b| a.1.total_cmp(&b.1));

        let nprobe = self.params.nprobe.max(1).min(centroid_order.len());
        let mut scored: Vec<(i64, f32)> = Vec::new();
        for &(cluster, _) in centroid_order.iter().take(nprobe) {
            for &slot in &self.lists[cluster] {
                let (id, vector) = &self.vectors[slot];
                scored.push((*id, distance_metric(vector, query, self.metric)));
            }
        }
        scored.sort_by(|a, b| a.1.total_cmp(&b.1));
        scored.truncate(k);
        scored
    }
}

// ===========================================================================
// Unified index
// ===========================================================================

#[derive(Debug, Clone)]
pub enum VectorIndex {
    Flat(FlatIndex),
    Hnsw(HnswIndex),
    Ivf(IvfIndex),
}

impl VectorIndex {
    pub fn build(config: &VectorIndexConfig, vectors: Vec<(i64, VectorValue)>) -> Result<Self> {
        match config.algorithm {
            VectorIndexAlgorithm::Flat => Ok(Self::Flat(FlatIndex::build(config.metric, vectors)?)),
            VectorIndexAlgorithm::Hnsw => Ok(Self::Hnsw(HnswIndex::build(
                config.metric,
                HnswParams::from_json(&config.params),
                vectors,
            )?)),
            VectorIndexAlgorithm::Ivf => Ok(Self::Ivf(IvfIndex::build(
                config.metric,
                IvfParams::from_json(&config.params),
                vectors,
            )?)),
        }
    }

    /// HNSW query-time `ef_search` override. Other index types ignore the
    /// hint and perform their normal search.
    pub fn search_with_ef(
        &self,
        query: &VectorValue,
        k: usize,
        ef_search: usize,
    ) -> Vec<(i64, f32)> {
        match self {
            Self::Hnsw(index) => index.search_with_ef(query, k, ef_search),
            _ => self.search(query, k),
        }
    }

    pub fn search(&self, query: &VectorValue, k: usize) -> Vec<(i64, f32)> {
        match self {
            Self::Flat(index) => index.search(query, k),
            Self::Hnsw(index) => index.search(query, k),
            Self::Ivf(index) => index.search(query, k),
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Self::Flat(index) => index.len(),
            Self::Hnsw(index) => index.len(),
            Self::Ivf(index) => index.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn dim(&self) -> usize {
        match self {
            Self::Flat(index) => index.dim(),
            Self::Hnsw(index) => index.dim(),
            Self::Ivf(index) => index.dim(),
        }
    }

    pub fn algorithm(&self) -> VectorIndexAlgorithm {
        match self {
            Self::Flat(_) => VectorIndexAlgorithm::Flat,
            Self::Hnsw(_) => VectorIndexAlgorithm::Hnsw,
            Self::Ivf(_) => VectorIndexAlgorithm::Ivf,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dataset(n: usize) -> Vec<(i64, VectorValue)> {
        (0..n)
            .map(|i| {
                let x = i as f32 * 0.37;
                let y = (i as f32 * 0.11).sin() * 3.0;
                let z = (i as f32 * 0.23).cos() * 2.0;
                (i as i64, VectorValue::F32(vec![x, y, z]))
            })
            .collect()
    }

    fn exact_ids(query: &VectorValue, vectors: &[(i64, VectorValue)], k: usize) -> Vec<i64> {
        crate::vector::knn_search(query, vectors, k, Distance::L2)
            .into_iter()
            .map(|(id, _)| id)
            .collect()
    }

    #[test]
    fn flat_matches_exact_search() {
        let vectors = dataset(200);
        let index = VectorIndex::build(
            &VectorIndexConfig {
                algorithm: VectorIndexAlgorithm::Flat,
                metric: Distance::L2,
                params: Value::Null,
            },
            vectors.clone(),
        )
        .unwrap();
        let query = VectorValue::F32(vec![2.0, 0.5, 1.0]);
        assert_eq!(
            index.search(&query, 10),
            crate::vector::knn_search(&query, &vectors, 10, Distance::L2)
        );
    }

    #[test]
    fn hnsw_recall_is_high() {
        let vectors = dataset(500);
        let index = VectorIndex::build(
            &VectorIndexConfig {
                algorithm: VectorIndexAlgorithm::Hnsw,
                metric: Distance::L2,
                params: serde_json::json!({"m": 16, "ef_construction": 200, "ef_search": 80}),
            },
            vectors.clone(),
        )
        .unwrap();
        let mut hits = 0usize;
        let mut total = 0usize;
        for q in 0..25 {
            let query = VectorValue::F32(vec![
                q as f32 * 0.9 + 0.2,
                (q as f32).cos(),
                (q as f32).sin(),
            ]);
            let expected = exact_ids(&query, &vectors, 10);
            let got: Vec<i64> = index
                .search(&query, 10)
                .into_iter()
                .map(|(id, _)| id)
                .collect();
            hits += got.iter().filter(|id| expected.contains(id)).count();
            total += expected.len();
        }
        let recall = hits as f64 / total as f64;
        assert!(recall >= 0.9, "HNSW recall too low: {recall}");
        // Nearest neighbour is essentially always exact on this dataset.
        let query = VectorValue::F32(vec![2.0, 0.5, 1.0]);
        assert_eq!(
            index.search(&query, 1)[0].0,
            exact_ids(&query, &vectors, 1)[0]
        );
    }

    #[test]
    fn hnsw_search_with_ef_override() {
        let vectors = dataset(300);
        let index = VectorIndex::build(
            &VectorIndexConfig {
                algorithm: VectorIndexAlgorithm::Hnsw,
                metric: Distance::L2,
                params: serde_json::json!({"m": 8, "ef_construction": 100, "ef_search": 8}),
            },
            vectors.clone(),
        )
        .unwrap();
        let query = VectorValue::F32(vec![2.0, 0.5, 1.0]);
        let got = index.search_with_ef(&query, 3, 256);
        assert_eq!(got.len(), 3);
        assert_eq!(got[0].0, exact_ids(&query, &vectors, 1)[0]);
        // A tiny search beam still returns up to k candidates without panicking.
        assert_eq!(index.search_with_ef(&query, 3, 1).len(), 3);
    }

    #[test]
    fn ivf_recall_is_reasonable() {
        let vectors = dataset(500);
        let index = VectorIndex::build(
            &VectorIndexConfig {
                algorithm: VectorIndexAlgorithm::Ivf,
                metric: Distance::L2,
                params: serde_json::json!({"nlist": 12, "nprobe": 4}),
            },
            vectors.clone(),
        )
        .unwrap();
        let query = VectorValue::F32(vec![2.0, 0.5, 1.0]);
        let expected: Vec<i64> = exact_ids(&query, &vectors, 5);
        let got: Vec<i64> = index
            .search(&query, 5)
            .into_iter()
            .map(|(id, _)| id)
            .collect();
        let hits = got.iter().filter(|id| expected.contains(id)).count();
        assert!(
            hits >= 4,
            "IVF recall too low: got {got:?}, expected {expected:?}"
        );
    }

    #[test]
    fn indexes_handle_int8_and_binary() {
        let vectors: Vec<(i64, VectorValue)> = (0..80)
            .map(|i| {
                (
                    i,
                    VectorValue::I8(vec![(i % 11) as i8, (i % 7) as i8 - 3, 4]),
                )
            })
            .collect();
        for algorithm in [
            VectorIndexAlgorithm::Hnsw,
            VectorIndexAlgorithm::Ivf,
            VectorIndexAlgorithm::Flat,
        ] {
            let index = VectorIndex::build(
                &VectorIndexConfig {
                    algorithm,
                    metric: Distance::L2,
                    params: serde_json::json!({"m": 8, "nlist": 6, "nprobe": 3}),
                },
                vectors.clone(),
            )
            .unwrap();
            let query = VectorValue::I8(vec![0, 0, 4]);
            let got = index.search(&query, 3);
            assert_eq!(got.len(), 3, "{algorithm:?}");
        }

        let bits: Vec<(i64, VectorValue)> = (0..64)
            .map(|i| {
                (
                    i,
                    VectorValue::Bits(vec![(i % 255) as u8, (i * 3 % 251) as u8]),
                )
            })
            .collect();
        let index = VectorIndex::build(
            &VectorIndexConfig {
                algorithm: VectorIndexAlgorithm::Hnsw,
                metric: Distance::L2,
                params: serde_json::json!({"m": 8}),
            },
            bits,
        )
        .unwrap();
        let got = index.search(&VectorValue::Bits(vec![0, 0]), 5);
        assert_eq!(got.len(), 5);
    }

    #[test]
    fn rejects_mixed_dimensions() {
        let vectors = vec![
            (1, VectorValue::F32(vec![1.0, 2.0])),
            (2, VectorValue::F32(vec![1.0, 2.0, 3.0])),
        ];
        let err = VectorIndex::build(
            &VectorIndexConfig {
                algorithm: VectorIndexAlgorithm::Hnsw,
                metric: Distance::L2,
                params: Value::Null,
            },
            vectors,
        )
        .unwrap_err();
        assert!(format!("{err}").contains("dimension"));
    }

    #[test]
    fn empty_index_returns_no_results() {
        let index = VectorIndex::build(
            &VectorIndexConfig {
                algorithm: VectorIndexAlgorithm::Hnsw,
                metric: Distance::L2,
                params: Value::Null,
            },
            Vec::new(),
        )
        .unwrap();
        assert!(index.search(&VectorValue::F32(vec![0.0]), 5).is_empty());
        assert_eq!(index.len(), 0);
    }

    #[test]
    fn deterministic_levels() {
        assert_eq!(level_for_id(42, 16), level_for_id(42, 16));
        assert!(level_for_id(42, 2) >= level_for_id(42, 16));
    }

    #[test]
    fn hnsw_rebuild_is_deterministic() {
        let vectors = dataset(300);
        let config = VectorIndexConfig {
            algorithm: VectorIndexAlgorithm::Hnsw,
            metric: Distance::L2,
            params: serde_json::json!({"m": 12, "ef_construction": 120, "ef_search": 40}),
        };
        let a = VectorIndex::build(&config, vectors.clone()).unwrap();
        let b = VectorIndex::build(&config, vectors).unwrap();
        for i in 0..10 {
            let query = VectorValue::F32(vec![i as f32, (i as f32).sin(), 0.5]);
            assert_eq!(a.search(&query, 10), b.search(&query, 10));
        }
    }
}
