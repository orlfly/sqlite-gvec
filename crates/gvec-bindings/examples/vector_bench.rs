//! Vector retrieval benchmark for gvec.
//!
//! Run with:
//! ```text
//! cargo run --release -p gvec --example vector_bench
//! ```
//!
//! The benchmark uses deterministic synthetic data (no `rand` dependency)
//! and reports build time, query latency percentiles, and recall@k against
//! exact brute-force ground truth.

use std::time::Instant;

use gvec_core::vector::{knn_search, Distance, ElementType, VectorValue};
use gvec_core::vector_index::{VectorIndex, VectorIndexAlgorithm, VectorIndexConfig};
use serde_json::json;

fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(splitmix64(seed))
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = splitmix64(self.0);
        self.0
    }

    fn next_f32(&mut self) -> f32 {
        // 24 random bits in [0, 1)
        ((self.next_u64() >> 40) as f32) / ((1u32 << 24) as f32)
    }

    fn next_i8(&mut self) -> i8 {
        (self.next_u64() % 256) as u8 as i8
    }
}

fn dataset(n: usize, dim: usize, element: ElementType) -> Vec<(i64, VectorValue)> {
    let mut rng = Rng::new(0x9E37_79B9_7F4A_7C15);
    (0..n)
        .map(|id| {
            let vector = match element {
                ElementType::Float32 => {
                    VectorValue::F32((0..dim).map(|_| rng.next_f32()).collect())
                }
                ElementType::Int8 => VectorValue::I8((0..dim).map(|_| rng.next_i8()).collect()),
                ElementType::Binary => {
                    let bytes = dim.div_ceil(8);
                    VectorValue::Bits((0..bytes).map(|_| rng.next_u64() as u8).collect())
                }
            };
            (id as i64, vector)
        })
        .collect()
}

fn queries(count: usize, dim: usize, element: ElementType) -> Vec<VectorValue> {
    let mut rng = Rng::new(0xD1B5_4A32_D192_ED03);
    (0..count)
        .map(|_| match element {
            ElementType::Float32 => VectorValue::F32((0..dim).map(|_| rng.next_f32()).collect()),
            ElementType::Int8 => VectorValue::I8((0..dim).map(|_| rng.next_i8()).collect()),
            ElementType::Binary => {
                let bytes = dim.div_ceil(8);
                VectorValue::Bits((0..bytes).map(|_| rng.next_u64() as u8).collect())
            }
        })
        .collect()
}

/// Clustered data approximates real embeddings: a small number of dense
/// regions rather than uniform noise across the whole cube.
fn clustered_data(
    n: usize,
    dim: usize,
    clusters: usize,
    query_count: usize,
) -> (Vec<(i64, VectorValue)>, Vec<VectorValue>) {
    let mut rng = Rng::new(0xA24B_AED4_963E_E407);
    let centers: Vec<Vec<f32>> = (0..clusters)
        .map(|_| (0..dim).map(|_| rng.next_f32()).collect())
        .collect();
    let data = (0..n)
        .map(|id| {
            let center = &centers[id % clusters];
            let vector = center
                .iter()
                .map(|x| x + (rng.next_f32() - 0.5) * 0.04)
                .collect();
            (id as i64, VectorValue::F32(vector))
        })
        .collect();
    let queries = (0..query_count)
        .map(|i| {
            let center = &centers[(i * 7) % clusters];
            VectorValue::F32(
                center
                    .iter()
                    .map(|x| x + (rng.next_f32() - 0.5) * 0.04)
                    .collect(),
            )
        })
        .collect();
    (data, queries)
}

fn recall_at_k(exact: &[i64], got: &[(i64, f32)]) -> f64 {
    if exact.is_empty() {
        return 1.0;
    }
    let hits = got.iter().filter(|(id, _)| exact.contains(id)).count();
    hits as f64 / exact.len() as f64
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((sorted.len() - 1) as f64 * p).round() as usize;
    sorted[idx]
}

struct Report {
    algorithm: &'static str,
    metric: Distance,
    element: ElementType,
    build_ms: f64,
    p50_ms: f64,
    p95_ms: f64,
    recall: f64,
}

#[allow(clippy::too_many_arguments)]
fn run_case(
    n: usize,
    dim: usize,
    element: ElementType,
    algorithm: VectorIndexAlgorithm,
    metric: Distance,
    params: serde_json::Value,
    query_count: usize,
    k: usize,
) -> Report {
    let data = dataset(n, dim, element);
    let queries = queries(query_count, dim, element);
    run_case_with_data(element, algorithm, metric, params, queries, data, k)
}

#[allow(clippy::too_many_arguments)]
fn run_case_with_data(
    element: ElementType,
    algorithm: VectorIndexAlgorithm,
    metric: Distance,
    params: serde_json::Value,
    queries: Vec<VectorValue>,
    data: Vec<(i64, VectorValue)>,
    k: usize,
) -> Report {
    let exact_start = Instant::now();
    let exact: Vec<Vec<i64>> = queries
        .iter()
        .map(|q| {
            knn_search(q, &data, k, metric)
                .into_iter()
                .map(|(id, _)| id)
                .collect()
        })
        .collect();
    let exact_total = exact_start.elapsed();

    let config = VectorIndexConfig {
        algorithm,
        metric,
        params,
    };
    let build_start = Instant::now();
    let index = VectorIndex::build(&config, data.clone()).unwrap();
    let build_ms = build_start.elapsed().as_secs_f64() * 1000.0;

    let mut latencies = Vec::with_capacity(queries.len());
    let mut recall_sum = 0.0;
    for (query, expected) in queries.iter().zip(exact.iter()) {
        let start = Instant::now();
        let got = index.search(query, k);
        latencies.push(start.elapsed().as_secs_f64() * 1000.0);
        recall_sum += recall_at_k(expected, &got);
    }
    latencies.sort_by(|a, b| a.total_cmp(b));

    let mut report = Report {
        algorithm: match algorithm {
            VectorIndexAlgorithm::Flat => "flat",
            VectorIndexAlgorithm::Hnsw => "hnsw",
            VectorIndexAlgorithm::Ivf => "ivf",
        },
        metric,
        element,
        build_ms,
        p50_ms: percentile(&latencies, 0.50),
        p95_ms: percentile(&latencies, 0.95),
        recall: recall_sum / queries.len() as f64,
    };
    // Brute-force latency is useful context for the flat case and the
    // ground-truth cost. For non-flat runs we do not repeat it.
    if matches!(algorithm, VectorIndexAlgorithm::Flat) {
        report.build_ms = 0.0;
        report.p50_ms = (exact_total.as_secs_f64() * 1000.0) / queries.len() as f64;
        report.p95_ms = report.p50_ms;
    }
    report
}

fn element_name(element: ElementType) -> &'static str {
    match element {
        ElementType::Float32 => "f32",
        ElementType::Int8 => "i8",
        ElementType::Binary => "bin",
    }
}

fn metric_name(metric: Distance) -> &'static str {
    match metric {
        Distance::L2 => "l2",
        Distance::Cosine => "cosine",
        Distance::InnerProduct => "ip",
    }
}

fn print_table(title: &str, n: usize, dim: usize, k: usize, reports: &[Report]) {
    println!("\n{title}");
    println!("dataset: n={n}, dim={dim}, queries=100, k={k}, exact ground truth = brute force");
    println!(
        "{:<8} {:<8} {:<5} {:>10} {:>10} {:>10} {:>10}",
        "algo", "element", "metric", "build_ms", "p50_ms", "p95_ms", "recall@k"
    );
    for report in reports {
        println!(
            "{:<8} {:<8} {:<5} {:>10.2} {:>10.4} {:>10.4} {:>10.3}",
            report.algorithm,
            element_name(report.element),
            metric_name(report.metric),
            report.build_ms,
            report.p50_ms,
            report.p95_ms,
            report.recall
        );
    }
}

fn print_hnsw_ef_sweep(
    n: usize,
    dim: usize,
    k: usize,
    query_count: usize,
    metric: Distance,
    params: serde_json::Value,
    ef_values: &[usize],
) {
    let data = dataset(n, dim, ElementType::Float32);
    let queries = queries(query_count, dim, ElementType::Float32);
    let exact: Vec<Vec<i64>> = queries
        .iter()
        .map(|q| {
            knn_search(q, &data, k, metric)
                .into_iter()
                .map(|(id, _)| id)
                .collect()
        })
        .collect();
    let index = VectorIndex::build(
        &VectorIndexConfig {
            algorithm: VectorIndexAlgorithm::Hnsw,
            metric,
            params,
        },
        data,
    )
    .unwrap();

    println!(
        "\nHNSW ef_search sweep (n={n}, dim={dim}, k={k}, {})",
        metric_name(metric)
    );
    println!(
        "{:<10} {:>10} {:>10} {:>10}",
        "ef_search", "p50_ms", "p95_ms", "recall@k"
    );
    for &ef in ef_values {
        let mut latencies = Vec::with_capacity(queries.len());
        let mut recall_sum = 0.0;
        for (query, expected) in queries.iter().zip(exact.iter()) {
            let start = Instant::now();
            let got = index.search_with_ef(query, k, ef);
            latencies.push(start.elapsed().as_secs_f64() * 1000.0);
            recall_sum += recall_at_k(expected, &got);
        }
        latencies.sort_by(|a, b| a.total_cmp(b));
        println!(
            "{:<10} {:>10.4} {:>10.4} {:>10.3}",
            ef,
            percentile(&latencies, 0.50),
            percentile(&latencies, 0.95),
            recall_sum / queries.len() as f64
        );
    }
}

fn main() {
    println!("gvec vector retrieval benchmark");
    println!("deterministic synthetic data, single-threaded, in-memory");

    let mut reports = Vec::new();
    reports.push(run_case(
        1_000,
        32,
        ElementType::Float32,
        VectorIndexAlgorithm::Flat,
        Distance::L2,
        json!({}),
        100,
        10,
    ));
    reports.push(run_case(
        1_000,
        32,
        ElementType::Float32,
        VectorIndexAlgorithm::Hnsw,
        Distance::L2,
        json!({"m": 16, "ef_construction": 200, "ef_search": 64}),
        100,
        10,
    ));
    reports.push(run_case(
        1_000,
        32,
        ElementType::Float32,
        VectorIndexAlgorithm::Ivf,
        Distance::L2,
        json!({"nlist": 16, "nprobe": 4}),
        100,
        10,
    ));
    print_table("Small dataset (1k x 32d, L2)", 1_000, 32, 10, &reports);
    reports.clear();

    reports.push(run_case(
        10_000,
        128,
        ElementType::Float32,
        VectorIndexAlgorithm::Flat,
        Distance::L2,
        json!({}),
        100,
        10,
    ));
    reports.push(run_case(
        10_000,
        128,
        ElementType::Float32,
        VectorIndexAlgorithm::Hnsw,
        Distance::L2,
        json!({"m": 16, "ef_construction": 200, "ef_search": 64}),
        100,
        10,
    ));
    reports.push(run_case(
        10_000,
        128,
        ElementType::Float32,
        VectorIndexAlgorithm::Hnsw,
        Distance::Cosine,
        json!({"m": 16, "ef_construction": 200, "ef_search": 64}),
        100,
        10,
    ));
    reports.push(run_case(
        10_000,
        128,
        ElementType::Float32,
        VectorIndexAlgorithm::Ivf,
        Distance::L2,
        json!({"nlist": 32, "nprobe": 4}),
        100,
        10,
    ));
    reports.push(run_case(
        10_000,
        128,
        ElementType::Float32,
        VectorIndexAlgorithm::Ivf,
        Distance::L2,
        json!({"nlist": 32, "nprobe": 8}),
        100,
        10,
    ));
    reports.push(run_case(
        10_000,
        128,
        ElementType::Float32,
        VectorIndexAlgorithm::Ivf,
        Distance::L2,
        json!({"nlist": 32, "nprobe": 16}),
        100,
        10,
    ));
    reports.push(run_case(
        10_000,
        128,
        ElementType::Float32,
        VectorIndexAlgorithm::Ivf,
        Distance::L2,
        json!({"nlist": 32, "nprobe": 32}),
        100,
        10,
    ));
    print_table("Large dataset (10k x 128d)", 10_000, 128, 10, &reports);
    reports.clear();

    print_hnsw_ef_sweep(
        10_000,
        128,
        10,
        100,
        Distance::L2,
        json!({"m": 16, "ef_construction": 200, "ef_search": 64}),
        &[32, 64, 128, 256],
    );

    let (clustered, clustered_queries) = clustered_data(10_000, 128, 32, 100);
    reports.push(run_case_with_data(
        ElementType::Float32,
        VectorIndexAlgorithm::Flat,
        Distance::L2,
        json!({}),
        clustered_queries.clone(),
        clustered.clone(),
        10,
    ));
    reports.push(run_case_with_data(
        ElementType::Float32,
        VectorIndexAlgorithm::Hnsw,
        Distance::L2,
        json!({"m": 16, "ef_construction": 200, "ef_search": 64}),
        clustered_queries.clone(),
        clustered.clone(),
        10,
    ));
    reports.push(run_case_with_data(
        ElementType::Float32,
        VectorIndexAlgorithm::Ivf,
        Distance::L2,
        json!({"nlist": 32, "nprobe": 4}),
        clustered_queries,
        clustered,
        10,
    ));
    print_table(
        "Clustered dataset (10k x 128d, 32 clusters, L2)",
        10_000,
        128,
        10,
        &reports,
    );
    reports.clear();

    reports.push(run_case(
        10_000,
        64,
        ElementType::Int8,
        VectorIndexAlgorithm::Flat,
        Distance::L2,
        json!({}),
        100,
        10,
    ));
    reports.push(run_case(
        10_000,
        64,
        ElementType::Int8,
        VectorIndexAlgorithm::Hnsw,
        Distance::L2,
        json!({"m": 16, "ef_construction": 200, "ef_search": 64}),
        100,
        10,
    ));
    reports.push(run_case(
        10_000,
        64,
        ElementType::Int8,
        VectorIndexAlgorithm::Ivf,
        Distance::L2,
        json!({"nlist": 32, "nprobe": 4}),
        100,
        10,
    ));
    print_table("Int8 dataset (10k x 64d, L2)", 10_000, 64, 10, &reports);
    reports.clear();

    reports.push(run_case(
        10_000,
        256,
        ElementType::Binary,
        VectorIndexAlgorithm::Flat,
        Distance::L2,
        json!({}),
        100,
        10,
    ));
    reports.push(run_case(
        10_000,
        256,
        ElementType::Binary,
        VectorIndexAlgorithm::Hnsw,
        Distance::L2,
        json!({"m": 16, "ef_construction": 200, "ef_search": 64}),
        100,
        10,
    ));
    print_table(
        "Binary dataset (10k x 256 bits, L2/hamming)",
        10_000,
        256,
        10,
        &reports,
    );

    println!("\nnotes:");
    println!("- recall@k compares against exact brute-force top-k.");
    println!("- build_ms excludes exact ground-truth computation.");
    println!("- flat rows report brute-force per-query latency as p50/p95.");
}
