//! Benchmark for persistent vector rows, exact KNN, and ANN search.
//!
//! Run with:
//! ```text
//! cargo run --release -p gvec --example vector_rows_bench
//! ```
//!
//! The dataset is deterministic (no `rand` dependency). The benchmark reports
//! node/row write throughput, exact KNN latency without an index, and HNSW
//! build/query latency with recall@10 against the exact result.

use std::time::Instant;

use gvec::{Database, Distance, VectorValue};
use serde_json::json;

fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    *state
}

fn next_f32(state: &mut u64) -> f32 {
    ((lcg(state) >> 40) as f32) / ((1u32 << 24) as f32)
}

fn clustered_vectors(n: usize, dim: usize) -> Vec<Vec<f32>> {
    let mut state = 0x9E37_79B9_7F4A_7C15;
    let clusters = 16;
    let centers: Vec<Vec<f32>> = (0..clusters)
        .map(|_| (0..dim).map(|_| next_f32(&mut state)).collect())
        .collect();
    (0..n)
        .map(|i| {
            let center = &centers[i % clusters];
            center
                .iter()
                .map(|x| x + (next_f32(&mut state) - 0.5) * 0.05)
                .collect()
        })
        .collect()
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((sorted.len() - 1) as f64 * p).round() as usize;
    sorted[idx]
}

fn recall_at_k(exact: &[(i64, f32)], got: &[(i64, f32)]) -> f64 {
    if exact.is_empty() {
        return 1.0;
    }
    let hits = got
        .iter()
        .filter(|(id, _)| exact.iter().any(|(exact_id, _)| exact_id == id))
        .count();
    hits as f64 / exact.len() as f64
}

fn main() {
    let n = 20_000;
    let dim = 64;
    let k = 10;
    let query_count = 20;
    let vectors = clustered_vectors(n, dim);
    let mut query_state = 0xD1B5_4A32_D192_ED03;
    let queries: Vec<VectorValue> = (0..query_count)
        .map(|_| {
            VectorValue::F32(
                vectors[(lcg(&mut query_state) as usize) % n]
                    .iter()
                    .map(|x| x + (next_f32(&mut query_state) - 0.5) * 0.02)
                    .collect(),
            )
        })
        .collect();

    let db = Database::open_in_memory().unwrap();
    let g = db.graph("bench").unwrap();

    let start = Instant::now();
    for vector in &vectors {
        let values: Vec<f64> = vector.iter().map(|x| *x as f64).collect();
        g.storage
            .create_node(&["Doc".to_string()], &json!({ "embedding": values }))
            .unwrap();
    }
    let insert_elapsed = start.elapsed();
    let per_node_us = insert_elapsed.as_secs_f64() * 1e6 / n as f64;

    let rows_start = Instant::now();
    let rows = g.vector_rows(Some("Doc"), Some("embedding")).unwrap();
    let rows_elapsed = rows_start.elapsed();
    assert_eq!(rows.len(), n);

    let mut exact_latencies = Vec::with_capacity(query_count);
    let mut exact_results = Vec::with_capacity(query_count);
    for query in &queries {
        let start = Instant::now();
        let hits = g
            .exact_vector_search("Doc", "embedding", query, k, Distance::L2)
            .unwrap();
        exact_latencies.push(start.elapsed().as_secs_f64() * 1e3);
        exact_results.push(hits);
    }

    let build_start = Instant::now();
    g.create_vector_index("Doc", "embedding", "hnsw", Some("l2"), None)
        .unwrap();
    let build_elapsed = build_start.elapsed();

    let mut ann_latencies = Vec::with_capacity(query_count);
    let mut recall_sum = 0.0;
    for (i, query) in queries.iter().enumerate() {
        let start = Instant::now();
        let hits = g.vector_search("Doc", "embedding", query, k).unwrap();
        ann_latencies.push(start.elapsed().as_secs_f64() * 1e3);
        recall_sum += recall_at_k(&exact_results[i], &hits);
    }

    exact_latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());
    ann_latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let recall = recall_sum / query_count as f64;

    println!("persistent vector rows");
    println!("  nodes / rows       : {n} / {}", rows.len());
    println!("  dimensions         : {dim}");
    println!(
        "  insert             : {:.3} s ({per_node_us:.2} us/node)",
        insert_elapsed.as_secs_f64()
    );
    println!(
        "  row load           : {:.3} ms ({} rows)",
        rows_elapsed.as_secs_f64() * 1e3,
        rows.len()
    );
    println!(
        "  exact KNN p50/p95  : {:.3} / {:.3} ms",
        percentile(&exact_latencies, 0.50),
        percentile(&exact_latencies, 0.95)
    );
    println!(
        "  HNSW build         : {:.3} s",
        build_elapsed.as_secs_f64()
    );
    println!(
        "  HNSW KNN p50/p95   : {:.3} / {:.3} ms",
        percentile(&ann_latencies, 0.50),
        percentile(&ann_latencies, 0.95)
    );
    println!("  recall@{k}          : {recall:.3}");
}
