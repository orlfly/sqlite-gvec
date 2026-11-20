//! Throughput benchmarks for gvec.
//!
//! Run with: `cargo bench --bench gvec_bench`
//!
//! We use a simple stdlib-based micro-benchmark here to avoid pulling in
//! Criterion for a quick smoke measurement. For production benchmarks,
//! swap in `criterion`.

use gvec::Database;
use std::time::Instant;

fn bench<F: FnMut()>(label: &str, n: u32, mut f: F) {
    // warmup
    f();
    let t = Instant::now();
    for _ in 0..n {
        f();
    }
    let d = t.elapsed();
    let per = d / n;
    println!("{label:30} n={n:6} total={d:?} per-op={per:?}");
}

fn main() {
    let db = Database::open_in_memory().unwrap();
    let g = db.graph("bench").unwrap();

    let n = 1000;
    bench("create_node", n, || {
        g.run(r#"CREATE (a:Person {name: "x"})"#).unwrap();
    });

    // MATCH+RETURN throughput
    let t = Instant::now();
    for _ in 0..n {
        g.run("MATCH (n:Person) RETURN n.name AS name").unwrap();
    }
    let d = t.elapsed() / n;
    println!("match_return              n={n:6} per-op={d:?}");

    // KNN throughput
    use gvec_core::vector::{Distance, VectorValue};
    let stored: Vec<(i64, VectorValue)> = (0..256)
        .map(|i| (i, VectorValue::F32(vec![i as f32, (i % 16) as f32])))
        .collect();
    let q = VectorValue::F32(vec![0.0, 0.0]);
    bench("knn_256", 100, || {
        let _ = g.knn(&q, &stored, 10, Distance::L2);
    });

    // ANN index build/search throughput (HNSW over 1,000 x 8-d vectors)
    use gvec_core::vector_index::{
        HnswIndex, HnswParams, VectorIndex, VectorIndexAlgorithm, VectorIndexConfig,
    };
    let ann_vectors: Vec<(i64, VectorValue)> = (0..1000)
        .map(|i| {
            let mut v = Vec::with_capacity(8);
            for d in 0..8 {
                v.push(((i * 31 + d * 17) % 101) as f32 / 100.0);
            }
            (i as i64, VectorValue::F32(v))
        })
        .collect();
    let config = VectorIndexConfig {
        algorithm: VectorIndexAlgorithm::Hnsw,
        metric: Distance::L2,
        params: serde_json::json!({"m": 16, "ef_construction": 200, "ef_search": 64}),
    };
    bench("hnsw_build_1000", 10, || {
        let _ = HnswIndex::build(Distance::L2, HnswParams::default(), ann_vectors.clone()).unwrap();
    });
    let index = VectorIndex::build(&config, ann_vectors).unwrap();
    let ann_query = VectorValue::F32(vec![0.5; 8]);
    bench("hnsw_search_1000", 100, || {
        let _ = index.search(&ann_query, 10);
    });

    // Storage-backed path: nodes with an 8-d embedding + persisted index.
    use serde_json::json;
    let t = Instant::now();
    for i in 0..1000 {
        let embedding: Vec<f32> = (0..8)
            .map(|d| ((i * 31 + d * 17) % 101) as f32 / 100.0)
            .collect();
        g.storage
            .create_node(&["Vec".to_string()], &json!({"embedding": embedding}))
            .unwrap();
    }
    let insert_time = t.elapsed();
    g.create_vector_index("Vec", "embedding", "hnsw", Some("l2"), None)
        .unwrap();
    bench("storage_vector_search_1000", 100, || {
        let _ = g.vector_search("Vec", "embedding", &ann_query, 10).unwrap();
    });
    println!(
        "{:30} n={:6} total={:?}",
        "storage_insert_1000", 1000, insert_time
    );

    println!("\nmethodology: in-memory SQLite, no network I/O, single-threaded");
}
