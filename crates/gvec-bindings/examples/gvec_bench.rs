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
    for _ in 0..n { f(); }
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

    println!("\nmethodology: in-memory SQLite, no network I/O, single-threaded");
}