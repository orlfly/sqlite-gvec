//! Exact tag index benchmark for gvec.
//!
//! Run with:
//! ```text
//! cargo run --release -p gvec --example tag_bench
//! ```
//!
//! Builds a deterministic corpus with three tags per document, verifies tag
//! lookups against a naive scan, and compares indexed tag queries with the
//! exact fallback used when no tag index exists.

use std::collections::HashSet;
use std::time::Instant;

use gvec::Database;
use serde_json::json;

const TAG_VOCAB: usize = 1_000;
const TAGS_PER_DOC: usize = 3;

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

    fn next_f64(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64) / ((1u64 << 53) as f64)
    }
}

fn zipf_cdf(n: usize, exponent: f64) -> Vec<f64> {
    let mut cdf = Vec::with_capacity(n);
    let mut total = 0.0;
    for rank in 0..n {
        total += 1.0 / ((rank + 1) as f64).powf(exponent);
        cdf.push(total);
    }
    for x in &mut cdf {
        *x /= total;
    }
    cdf
}

fn sample_tag(rng: &mut Rng, cdf: &[f64]) -> usize {
    let u = rng.next_f64();
    cdf.partition_point(|&x| x < u).min(cdf.len() - 1)
}

fn corpus(n: usize) -> Vec<(String, Vec<String>)> {
    let cdf = zipf_cdf(TAG_VOCAB, 1.1);
    let mut rng = Rng::new(0x7A6B_5C4D_3E2F_1098);
    (0..n)
        .map(|i| {
            let mut tags: Vec<String> = Vec::with_capacity(TAGS_PER_DOC);
            for _ in 0..TAGS_PER_DOC {
                let tag = format!("tag{:03}", sample_tag(&mut rng, &cdf));
                if !tags.contains(&tag) {
                    tags.push(tag);
                }
            }
            tags.sort();
            let body = if i % 4 == 0 {
                "alpha beta".to_string()
            } else {
                "beta gamma".to_string()
            };
            (body, tags)
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

fn page_stats(g: &gvec::Graph) -> (i64, i64) {
    let count = g
        .storage
        .exec_ref()
        .query_one_json("PRAGMA page_count", &[])
        .unwrap()
        .and_then(|r| r.first().and_then(|v| v.as_i64()))
        .unwrap_or(0);
    let size = g
        .storage
        .exec_ref()
        .query_one_json("PRAGMA page_size", &[])
        .unwrap()
        .and_then(|r| r.first().and_then(|v| v.as_i64()))
        .unwrap_or(0);
    (count, size)
}

fn measure<F: FnMut()>(iterations: usize, mut f: F) -> (f64, f64) {
    let mut latencies = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let start = Instant::now();
        f();
        latencies.push(start.elapsed().as_secs_f64() * 1000.0);
    }
    latencies.sort_by(|a, b| a.total_cmp(b));
    (percentile(&latencies, 0.50), percentile(&latencies, 0.95))
}

struct Row {
    name: &'static str,
    matches: usize,
    direct_p50: f64,
    direct_p95: f64,
    cypher_p50: f64,
    cypher_p95: f64,
}

fn print_table(n: usize, rows: &[Row]) {
    println!("\nTag queries (n={n} docs, {TAGS_PER_DOC} tags/doc, vocab={TAG_VOCAB})");
    println!(
        "{:<20} {:>9} {:>12} {:>12} {:>12} {:>12}",
        "case", "matches", "direct_p50", "direct_p95", "cypher_p50", "cypher_p95"
    );
    for row in rows {
        println!(
            "{:<20} {:>9} {:>12} {:>12} {:>12} {:>12}",
            row.name,
            row.matches,
            fmt_ms(row.direct_p50),
            fmt_ms(row.direct_p95),
            fmt_ms(row.cypher_p50),
            fmt_ms(row.cypher_p95)
        );
    }
    println!("latencies in milliseconds, direct = Graph::tag_search(k=10), cypher = tag_match + LIMIT 10");
}

fn fmt_ms(value: f64) -> String {
    if value.is_nan() {
        "n/a".to_string()
    } else {
        format!("{value:.4}")
    }
}

fn run_case(
    g: &gvec::Graph,
    name: &'static str,
    tag: &str,
    expected: &HashSet<i64>,
    iterations: usize,
) -> Row {
    let (direct_p50, direct_p95) = measure(iterations, || {
        let hits = g.tag_search("Doc", "tags", tag, 10).unwrap();
        assert!(hits.len() <= 10);
    });
    let cypher =
        format!("MATCH (n:Doc) WHERE tag_match(n.tags, \"{tag}\") RETURN n.body AS body LIMIT 10");
    let (cypher_p50, cypher_p95) = measure(iterations, || {
        let rs = g.run(&cypher).unwrap();
        assert!(rs.rows.len() <= 10);
    });
    Row {
        name,
        matches: expected.len(),
        direct_p50,
        direct_p95,
        cypher_p50,
        cypher_p95,
    }
}

fn run_dataset(n: usize, iterations: usize) {
    println!("\n=== Tag dataset: n={n} ===");
    let docs = corpus(n);
    let db = Database::open_in_memory().unwrap();
    let g = db.graph("test").unwrap();

    let insert_start = Instant::now();
    let mut ids = Vec::with_capacity(n);
    for (body, tags) in &docs {
        let node = g
            .storage
            .create_node(&["Doc".into()], &json!({"body": body, "tags": tags}))
            .unwrap();
        ids.push(node.0);
    }
    let insert_ms = insert_start.elapsed().as_secs_f64() * 1000.0;

    let (pages_before, page_size) = page_stats(&g);
    let build_start = Instant::now();
    let meta = g.create_tag_index("Doc", "tags").unwrap();
    let build_ms = build_start.elapsed().as_secs_f64() * 1000.0;
    let (pages_after, _) = page_stats(&g);
    let index_bytes = (pages_after - pages_before) as f64 * page_size as f64;

    println!(
        "insert_nodes: {insert_ms:.1} ms ({:.1} us/doc)",
        insert_ms * 1000.0 / n as f64
    );
    println!(
        "build_tag_index: {build_ms:.1} ms ({:.1} us/doc), index_bytes: {:.2} MB ({:.1} bytes/doc), index_id={}",
        build_ms * 1000.0 / n as f64,
        index_bytes / (1024.0 * 1024.0),
        index_bytes / n as f64,
        meta.id
    );

    let tag_names = ["tag000", "tag010", "tag200"];
    let expected: Vec<HashSet<i64>> = tag_names
        .iter()
        .map(|tag| {
            docs.iter()
                .enumerate()
                .filter(|(_, (_, tags))| tags.iter().any(|t| t == tag))
                .map(|(i, _)| ids[i])
                .collect()
        })
        .collect();

    for (i, tag) in tag_names.iter().enumerate() {
        let got: HashSet<i64> = g
            .storage
            .tag_search("Doc", "tags", tag)
            .unwrap()
            .into_iter()
            .collect();
        assert_eq!(&got, &expected[i], "tag {tag} mismatch");
    }
    println!("verification: ok (3/3 tag sets match naive scan)");

    let mut rows = Vec::new();
    for (i, tag) in tag_names.iter().enumerate() {
        rows.push(run_case(
            &g,
            ["tag common", "tag medium", "tag rare"][i],
            tag,
            &expected[i],
            iterations,
        ));
    }

    // Hybrid text + tag retrieval.
    g.create_text_index("Doc", "body", None).unwrap();
    let hybrid = "MATCH (n:Doc) WHERE text_match(n.body, \"alpha\") \
                  AND tag_match(n.tags, \"tag000\") RETURN n.body AS body LIMIT 10";
    let hybrid_expected = expected[0]
        .iter()
        .filter(|id| {
            let idx = (**id - 1) as usize;
            idx.is_multiple_of(4)
        })
        .count();
    let (hp50, hp95) = measure(iterations.min(20), || {
        let rs = g.run(hybrid).unwrap();
        assert!(rs.rows.len() <= 10);
    });
    rows.push(Row {
        name: "text + tag",
        matches: hybrid_expected,
        direct_p50: f64::NAN,
        direct_p95: f64::NAN,
        cypher_p50: hp50,
        cypher_p95: hp95,
    });

    // Exact fallback after dropping the index. This scans the property array.
    g.drop_tag_index("Doc", "tags").unwrap();
    let fallback_iterations = iterations.min(5);
    for (i, tag) in tag_names.iter().take(2).enumerate() {
        rows.push(run_case(
            &g,
            ["fallback common", "fallback medium", "fallback rare"][i],
            tag,
            &expected[i],
            fallback_iterations,
        ));
    }
    print_table(n, &rows);
    println!("hybrid text+tag rows show cypher only; NaN direct means not applicable.");
}

fn main() {
    println!("gvec exact tag index benchmark");
    println!("deterministic Zipf tags, in-memory SQLite, single-threaded");
    for n in [10_000, 50_000] {
        run_dataset(n, 50);
    }
    println!("\nnotes:");
    println!("- tag vocabulary: tag000..tag999, three distinct tags per document.");
    println!("- indexed tag lookup uses ({{prefix}}_tags) with (index_id, tag, node_id).");
    println!("- fallback rows run after DROP TAG INDEX and scan each node's tag array.");
}
