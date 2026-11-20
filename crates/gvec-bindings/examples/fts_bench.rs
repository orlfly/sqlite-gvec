//! Full-text search benchmark for gvec.
//!
//! Run with:
//! ```text
//! cargo run --release -p gvec --example fts_bench
//! ```
//!
//! The benchmark uses a deterministic Zipf-distributed corpus, verifies FTS5
//! result sets against a naive token scan, and reports index build time,
//! storage amplification, and query latency for direct and Cypher paths.

use std::collections::HashSet;
use std::fmt::Write as _;
use std::time::Instant;

use gvec::Database;
use gvec_core::fts::text_search_all;
use serde_json::json;

const VOCAB: usize = 20_000;
const DOC_LEN: usize = 80;
const SPECIALS: [&str; 3] = ["graphdb", "vectorengine", "fulltextsearch"];

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

    /// Uniform in [0, 1), 53 random bits.
    fn next_f64(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64) / ((1u64 << 53) as f64)
    }
}

fn zipf_cdf(n: usize, exponent: f64) -> Vec<f64> {
    let mut weights = Vec::with_capacity(n);
    let mut total = 0.0;
    for rank in 0..n {
        let w = 1.0 / ((rank + 1) as f64).powf(exponent);
        total += w;
        weights.push(total);
    }
    for w in &mut weights {
        *w /= total;
    }
    weights
}

fn sample_word(rng: &mut Rng, cdf: &[f64]) -> u32 {
    let u = rng.next_f64();
    cdf.partition_point(|&x| x < u).min(cdf.len() - 1) as u32
}

fn push_word(out: &mut String, token: u32) {
    if !out.is_empty() {
        out.push(' ');
    }
    let t = token as usize;
    if t < VOCAB {
        write!(out, "w{t:05}").unwrap();
    } else {
        out.push_str(SPECIALS[t - VOCAB]);
    }
}

fn body_from(tokens: &[u32]) -> String {
    let mut out = String::with_capacity(tokens.len() * 9);
    for &token in tokens {
        push_word(&mut out, token);
    }
    out
}

fn corpus(n: usize, seed: u64) -> Vec<(u32, String)> {
    let cdf = zipf_cdf(VOCAB, 1.0);
    let mut rng = Rng::new(seed);
    let mut docs = Vec::with_capacity(n);
    for d in 0..n {
        let mut tokens: Vec<u32> = (0..DOC_LEN).map(|_| sample_word(&mut rng, &cdf)).collect();
        // Embed a fixed phrase in roughly 1/13 of the corpus so phrase queries
        // have a known, non-trivial result set.
        if d % 13 == 0 {
            tokens.push(VOCAB as u32);
            tokens.push((VOCAB + 1) as u32);
            tokens.push((VOCAB + 2) as u32);
        }
        docs.push((d as u32 + 1, body_from(&tokens)));
    }
    docs
}

fn tokens_for(docs: &[(u32, String)], n: usize) -> Vec<Vec<u32>> {
    // Re-tokenize the generated text with the same vocabulary mapping.
    let mut reverse = std::collections::HashMap::with_capacity(VOCAB + SPECIALS.len());
    for i in 0..VOCAB {
        reverse.insert(format!("w{i:05}"), i as u32);
    }
    for (i, s) in SPECIALS.iter().enumerate() {
        reverse.insert((*s).to_string(), (VOCAB + i) as u32);
    }
    docs.iter()
        .take(n)
        .map(|(_, body)| {
            body.split_whitespace()
                .map(|w| *reverse.get(w).expect("generated token must exist"))
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

fn cypher_string(s: &str) -> String {
    format!("\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\""))
}

fn direct_latencies(g: &gvec::Graph, queries: &[String], k: usize) -> Vec<f64> {
    let mut latencies = Vec::with_capacity(queries.len());
    for q in queries {
        let start = Instant::now();
        let hits = g.text_search("Doc", "body", q, k).unwrap();
        latencies.push(start.elapsed().as_secs_f64() * 1000.0);
        // BM25 scores are negated to make higher = more relevant.
        assert!(
            hits.windows(2).all(|w| w[0].1 >= w[1].1),
            "scores must be descending for {q}"
        );
    }
    latencies
}

fn cypher_latencies(g: &gvec::Graph, queries: &[String], limit: usize) -> Vec<f64> {
    let mut latencies = Vec::with_capacity(queries.len());
    for q in queries {
        let query = format!(
            "MATCH (n:Doc) WHERE text_match(n.body, {}) RETURN n.body AS body LIMIT {limit}",
            cypher_string(q)
        );
        let start = Instant::now();
        let rs = g.run(&query).unwrap();
        latencies.push(start.elapsed().as_secs_f64() * 1000.0);
        assert!(
            rs.rows.len() <= limit,
            "LIMIT must be respected, got {} rows",
            rs.rows.len()
        );
    }
    latencies
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

fn verify(g: &gvec::Graph, name: &str, query: &str, expected: &HashSet<i64>) {
    let got: HashSet<i64> = text_search_all(g.storage.exec_ref(), "test", "Doc", "body", query)
        .unwrap()
        .into_iter()
        .map(|(id, _)| id)
        .collect();
    assert_eq!(
        got.len(),
        expected.len(),
        "{name}: count mismatch (query {query:?})"
    );
    assert!(
        &got == expected,
        "{name}: rowid set mismatch (query {query:?})"
    );
}

fn token_starts_with(token: u32, prefix: &str) -> bool {
    let t = token as usize;
    if t < VOCAB {
        format!("w{t:05}").starts_with(prefix)
    } else {
        SPECIALS[t - VOCAB].starts_with(prefix)
    }
}

struct CaseResult {
    name: &'static str,
    matches: usize,
    direct_p50: f64,
    direct_p95: f64,
    cypher_p50: f64,
    cypher_p95: f64,
}

fn run_case(
    g: &gvec::Graph,
    name: &'static str,
    queries: Vec<String>,
    expected: &HashSet<i64>,
    iterations: usize,
    limit: usize,
) -> CaseResult {
    let direct_queries: Vec<String> = (0..iterations)
        .map(|i| queries[i % queries.len()].clone())
        .collect();
    let mut direct = direct_latencies(g, &direct_queries, limit);
    direct.sort_by(|a, b| a.total_cmp(b));

    let cypher_queries: Vec<String> = (0..iterations)
        .map(|i| queries[i % queries.len()].clone())
        .collect();
    let mut cypher = cypher_latencies(g, &cypher_queries, limit);
    cypher.sort_by(|a, b| a.total_cmp(b));

    CaseResult {
        name,
        matches: expected.len(),
        direct_p50: percentile(&direct, 0.50),
        direct_p95: percentile(&direct, 0.95),
        cypher_p50: percentile(&cypher, 0.50),
        cypher_p95: percentile(&cypher, 0.95),
    }
}

fn print_case_table(n: usize, cases: &[CaseResult]) {
    println!("\nFTS queries (n={n} docs)");
    println!(
        "{:<18} {:>9} {:>12} {:>12} {:>12} {:>12}",
        "case", "matches", "direct_p50", "direct_p95", "cypher_p50", "cypher_p95"
    );
    for c in cases {
        println!(
            "{:<18} {:>9} {:>12.4} {:>12.4} {:>12.4} {:>12.4}",
            c.name, c.matches, c.direct_p50, c.direct_p95, c.cypher_p50, c.cypher_p95
        );
    }
    println!("latencies in milliseconds, direct = Graph::text_search, cypher = text_match + LIMIT");
}

fn run_dataset(n: usize, iterations: usize) {
    println!("\n=== FTS dataset: n={n}, doc_len={DOC_LEN}, vocab={VOCAB} ===");
    let generated = corpus(n, 0x5EED_F00D_1234_5678);
    let docs = tokens_for(&generated, n);
    let db = Database::open_in_memory().unwrap();
    let g = db.graph("test").unwrap();

    let insert_start = Instant::now();
    let mut ids = Vec::with_capacity(n);
    for (id, (_, body)) in generated.iter().enumerate() {
        let node = g
            .storage
            .create_node(&["Doc".into()], &json!({"body": body}))
            .unwrap();
        assert_eq!(node.0, id as i64 + 1);
        ids.push(node.0);
    }
    let insert_ms = insert_start.elapsed().as_secs_f64() * 1000.0;

    let (pages_before, page_size) = page_stats(&g);
    let build_start = Instant::now();
    g.create_text_index("Doc", "body", None).unwrap();
    let build_ms = build_start.elapsed().as_secs_f64() * 1000.0;
    let (pages_after, _) = page_stats(&g);
    let fts_bytes = (pages_after - pages_before) as f64 * page_size as f64;

    println!(
        "insert_nodes: {insert_ms:.1} ms ({:.1} us/doc)",
        insert_ms * 1000.0 / n as f64
    );
    println!(
        "build_index: {build_ms:.1} ms ({:.1} us/doc), fts_bytes: {:.1} MB ({:.1} bytes/doc)",
        build_ms * 1000.0 / n as f64,
        fts_bytes / (1024.0 * 1024.0),
        fts_bytes / n as f64
    );

    // ---- correctness against a naive token scan ----
    let common = 0u32;
    let medium = 100u32;
    let rare = 15_000u32;
    let expected_single = |needle: u32| -> HashSet<i64> {
        docs.iter()
            .enumerate()
            .filter(|(_, tokens)| tokens.contains(&needle))
            .map(|(i, _)| ids[i])
            .collect()
    };
    let expected_pair: HashSet<i64> = docs
        .iter()
        .enumerate()
        .filter(|(_, tokens)| tokens.contains(&10) && tokens.contains(&100))
        .map(|(i, _)| ids[i])
        .collect();
    let expected_or: HashSet<i64> = docs
        .iter()
        .enumerate()
        .filter(|(_, tokens)| tokens.contains(&5) || tokens.contains(&15_000))
        .map(|(i, _)| ids[i])
        .collect();
    let expected_phrase: HashSet<i64> = docs
        .iter()
        .enumerate()
        .filter(|(_, tokens)| {
            tokens
                .windows(2)
                .any(|w| w == [VOCAB as u32, (VOCAB + 1) as u32])
        })
        .map(|(i, _)| ids[i])
        .collect();
    let expected_prefix: HashSet<i64> = docs
        .iter()
        .enumerate()
        .filter(|(_, tokens)| tokens.iter().any(|&t| token_starts_with(t, "w1234")))
        .map(|(i, _)| ids[i])
        .collect();

    verify(&g, "single common", "w00000", &expected_single(common));
    verify(&g, "single medium", "w00100", &expected_single(medium));
    verify(&g, "single rare", "w15000", &expected_single(rare));
    verify(&g, "two-term AND", "w00010 AND w00100", &expected_pair);
    verify(&g, "boolean OR", "w00005 OR w15000", &expected_or);
    verify(&g, "phrase", "\"graphdb vectorengine\"", &expected_phrase);
    verify(&g, "prefix", "w1234*", &expected_prefix);
    verify(&g, "no hit", "zzzznotfound", &HashSet::new());
    println!("verification: ok (8/8 cases match naive token scan)");

    // ---- query latency ----
    let single_queries = |rank: u32, count: usize| -> Vec<String> {
        (0..count)
            .map(|i| format!("w{:05}", rank + i as u32))
            .collect()
    };
    let pair_queries = (0..iterations)
        .map(|i| format!("w{:05} AND w{:05}", 10 + i % 50, 100 + i % 50))
        .collect();
    let or_queries = (0..iterations)
        .map(|i| format!("w{:05} OR w{:05}", 5 + i % 50, 15_000 + i % 50))
        .collect();
    let phrase_queries = vec!["\"graphdb vectorengine\"".to_string()];
    let prefix_queries = vec!["w1234*".to_string()];
    let miss_queries = vec!["zzzznotfound".to_string()];

    let cases = vec![
        run_case(
            &g,
            "single common",
            single_queries(0, iterations),
            &expected_single(common),
            iterations.min(10),
            10,
        ),
        run_case(
            &g,
            "single medium",
            single_queries(100, iterations),
            &expected_single(medium),
            iterations,
            10,
        ),
        run_case(
            &g,
            "single rare",
            single_queries(15_000, iterations),
            &expected_single(rare),
            iterations,
            10,
        ),
        run_case(
            &g,
            "two-term AND",
            pair_queries,
            &expected_pair,
            iterations,
            10,
        ),
        run_case(&g, "boolean OR", or_queries, &expected_or, iterations, 10),
        run_case(
            &g,
            "phrase",
            phrase_queries,
            &expected_phrase,
            iterations,
            10,
        ),
        run_case(
            &g,
            "prefix",
            prefix_queries,
            &expected_prefix,
            iterations,
            10,
        ),
        run_case(&g, "no hit", miss_queries, &HashSet::new(), iterations, 10),
    ];
    print_case_table(n, &cases);

    // ---- incremental writes with the index active ----
    let incremental = 1_000.min(n);
    let mut rng = Rng::new(0xABCD_EF01_2345_6789);
    let cdf = zipf_cdf(VOCAB, 1.0);
    let write_start = Instant::now();
    for _ in 0..incremental {
        let tokens: Vec<u32> = (0..DOC_LEN).map(|_| sample_word(&mut rng, &cdf)).collect();
        g.storage
            .create_node(&["Doc".into()], &json!({"body": body_from(&tokens)}))
            .unwrap();
    }
    let write_ms = write_start.elapsed().as_secs_f64() * 1000.0;
    println!(
        "incremental insert with FTS triggers: {incremental} docs in {write_ms:.1} ms ({:.1} us/doc)",
        write_ms * 1000.0 / incremental as f64
    );
}

fn main() {
    println!("gvec full-text search benchmark");
    println!("deterministic Zipf corpus, in-memory SQLite FTS5, single-threaded");
    for n in [1_000, 10_000, 50_000] {
        run_dataset(n, 50);
    }
    println!("\nnotes:");
    println!("- direct = Graph::text_search(k=10); cypher = MATCH ... text_match ... LIMIT 10.");
    println!("- build_index time covers CREATE VIRTUAL TABLE + backfill + trigger creation.");
    println!("- fts_bytes is the page-count delta across index creation, an upper bound.");
    println!("- every measured query set is verified against a naive token scan.");
}
