//! Graph traversal benchmark for gvec.
//!
//! Run with:
//! ```text
//! cargo run --release -p gvec --example graph_bench
//! ```
//!
//! Builds a deterministic directed social graph, verifies fixed and
//! variable-length traversals against a naive adjacency model, and reports
//! insert and query latency. Edge `source` and `target` are indexed by
//! `ensure_schema`, so each hop is an index lookup rather than a full scan.

use std::collections::HashSet;
use std::time::Instant;

use gvec::Database;
use gvec_core::ids::NodeId;
use serde_json::json;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Kind {
    Follow,
    Like,
}

impl Kind {
    fn as_str(self) -> &'static str {
        match self {
            Kind::Follow => "FOLLOWS",
            Kind::Like => "LIKES",
        }
    }
}

#[derive(Debug, Clone)]
struct MEdge {
    id: i64,
    from: usize,
    to: usize,
    kind: Kind,
    weight: f64,
}

struct Model {
    out: Vec<Vec<MEdge>>,
    inc: Vec<Vec<MEdge>>,
    edge_count: usize,
}

fn build_model(n: usize) -> Model {
    let mut seen: HashSet<(usize, usize, Kind)> = HashSet::new();
    let mut edges: Vec<MEdge> = Vec::new();
    let mut next_id = 1i64;
    let mut push = |from: usize, to: usize, kind: Kind, weight: f64| {
        if from == to || !seen.insert((from, to, kind)) {
            return;
        }
        edges.push(MEdge {
            id: next_id,
            from,
            to,
            kind,
            weight,
        });
        next_id += 1;
    };

    for i in 0..n {
        for k in 0..4 {
            let target = (i * 7 + k * 104_729 + 1) % n;
            let weight = ((i * 17 + k * 31 + 7) % 100) as f64 / 100.0;
            push(i, target, Kind::Follow, weight);
        }
        let target = (i * 13 + 5) % n;
        push(i, target, Kind::Like, 0.5);
    }

    let mut out = vec![Vec::new(); n];
    let mut inc = vec![Vec::new(); n];
    for edge in &edges {
        out[edge.from].push(edge.clone());
        inc[edge.to].push(edge.clone());
    }
    Model {
        out,
        inc,
        edge_count: edges.len(),
    }
}

fn fixed_paths(model: &Model, anchor: usize, hops: usize, kinds: &[Kind]) -> Vec<usize> {
    let mut frontier = vec![anchor];
    for _ in 0..hops {
        let mut next = Vec::new();
        for node in frontier {
            for edge in &model.out[node] {
                if kinds.contains(&edge.kind) {
                    next.push(edge.to);
                }
            }
        }
        frontier = next;
    }
    frontier
}

fn undirected_paths(model: &Model, anchor: usize, kinds: &[Kind]) -> Vec<usize> {
    let mut out = Vec::new();
    for edge in &model.out[anchor] {
        if kinds.contains(&edge.kind) {
            out.push(edge.to);
        }
    }
    for edge in &model.inc[anchor] {
        if kinds.contains(&edge.kind) {
            out.push(edge.from);
        }
    }
    out
}

fn weighted_paths(model: &Model, anchor: usize, min_weight: f64) -> Vec<usize> {
    model.out[anchor]
        .iter()
        .filter(|edge| edge.kind == Kind::Follow && edge.weight > min_weight)
        .map(|edge| edge.to)
        .collect()
}

fn var_paths(model: &Model, anchor: usize, lo: usize, hi: usize, kinds: &[Kind]) -> Vec<usize> {
    let mut out = Vec::new();
    let mut stack = vec![(anchor, Vec::<i64>::new())];
    while let Some((node, visited)) = stack.pop() {
        let depth = visited.len();
        if depth >= lo {
            out.push(node);
        }
        if depth >= hi {
            continue;
        }
        for edge in &model.out[node] {
            if kinds.contains(&edge.kind) && !visited.contains(&edge.id) {
                let mut next_visited = visited.clone();
                next_visited.push(edge.id);
                stack.push((edge.to, next_visited));
            }
        }
    }
    out
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

struct Row {
    name: &'static str,
    avg_rows: f64,
    p50: f64,
    p95: f64,
}

fn run_case<F>(
    g: &gvec::Graph,
    name: &'static str,
    anchors: &[usize],
    expected: &[Vec<usize>],
    mut make_query: F,
) -> Row
where
    F: FnMut(usize) -> String,
{
    let mut latencies = Vec::with_capacity(anchors.len());
    let mut total_rows = 0usize;
    for (i, anchor) in anchors.iter().enumerate() {
        let query = make_query(*anchor);
        let start = Instant::now();
        let rs = g.run(&query).unwrap();
        latencies.push(start.elapsed().as_secs_f64() * 1000.0);
        let mut got: Vec<i64> = rs
            .rows
            .iter()
            .filter_map(|row| row.first().and_then(|v| v.as_i64()))
            .collect();
        total_rows += got.len();
        got.sort_unstable();
        let mut want: Vec<i64> = expected[i].iter().map(|v| *v as i64).collect();
        want.sort_unstable();
        assert_eq!(
            got, want,
            "result mismatch for {name} anchor {anchor}: query {query:?}"
        );
    }
    latencies.sort_by(|a, b| a.total_cmp(b));
    Row {
        name,
        avg_rows: total_rows as f64 / anchors.len() as f64,
        p50: percentile(&latencies, 0.50),
        p95: percentile(&latencies, 0.95),
    }
}

fn run_optional(g: &gvec::Graph, anchors: &[usize]) -> Row {
    let mut latencies = Vec::with_capacity(anchors.len());
    for anchor in anchors {
        let query = format!(
            "MATCH (a:Person {{id: {anchor}}}) \
             OPTIONAL MATCH (a)-[:MISSING]->(b) \
             RETURN a.id AS a_id, b.id AS b_id"
        );
        let start = Instant::now();
        let rs = g.run(&query).unwrap();
        latencies.push(start.elapsed().as_secs_f64() * 1000.0);
        assert_eq!(rs.rows.len(), 1, "optional miss must keep the base row");
        assert_eq!(rs.rows[0][0], json!(*anchor as i64));
        assert_eq!(rs.rows[0][1], json!(null));
    }
    latencies.sort_by(|a, b| a.total_cmp(b));
    Row {
        name: "optional miss",
        avg_rows: 1.0,
        p50: percentile(&latencies, 0.50),
        p95: percentile(&latencies, 0.95),
    }
}

fn print_table(n: usize, m: usize, rows: &[Row]) {
    println!("\nGraph queries (n={n}, m={m})");
    println!(
        "{:<20} {:>10} {:>10} {:>10}",
        "case", "avg_rows", "p50_ms", "p95_ms"
    );
    for row in rows {
        println!(
            "{:<20} {:>10.1} {:>10.4} {:>10.4}",
            row.name, row.avg_rows, row.p50, row.p95
        );
    }
}

fn run_dataset(n: usize) {
    println!("\n=== Graph dataset: n={n} ===");
    let model = build_model(n);
    let m = model.edge_count;
    let db = Database::open_in_memory().unwrap();
    let g = db.graph("test").unwrap();

    let cities = ["Beijing", "Shanghai", "Shenzhen", "Hangzhou", "Chengdu"];
    let node_start = Instant::now();
    for i in 0..n {
        g.storage
            .create_node(
                &["Person".into()],
                &json!({
                    "id": i as i64,
                    "age": 18 + (i * 7) % 60,
                    "city": cities[i % cities.len()],
                }),
            )
            .unwrap();
    }
    let node_ms = node_start.elapsed().as_secs_f64() * 1000.0;

    let edge_start = Instant::now();
    for edge in model.out.iter().flatten() {
        g.storage
            .create_edge(
                NodeId(edge.from as i64 + 1),
                NodeId(edge.to as i64 + 1),
                Some(edge.kind.as_str()),
                &json!({"weight": edge.weight}),
            )
            .unwrap();
    }
    let edge_ms = edge_start.elapsed().as_secs_f64() * 1000.0;

    // Pick anchors whose three-hop fanout stays small enough to verify the
    // complete result set without LIMIT.
    let mut anchors = Vec::new();
    let step = (n / 40).max(1);
    for i in (0..n).step_by(step) {
        if fixed_paths(&model, i, 3, &[Kind::Follow]).len() <= 5_000 {
            anchors.push(i);
        }
        if anchors.len() >= 20 {
            break;
        }
    }
    assert!(!anchors.is_empty(), "no benchmark anchors selected");
    let exp_anchor: Vec<Vec<usize>> = anchors.iter().map(|a| vec![*a]).collect();
    let anchor_scan = run_case(&g, "anchor scan", &anchors, &exp_anchor, |a| {
        format!("MATCH (a:Person {{id: {a}}}) RETURN a.id AS id")
    });

    let tag_start = Instant::now();
    g.create_tag_index("Person", "id").unwrap();
    let tag_ms = tag_start.elapsed().as_secs_f64() * 1000.0;
    let (pages, page_size) = page_stats(&g);
    let bytes = pages as f64 * page_size as f64;

    println!(
        "insert_nodes: {node_ms:.1} ms ({:.1} us/node)",
        node_ms * 1000.0 / n as f64
    );
    println!(
        "insert_edges: {edge_ms:.1} ms ({:.2} us/edge, {:.0} edges/s)",
        edge_ms * 1000.0 / m as f64,
        m as f64 / (edge_ms / 1000.0)
    );
    println!(
        "storage: {:.2} MB ({:.0} bytes/node, {:.0} bytes/edge)",
        bytes / (1024.0 * 1024.0),
        bytes / n as f64,
        bytes / m as f64
    );
    println!("tag_index_build: {tag_ms:.1} ms (Person.id)");

    let follow = [Kind::Follow];
    let both = [Kind::Follow, Kind::Like];
    let exp_1out: Vec<Vec<usize>> = anchors
        .iter()
        .map(|a| fixed_paths(&model, *a, 1, &follow))
        .collect();
    let exp_1in: Vec<Vec<usize>> = anchors
        .iter()
        .map(|a| {
            model.inc[*a]
                .iter()
                .filter(|e| e.kind == Kind::Follow)
                .map(|e| e.from)
                .collect()
        })
        .collect();
    let exp_2: Vec<Vec<usize>> = anchors
        .iter()
        .map(|a| fixed_paths(&model, *a, 2, &follow))
        .collect();
    let exp_3: Vec<Vec<usize>> = anchors
        .iter()
        .map(|a| fixed_paths(&model, *a, 3, &follow))
        .collect();
    let exp_var2: Vec<Vec<usize>> = anchors
        .iter()
        .map(|a| var_paths(&model, *a, 1, 2, &follow))
        .collect();
    let exp_var3: Vec<Vec<usize>> = anchors
        .iter()
        .map(|a| var_paths(&model, *a, 1, 3, &follow))
        .collect();
    let exp_undirected: Vec<Vec<usize>> = anchors
        .iter()
        .map(|a| undirected_paths(&model, *a, &follow))
        .collect();
    let exp_multi: Vec<Vec<usize>> = anchors
        .iter()
        .map(|a| fixed_paths(&model, *a, 1, &both))
        .collect();
    let exp_weighted: Vec<Vec<usize>> = anchors
        .iter()
        .map(|a| weighted_paths(&model, *a, 0.8))
        .collect();

    let rows = vec![
        anchor_scan,
        run_case(&g, "anchor tag index", &anchors, &exp_anchor, |a| {
            format!("MATCH (a:Person {{id: {a}}}) RETURN a.id AS id")
        }),
        run_case(&g, "1-hop out", &anchors, &exp_1out, |a| {
            format!("MATCH (a:Person {{id: {a}}})-[:FOLLOWS]->(b) RETURN b.id AS id")
        }),
        run_case(&g, "1-hop in", &anchors, &exp_1in, |a| {
            format!("MATCH (a:Person {{id: {a}}})<-[:FOLLOWS]-(b) RETURN b.id AS id")
        }),
        run_case(&g, "2-hop fixed", &anchors, &exp_2, |a| {
            format!(
                "MATCH (a:Person {{id: {a}}})-[:FOLLOWS]->(b)-[:FOLLOWS]->(c) RETURN c.id AS id"
            )
        }),
        run_case(&g, "3-hop fixed", &anchors, &exp_3, |a| {
            format!(
                "MATCH (a:Person {{id: {a}}})-[:FOLLOWS]->(b)-[:FOLLOWS]->(c)-[:FOLLOWS]->(d) RETURN d.id AS id"
            )
        }),
        run_case(&g, "var 1..2", &anchors, &exp_var2, |a| {
            format!("MATCH (a:Person {{id: {a}}})-[:FOLLOWS*1..2]->(b) RETURN b.id AS id")
        }),
        run_case(&g, "var 1..3", &anchors, &exp_var3, |a| {
            format!("MATCH (a:Person {{id: {a}}})-[:FOLLOWS*1..3]->(b) RETURN b.id AS id")
        }),
        run_case(&g, "undirected 1-hop", &anchors, &exp_undirected, |a| {
            format!("MATCH (a:Person {{id: {a}}})-[:FOLLOWS]-(b) RETURN b.id AS id")
        }),
        run_case(&g, "multi-type 1-hop", &anchors, &exp_multi, |a| {
            format!("MATCH (a:Person {{id: {a}}})-[:FOLLOWS|LIKES]->(b) RETURN b.id AS id")
        }),
        run_case(&g, "edge property", &anchors, &exp_weighted, |a| {
            format!(
                "MATCH (a:Person {{id: {a}}})-[r:FOLLOWS]->(b) \
                     WHERE r.weight > 0.8 RETURN b.id AS id"
            )
        }),
        run_optional(&g, &anchors),
    ];

    print_table(n, m, &rows);
    println!(
        "verification: ok ({} anchors, complete result sets, no LIMIT)",
        anchors.len()
    );
}

fn main() {
    println!("gvec graph traversal benchmark");
    println!("deterministic directed social graph, in-memory SQLite, single-threaded");
    for n in [10_000, 50_000] {
        run_dataset(n);
    }
    println!("\nnotes:");
    println!("- edge endpoints are indexed on (source) and (target).");
    println!("- fixed patterns enumerate paths; duplicate endpoints are separate rows.");
    println!("- variable-length patterns disallow reusing the same relationship in one path.");
    println!("- anchor scan is a label scan plus inline property filter.");
    println!("- Person.id tag index turns inline `{{id: ...}}` anchors into index lookups.");
}
