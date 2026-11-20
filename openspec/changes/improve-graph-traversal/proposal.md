## Why

The graph executor handled simple fixed patterns but had correctness and
scalability gaps that only appear on real graph workloads:

- A second `MATCH` reused the same variable by rescanning every node and
  overwriting the binding, so `MATCH ... MATCH ...` and `OPTIONAL MATCH` after a
  `MATCH` produced the wrong rows.
- Variable-length patterns did not filter destination labels or inline
  properties at every accepted endpoint, applied edge-type filters
  inconsistently after the first hop, bound a relationship variable to one edge
  instead of the whole path, and ignored pattern segments after the
  variable-length segment.
- Unbounded `*` could not terminate on cyclic graphs because a path could reuse
  the same relationship.
- Multi-label node patterns used `ANY` label instead of `ALL`.
- The parser accepted `*3`, `*..3`, `*2..`, and `*0..2`, but bare `*` failed.
- The parser had a `Pipe` token for `[:A|B]` but the lexer rejected `|`, so
  multi-type relationships never parsed.
- Every edge hop used a full edge-table scan. The edge endpoints had no indexes.
- Inline property anchors such as `MATCH (n:Person {id: 42})` decoded every node
  of the label in Rust, so anchor lookup dominated traversal latency.
- Tag indexes covered strings and string arrays but not scalar numbers, which
  are the natural anchor key for `id` properties.

## What Changes

- Make `MATCH` continue from an already-bound first variable instead of scanning
  and overwriting it. This fixes consecutive `MATCH` clauses and
  `MATCH ... OPTIONAL MATCH ...`, including the case where the optional
  relationship is absent.
- Rewrite variable-length traversal as a relationship-unique depth-first
  enumeration:
  - `*` is `1..inf`, `*3` is exactly three, `*..3` is `1..3`, `*2..` is
    `2..inf`, and `*0..2` is `0..2`.
  - Every accepted endpoint must satisfy the destination node's labels and
    inline properties.
  - Edge-type filters apply at every hop.
  - The relationship variable binds to a JSON array of edge objects.
  - Remaining fixed segments after a variable-length segment execute normally.
- Require all labels in a multi-label pattern.
- Add `{prefix}_edges_source` and `{prefix}_edges_target` indexes so
  `list_edges(source, target)` uses index lookups.
- Push inline top-level scalar property comparisons into SQLite with
  `json_extract` as a prefilter, while keeping the exact Rust-side match.
- Allow exact tag indexes to cover scalar numbers as well as strings and string
  arrays. The query planner uses a matching tag index to resolve inline scalar
  anchors such as `MATCH (n:Person {id: 42})` directly.
- Add seven correctness tests for the traversal semantics, two tests for
  bound-variable and optional matching, one multi-type parser test, and one
  numeric tag-index anchor test.
- Add a deterministic `graph_bench` that verifies complete result sets against a
  naive adjacency model and reports insert, anchor, and traversal latency.

## Capabilities

### New Capabilities

- `graph-traversal-benchmark`: deterministic graph dataset, naive adjacency
  oracle, and latency verification for fixed, variable-length, undirected,
  multi-type, edge-filtered, and optional traversals.

### Modified Capabilities

- `graph-query-language`: bound-variable reuse, correct optional match,
  variable-length semantics, all-label matching, multi-type syntax, and numeric
  scalar tag-index anchors.
- `graph-storage`: edge endpoint indexes, SQL inline property prefilter, and
  scalar numeric tag indexes.
- `tag-index`: exact tag values now include scalar numbers.

## Impact

- Modified: `crates/gvec-core/src/query.rs` (lexer, parser, pattern execution).
- Modified: `crates/gvec-core/src/storage.rs` (edge indexes, property prefilter).
- Modified: `crates/gvec-core/src/tag.rs` (numeric scalar tags).
- Modified: `crates/gvec-bindings/src/lib.rs` (tests).
- New: `crates/gvec-bindings/examples/graph_bench.rs`.
- Storage cost: two edge endpoint indexes and, when created, a tag index over
  the anchor property. The benchmark's `Person.id` tag index adds about 54
  bytes per node at 50k nodes.
- No new external crates and no dependency on `sqlite-graph` or `sqlite-vec`.
- Fixed patterns still enumerate paths; they do not deduplicate endpoint rows.
  Aggregation and `DISTINCT` remain out of scope.
- Unbounded variable-length traversal is relationship-unique but still
  exponential in the worst case; callers should bound it on dense graphs.
