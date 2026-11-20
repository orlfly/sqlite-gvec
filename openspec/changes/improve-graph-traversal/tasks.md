## 1. Correctness

- [x] 1.1 Reuse an already-bound first variable in `execute_match` instead of
  rescanning the label and overwriting the binding
- [x] 1.2 Keep the base row for a missing `OPTIONAL MATCH`, including when the
  first variable is bound
- [x] 1.3 Make variable-length traversal relationship-unique so bare `*`
  terminates on cycles
- [x] 1.4 Filter destination labels and inline properties at every accepted
  variable-length endpoint
- [x] 1.5 Apply edge-type filters at every hop and bind the relationship variable
  to an array of path edges
- [x] 1.6 Execute fixed pattern segments after a variable-length segment
- [x] 1.7 Require all labels in a multi-label node pattern
- [x] 1.8 Parse bare `*`, `*3`, `*..3`, `*2..`, and `*0..2`
- [x] 1.9 Lex `|` so `[:A|B]` patterns parse and execute

## 2. Storage and performance

- [x] 2.1 Add `{prefix}_edges_source` and `{prefix}_edges_target` indexes
- [x] 2.2 Push top-level scalar inline properties into `json_extract` as a SQL
  prefilter, with exact Rust filtering retained
- [x] 2.3 Extend tag indexes to scalar numbers in backfill, triggers, and the
  fallback matcher
- [x] 2.4 Resolve inline scalar anchors through a matching tag index when one
  exists

## 3. Tests

- [x] 3.1 Variable-length zero-length plus a remaining fixed segment
- [x] 3.2 Variable-length destination label and property filtering
- [x] 3.3 Fixed-pattern filtering on later node properties
- [x] 3.4 Multi-label all-labels requirement
- [x] 3.5 Undirected self-loop edge deduplication
- [x] 3.6 Variable-length relationship variable is an array
- [x] 3.7 Unbounded variable-length termination on a cycle
- [x] 3.8 Consecutive `MATCH` reuses the bound variable
- [x] 3.9 `OPTIONAL MATCH` after `MATCH` keeps the row on miss and returns the
  edge on hit
- [x] 3.10 Multi-type `[:A|B]` matches every listed type and nothing else
- [x] 3.11 Numeric tag index lookup, inline anchor acceleration, and update
  synchronization

## 4. Benchmark

- [x] 4.1 Deterministic directed graph with 10k and 50k nodes, four `FOLLOWS`
  and one `LIKES` edge per node
- [x] 4.2 Naive adjacency model that verifies complete result sets without
  `LIMIT`
- [x] 4.3 Fixed 1/2/3-hop, variable 1..2 and 1..3, undirected, multi-type,
  edge-property, and optional-miss cases
- [x] 4.4 Compare anchor scan with `Person.id` tag-index lookup
- [x] 4.5 Report insert throughput, storage size, index build time, p50, and p95

## 5. Docs and finalize

- [x] 5.1 README graph section with semantics, benchmark table, and limitations
- [x] 5.2 Openspec proposal, design, and task list
- [x] 5.3 Run fmt, clippy with `-D warnings`, and full workspace tests
- [x] 5.4 Commit the graph traversal work
