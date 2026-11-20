## Why

gvec supports graph traversal, BM25 full-text search, and vector retrieval, but
has no exact metadata/tag filter. Documents commonly carry a `tags` array, and
applications need queries such as "rust documents tagged database" or "nearest
vectors tagged published". Today tags can be stored and returned as JSON arrays,
but there is no membership operator, and combining a tag filter with text or
vector predicates requires scanning every labeled node.

Full-text search is the wrong tool for exact tags: tokenizers stem and split
values, so `database` can match `databases`, and tag values with spaces become
multiple tokens. Tags need exact, case-sensitive equality and their own index.

## What Changes

- Add a persisted tag index in `gvec-core`:
  - `{prefix}_tag_indexes` stores `(id, label, property)`.
  - `{prefix}_tags` stores `(index_id, node_id, tag)` with a primary key for
    node cleanup and an index on `(index_id, tag, node_id)` for lookups.
- Maintain tags automatically with SQLite triggers on node insert, update,
  delete, property changes, and label changes. Tags may be JSON strings or
  arrays of strings/numbers; values are compared exactly and case-sensitively.
- Add query-language DDL: `CREATE TAG INDEX ON :Label(property)` and
  `DROP TAG INDEX ON :Label(property)`.
- Add `tag_match(variable.property, "tag")` as a rowid-set constraint that
  intersects with graph patterns, `text_match`, and `vector_nn`.
- Allow multiple `tag_match` predicates on the same variable under `AND`; their
  result sets are intersected. Reject tags under `OR`/`NOT` or on different
  variables with an explicit error.
- Fetch matching nodes by rowid during graph scans, so selective tag queries do
  not materialize every labeled node.
- Fall back to an exact property scan when no tag index exists, so
  `tag_match` is usable before `CREATE TAG INDEX`.
- Expose `create_tag_index`, `drop_tag_index`, `list_tag_indexes`,
  `tag_count`, and `tag_search` in the Rust and Python bindings.
- Add a deterministic tag benchmark that verifies against a naive scan and
  compares indexed lookup with the fallback.

## Capabilities

### New Capabilities

- `tag-index`: persistent exact tag indexes over node properties, automatic
  trigger-based maintenance, exact tag lookup, and lifecycle APIs.

### Modified Capabilities

- `graph-query-language`: `CREATE/DROP TAG INDEX` and `tag_match(...)`.
- `graph-vector-integration`: tag rowid sets intersect with text and vector
  constraints in one MATCH.

## Impact

- New module: `crates/gvec-core/src/tag.rs`.
- Modified: `crates/gvec-core/src/storage.rs` (schema, lifecycle, lookup,
  fallback), `query.rs` (AST, parser, DDL, planner/executor constraints).
- Modified: `crates/gvec-bindings/src/lib.rs` and
  `crates/gvec-python/src/lib.rs` (public APIs).
- Storage cost: one row per `(index, node, tag)` plus a B-tree index, roughly
  170 to 190 bytes per document for three tags in the benchmark.
- No new external crates. SQLite JSON1 (`json_each`) is already available.
- The tag index is exact and case-sensitive by design in this iteration.
