## Context

gvec already has three query constraints: graph patterns, FTS5 `text_match`,
and ANN `vector_nn`. All three are converted into rowid sets before graph
scanning, then intersected. Tags are a fourth kind of constraint with different
semantics: exact, case-sensitive membership over one or more values per node.
They are not full-text tokens and should not use FTS tokenizers.

The storage layer is SQLite shadow tables. Node properties are JSON in
`{prefix}_nodes.properties`. SQLite JSON1 provides `json_each`, so tag arrays
can be expanded inside triggers and backfill queries without a new dependency.

## Goals

- Exact, case-sensitive tag membership for JSON string and array properties.
- Persisted index with automatic maintenance on insert, update, delete, and
  label change.
- `tag_match(variable.property, "tag")` integrated into query planning.
- Multiple `tag_match` predicates on the same variable intersect.
- Correct results with or without a tag index; the unindexed path scans the
  property.
- Public lifecycle and search APIs in Rust and Python.
- Deterministic verification and benchmark.

## Non-Goals

- Case folding, Unicode normalization, or stemming of tags.
- Hierarchical tags, faceting, aggregation, or tag counts per value.
- Range predicates, numeric comparisons, or geo filters.
- Tag indexes on edge properties.
- Exact filtered-vector top-k. `vector_nn` still returns its k nearest
  candidates first, and `tag_match` filters those candidates afterwards. A
  future iteration can pass an allowed-id set into the ANN search.
- Backfilling historical databases with a migration framework. `ensure_schema`
  creates the tables on first use.

## Decisions

### Shared tag table with `index_id`

`{prefix}_tags(index_id, node_id, tag)` lets multiple tag indexes coexist
without table-name collisions. The primary key `(index_id, node_id, tag)`
supports fast node cleanup. A secondary index `(index_id, tag, node_id)`
supports tag lookup. Metadata lives in
`{prefix}_tag_indexes(id, label, property)` with a unique `(label, property)`.

### SQLite triggers

Triggers keep tags in sync without routing every write through Rust. This
matches the FTS approach and survives direct SQL writes. The insert and update
triggers use:

```sql
json_each(CASE json_type(column, '$.tags')
  WHEN 'array' THEN json_extract(column, '$.tags')
  WHEN 'text'  THEN json_quote(json_extract(column, '$.tags'))
END)
```

`json_quote` is required because `json_each` expects a JSON value; a bare
string such as `database` is malformed JSON. `j.type IN ('text','integer','real')`
skips objects, arrays, booleans, and nulls. Values are cast to TEXT, so numeric
tags match their decimal string form.

The update trigger always deletes the node's old rows for this index and then
conditionally inserts new ones. This handles property changes and label
addition/removal in one trigger.

### Rebuild on CREATE

`CREATE TAG INDEX` drops the old triggers, deletes rows for the index, backfills
from current nodes, and recreates triggers. This makes repeated DDL
idempotent and picks up nodes created since the last build.

### Planner integration

`extract_tag_match` collects `tag_match` expressions under top-level `AND`,
verifies they all target the same variable, resolves the label, runs
`storage.tag_search`, and intersects the sets. `OR`, `NOT`, and comparisons are
rejected because rowid-set pre-filtering would change semantics.

The executor represents vector, text, and tag constraints uniformly in
`CandidateConstraints`. When more than one constraint targets the first node
of a pattern, it fetches nodes by the smallest rowid set and then applies the
remaining constraints. Row-level filtering remains as a final correctness
guard.

### Fallback

If no tag index exists, `Storage::tag_search` scans nodes with the target label
and checks string/array membership. This keeps `tag_match` usable before DDL
and after `DROP TAG INDEX`, at the cost of a full label scan.

### Exact matching

Tags compare with `==` on strings. No lowercase conversion or stemming. This is
predictable for values such as `published`, `draft`, `topic/ai`, or a user id.
Case-insensitive tags should be normalized by the application before writing.

## Risks / Trade-offs

- Triggers fire for every tag index on every node update, even when the indexed
  property did not change. This is acceptable for a small number of indexes.
- `tag_match` with a broad tag materializes every matching node before `LIMIT`.
  The direct `tag_search` API is the fast path for top-k tag retrieval.
- Filtered vector search can return fewer than k rows when the tag filter is
  applied after ANN retrieval. This is documented and left for a future
  filtered-ANN iteration.
- Multiple tag indexes on the same `(label, property)` reuse one metadata row;
  changing the index definition in place is not supported because there are no
  options yet.

## Verification

- Rust e2e tests cover create/search, exactness, string and array values,
  multiple tags, insert/update/delete/label sync, text+tag, vector+tag, drop
  and fallback, reopen persistence, and unsupported predicate positions.
- The benchmark compares indexed lookups with the exact fallback and checks all
  tag result sets against a naive scan.
- Python smoke test exercises the lifecycle and tag-aware hybrid query.
