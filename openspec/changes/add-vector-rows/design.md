## Context

The engine already persists vectors for declared ANN indexes in
`{prefix}_vectors (index_id, node_id, vector)`. That table answers "what is in
this index", not "what vector does this node have". The two questions diverge
when no index exists, when an embedding is stored under a nested path, or when
an application keeps vectors in a sidecar table.

The storage layer is SQLite with per-graph shadow tables. A vector row is small
and immutable in shape, so a dedicated table is cheaper than re-parsing node
JSON for every KNN query.

## Decisions

### One row per `(node_id, property)`

`{prefix}_vector_rows` uses `PRIMARY KEY (node_id, property)` and stores
`vector TEXT` (hex), `dim INTEGER`, and `element_type TEXT`. Hex keeps SQLite
values text-safe while `VectorBlob` provides exact encoding for float32, int8,
and binary vectors. The primary key makes explicit upserts idempotent and gives
`get_vector_row` an index-only lookup.

A foreign key to `{prefix}_nodes(id) ON DELETE CASCADE` mirrors graph
lifetime. `delete_node` also clears rows explicitly so the behaviour does not
depend on `PRAGMA foreign_keys`.

### Detection is conservative

Only two shapes are auto-detected on a normal node write:

- a non-empty JSON array whose elements are all numbers or null, and
- a typed vector object such as `{"type":"int8","values":[...]}`.

String arrays (tags, aliases) and scalar identifiers are ignored. Nested
paths and hex strings are persisted only when a vector index declares the
label/property, which is also the compatibility path for existing indexed
graphs.

### Exact KNN reads rows first

`exact_vector_search` loads vector rows for the requested label/property,
filters by dimension and element type, and runs the existing `knn_search`.
If no rows are found it falls back to the old JSON scan. The fallback keeps
older databases usable without a mandatory migration; rows are written the
next time a node is updated or an index is created.

On first creation of the table, vectors from existing `{prefix}_vectors` rows
are copied into `{prefix}_vector_rows`. This is a one-time migration and does
not rescan node JSON.

### Distance and score are distinct

The metric value is a distance: lower is better. `vector_distance(v)` returns
it unchanged. `vector_score(v)` converts it to a descending similarity:

| Metric | Distance | Score |
| --- | --- | --- |
| L2 | `d` | `1 / (1 + d)` |
| Cosine | `1 - cos` | `cos` |
| Inner product | `-dot` | `dot` |

This lets `ORDER BY vector_score(v) DESC` express "most similar first" without
metric-specific client logic. `vector_nn` still defaults to ascending distance
when no `ORDER BY` is present, preserving existing query semantics.

### Unlabeled `vector_nn`

If the predicate variable is not bound to a label, the planner passes an empty
label to `exact_vector_search`, which scans all vector rows for the property.
This makes `MATCH (n) WHERE vector_nn(n.embedding, ...)` work across labels.
An ANN index still requires a label because index definitions are per label.

### Explicit sidecar vectors

`upsert_vector_row` lets callers keep vectors outside node JSON. It validates
the node exists, checks any matching index's dimension and element type, writes
the row, and updates matching ANN index rows in the same call. Creating an
index later also merges existing explicit rows into the backfill, so sidecar
vectors are not lost. `node_vector` reads the row first, then falls back to the
JSON property.

## Risks and mitigations

- **False positives**: numeric arrays that are not embeddings become rows.
  Mitigation: conservative shape check and per-property filtering; storage
  cost is one encoded vector per numeric array. Applications with many numeric
  arrays can avoid them by using explicit `set_vector_row` and non-array
  properties.
- **Migration cost**: the first open of an old database copies index vectors
  into rows. Mitigation: single indexed query per existing vector index, not a
  full JSON scan.
- **Duplicate state**: a vector lives in both node JSON and a row. Mitigation:
  node writes replace rows derived from the old JSON state and preserve
  explicit sidecar rows for properties that never appeared in JSON, so JSON
  remains the source for ordinary properties. Explicit rows are overwritten
  only by `set_vector_row` or by writing the same property in JSON.
