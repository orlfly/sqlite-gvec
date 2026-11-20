## Why

Vectors currently exist only in two places: the JSON node property and an
optional per-index table (`{prefix}_vectors`). A graph with no ANN index has no
persistent vector storage at all: `vector_nn` re-parses every node's JSON
property on every exact search, and `fusion::node_vector` cannot read a vector
back. This makes KNN, hybrid scoring, and sidecar embeddings awkward, and it
prevents vector rows from being used as a first-class graph feature.

## What Changes

- Add a `{prefix}_vector_rows` shadow table keyed by `(node_id, property)`,
  storing the encoded vector, dimension, and element type.
- Populate vector rows automatically on node create/update/delete for:
  - top-level arrays of numbers (`embedding: [0.1, 0.2]`),
  - typed vector objects (`{"type": "int8", "values": [...]}`),
  - properties declared by a vector index, including nested paths and hex
    strings.
- Backfill vector rows from existing vector indexes once, when the table is
  first created on an older database.
- Use vector rows as the authoritative source for exact KNN, with a JSON
  fallback for legacy rows that have not been rewritten yet.
- Add `Storage` / `Graph` APIs: `vector_row_count`, `vector_rows`,
  `get_vector_row`, `upsert_vector_row`, `node_vector`, and
  `set_vector_row`.
- Split `vector_distance(v)` from `vector_score(v)`: distance stays the metric
  value (ascending is better), score is a descending similarity
  (`1/(1+d)` for L2, `1-d` for cosine, `-d` for inner product).
- Allow `vector_nn` without a label, searching every node that has the
  property.

## Capabilities

### New Capabilities

- `vector-rows`: persistent per-node vectors independent of any ANN index,
  maintained on node writes and exposed through Rust/Python APIs.

### Modified Capabilities

- `graph-vector-integration`: `vector_nn` can run without a label, reads
  persistent rows for exact KNN, and exposes separate distance and similarity
  score expressions.

## Impact

- New table per graph: `{prefix}_vector_rows`, plus a property index.
- `create_node`, `update_node`, `set_node_labels`, and `delete_node` now
  maintain vector rows in addition to ANN index rows.
- `exact_vector_search` prefers vector rows; legacy JSON property scanning is
  retained as a migration fallback.
- README and Python bindings gain vector-row APIs.
