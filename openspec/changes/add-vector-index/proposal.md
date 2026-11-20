## Why

gvec can store graph nodes and search text with FTS5, but vector retrieval is
still brute force over caller-supplied in-memory arrays. `vector_nn(...)` is
parsed but evaluates to `true`, so it neither filters nor orders results, and
there is no persistent, incrementally maintained vector index. Applications
that want graph + full-text + semantic search in one library must currently
maintain a separate vector database (Milvus, Qdrant, pgvector) next to gvec.

Popular vector databases solve this with approximate nearest-neighbour (ANN)
indexes: Qdrant builds HNSW graphs, Milvus offers IVF_FLAT/IVF_PQ inverted
files, and both expose distance metrics plus per-collection metadata filters.
gvec should provide the same class of capability natively, persisted in SQLite
shadow tables, and integrated with its Cypher-like query language.

## What Changes

- Add a dependency-free ANN module to `gvec-core`:
  - **HNSW** (Qdrant-style hierarchical navigable small world) with
    deterministic level assignment, `m`/`m0`/`ef_construction`/`ef_search`.
  - **IVF** (Milvus-style inverted file) with farthest-first k-means,
    `nlist`/`nprobe`, float means and int8/binary medoids.
  - **Flat** exact scan, used as a fallback and recall oracle.
- Persist vector indexes in two new shadow tables per graph:
  `{prefix}_vector_indexes` (label, property, algorithm, metric, params, dim,
  element type) and `{prefix}_vectors` (index id, node id, encoded vector).
- Maintain vectors automatically on node create, property update, label
  change, and delete. Invalid or dimension-mismatched vectors are rejected
  before the node write commits.
- Add query-language DDL: `CREATE VECTOR INDEX ON :Label(property) USING
  HNSW(...)` / `USING IVF(...)` / `USING FLAT`, and `DROP VECTOR INDEX`.
- Turn `vector_nn(n.embedding, [..], k)` into a real ANN rowid-set constraint
  that intersects with graph patterns and `text_match`, returns at most `k`
  rows, exposes `vector_distance(n)`, and defaults to ascending-distance order.
- Support inline array literals in expressions, including node property
  values such as `CREATE (n:Doc {embedding: [0.1, 0.2]})`.
- Expose the index lifecycle and search through the Rust binding:
  `Graph::create_vector_index`, `drop_vector_index`, `list_vector_indexes`,
  `vector_count`, `vector_search`, `exact_vector_search`.
- Document the three-signal model: graph structure, vector similarity, and
  full-text relevance in one engine.

## Capabilities

### New Capabilities

- `vector-index`: persistent HNSW/IVF/Flat indexes over node properties, with
  distance metrics (L2, cosine, inner product), element types (float32, int8,
  binary), automatic maintenance, and lazy rebuild/caching.

### Modified Capabilities

- `graph-query-language`: array literals, `CREATE/DROP VECTOR INDEX` DDL,
  `vector_distance` / `vector_score` expressions.
- `graph-vector-integration`: `vector_nn` now plans as an ANN scan, intersects
  with text and graph constraints, preserves distance ordering, and exposes
  distances for hybrid ranking.

## Impact

- New module: `crates/gvec-core/src/vector_index.rs`.
- Modified: `crates/gvec-core/src/storage.rs` (shadow tables, lifecycle,
  node-write synchronization), `vector.rs` (JSON/hex encoding, element-type
  helpers), `query.rs` (DDL, arrays, planner/executor integration),
  `fusion.rs` (shared vector parser).
- Modified: `crates/gvec-bindings/src/lib.rs` (index API).
- No new external crates: HNSW and IVF are implemented in safe Rust.
- Storage cost: one encoded vector row per indexed node, plus rebuildable
  in-memory index structures. No custom on-disk file format.
