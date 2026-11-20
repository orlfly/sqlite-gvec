## Context

`gvec-core` stores nodes and edges in SQLite shadow tables and executes a
Cypher-like language. Full-text search uses FTS5 rowid sets. Vector support so
far has been an in-memory brute-force helper (`knn_search`) with no persisted
index and no executor integration: `Expr::VectorNn` evaluated to `true`.

The target is a single embedded store where a query can combine all three
signals:

```cypher
MATCH (d:Doc)
WHERE text_match(d.body, "rust") AND vector_nn(d.embedding, [0.2, 0.9, 0.1], 10)
RETURN d.title, text_score(d), vector_distance(d)
ORDER BY vector_distance(d) ASC
```

## Goals / Non-Goals

**Goals**
- Native, dependency-free ANN indexes: HNSW (Qdrant-style), IVF (Milvus-style),
  Flat exact.
- Persist vectors and index definitions in SQLite; reopen and search without
  external state.
- Keep indexes current on node insert/update/delete.
- Support float32, int8, and binary vectors, and L2 / cosine / inner-product
  distance.
- Make `vector_nn` a true rowid-set constraint that composes with text and
  graph constraints.
- Expose index lifecycle and search in the Rust binding.
- Deterministic, reproducible builds of ANN structures from persisted data.

**Non-Goals**
- Product quantization (IVF-PQ), scalar/binary quantization, or DiskANN.
- Distributed sharding, replication, or external index services.
- GPU acceleration.
- Persisting the ANN graph links themselves in this iteration (they are
  deterministically rebuildable from vectors).
- Vector indexes over edge properties or multiple vector columns per node.

## Decisions

### D1. Algorithm portfolio: HNSW default, IVF optional, Flat fallback
- **HNSW** is the default because it offers the best recall/latency trade-off
  for the append-heavy workloads gvec targets and mirrors Qdrant's core index.
  Levels are assigned from `splitmix64(node_id)` so rebuilding produces the
  same graph. Each node keeps `m` neighbours above layer 0 and `m0 = 2m` at
  layer 0; construction uses `ef_construction`, search uses `ef_search`.
- **IVF** mirrors Milvus `IVF_FLAT`: `nlist` clusters built with farthest-first
  initialisation followed by up to 10 assignment/update iterations, queries
  probe `nprobe` nearest lists. Float centroids are arithmetic means; int8 and
  binary centroids are medoids so no invalid arithmetic is invented.
- **Flat** is exact and is both a fallback when no index exists and the recall
  oracle in tests.
- Alternative considered: linking `hnsw-rs`/`usearch`. Rejected to keep the
  engine self-contained and to control serialization and metric semantics.

### D2. Persistence: vectors + definition, rebuild index lazily
`CREATE VECTOR INDEX` writes a row to `{prefix}_vector_indexes` and vector
payloads to `{prefix}_vectors`. Vectors are hex-encoded `VectorBlob`s (type
byte + little-endian dimension + payload), which round-trip through the
JSON-shaped `SqlExecutor` surface without needing a new BLOB API.

The ANN graph is cached in a `Mutex<HashMap<(label, property), Arc<VectorIndex>>>`
inside `Storage` and rebuilt from persisted vectors after any write or process
restart. This trades a rebuild for O(1) write amplification; it also guarantees
that the persisted truth is the vectors themselves, not a potentially corrupt
serialized graph.
- Alternative considered: persist HNSW adjacency rows and update incrementally.
  Rejected for this iteration because it requires per-hop SQL or a much larger
  cache-coherence design; deterministic rebuild keeps correctness simple.
- Trade-off: caches are per-`Storage`. A second `Graph` handle (or a separate
  process) does not observe vector writes or index-definition changes made
  through the first until it rebuilds its own cache; the single-writer/embedded
  model is the primary use case. Reopening a database always starts from the
  persisted vectors and metadata.

### D3. Automatic maintenance on the write path
`Storage::create_node`, `update_node`, `set_node_labels`, and `delete_node`
synchronize vectors for every index whose label the node carries:
1. `prepare_node_vectors` parses and validates the node's vector properties
   *before* the node row is written, so dimension/type errors abort cleanly.
2. The node row is written.
3. Staged vectors are upserted; on update/delete, stale rows are removed.

`SET n.embedding = [...]` routes through `Storage::update_node` rather than a
raw SQL update so the index stays consistent.

### D4. Query planning: ANN rowid set, not a boolean
During `MATCH`, the executor extracts `vector_nn` from `WHERE`, resolves the
label that binds the variable, runs the ANN search, and uses the resulting ids
to constrain the first-node scan. Rows are then restricted to the candidate set
and annotated with `__vector_distance__`. `text_match` behaves the same way
with BM25 rowids, so the two sets intersect naturally. With no explicit
`ORDER BY`, rows keep ascending-distance order. `vector_distance(v)` /
`vector_score(v)` expose the annotation to `RETURN` and `ORDER BY`.

### D5. Element types and metrics
`VectorValue` already models `F32`, `I8`, and bit-packed `Bits`. The same index
implementations operate on all three through `distance_metric`. JSON arrays
mean float32; hex strings and typed objects (`{"type":"int8","values":[...]}`,
`{"type":"binary","bits":[...]}`) cover the other element types. An index locks
its dimension and element type on first use and rejects mismatches.

## Risks / Trade-offs

- [HNSW recall depends on parameters] → Defaults `m=16`, `ef_construction=200`,
  `ef_search=64`; tests enforce ≥90% recall@10 on a 500-vector dataset.
- [Rebuild cost after writes] → Indexes are cached after the first query;
  writes invalidate lazily, so a batch of writes costs one rebuild, not one
  per write.
- [Vector data stored as hex text] → Larger than BLOB by 2×. Chosen to preserve
  the existing JSON-shaped executor contract; a future BLOB API can migrate it.
- [No quantization] → Memory scales with raw vectors. PQ/quantization can be
  added behind the same `VectorIndex` enum without changing the query surface.
- [Label/property identifiers in DDL] → Validated as plain SQL identifiers
  (letter/underscore start, alphanumeric/underscore) before interpolation.

## Migration

No migration is required. The two new shadow tables are created idempotently
by `Storage::ensure_schema`; existing databases gain them on next open.
