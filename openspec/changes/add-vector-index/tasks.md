## 1. ANN core (`gvec-core/src/vector_index.rs`)

- [x] 1.1 Add `VectorIndexAlgorithm` (`flat`, `hnsw`, `ivf`) and config structs
- [x] 1.2 Implement `FlatIndex` exact scan over all element types and metrics
- [x] 1.3 Implement `HnswIndex`: deterministic levels, greedy descent,
  `searchLayer`, neighbour selection/pruning, `ef_construction` and
  `ef_search`
- [x] 1.4 Implement `IvfIndex`: farthest-first centroid initialisation,
  iterative assignment, `nlist`/`nprobe`, float means, int8/binary medoids
- [x] 1.5 Add unified `VectorIndex` enum with build/search/len/dim/algorithm
- [x] 1.6 Validate consistent dimension and element type at build time
- [x] 1.7 Unit tests: flat parities, HNSW recall ≥ 0.9, IVF recall, int8 and
  binary vectors, mixed-dimension rejection, empty index, deterministic levels

## 2. Persistence (`gvec-core/src/storage.rs`)

- [x] 2.1 Create `{prefix}_vector_indexes` and `{prefix}_vectors` shadow tables
- [x] 2.2 Add `VectorIndexMeta` with config parsing and cache
- [x] 2.3 Implement `create_vector_index` with option parsing, validation,
  backfill, restartable reindex, and idempotent reconfiguration
- [x] 2.4 Implement `drop_vector_index`, `vector_count`, `vector_indexes`,
  `find_vector_index`
- [x] 2.5 Synchronize vectors on `create_node`, `update_node`,
  `set_node_labels`, and `delete_node`; validate before node write
- [x] 2.6 Implement `vector_search` with lazy `Arc<VectorIndex>` cache and
  invalidation on writes
- [x] 2.7 Implement `exact_vector_search` fallback for unindexed properties
- [x] 2.8 Reject dimension/type inconsistencies during backfill and node writes

## 3. Encoding (`gvec-core/src/vector.rs`)

- [x] 3.1 Add `VectorValue::from_json` supporting arrays, hex strings, and
  typed `float32` / `int8` / `binary` objects
- [x] 3.2 Add `VectorBlob::to_hex` / `from_hex` for SQLite-safe persistence
- [x] 3.3 Add `ElementType::as_str`
- [x] 3.4 Share the parser with `fusion::vector_from_scalar`

## 4. Query language (`gvec-core/src/query.rs`)

- [x] 4.1 Parse inline array literals and negative numeric literals
- [x] 4.2 Add `CREATE VECTOR INDEX ON :Label(property) USING
  ALGORITHM(key = value, ...)` and `DROP VECTOR INDEX` AST + parser
- [x] 4.3 Add `Expr::VectorScore` for `vector_distance(v)` / `vector_score(v)`
- [x] 4.4 Extract `vector_nn` into an ANN rowid set before graph expansion
- [x] 4.5 Resolve the label bound to the predicate variable and run
  `Storage::vector_search`
- [x] 4.6 Intersect vector candidates with graph rows and FTS rowids; attach
  distances to rows
- [x] 4.7 Default to ascending-distance ordering when no explicit `ORDER BY`
- [x] 4.8 Support multi-key `ORDER BY`
- [x] 4.9 Route `SET` node updates through `Storage::update_node` so vectors
  stay in sync

## 5. Bindings and tests

- [x] 5.1 Add `Graph::create_vector_index`, `drop_vector_index`,
  `list_vector_indexes`, `vector_count`, `vector_search`,
  `exact_vector_search`
- [x] 5.2 End-to-end tests: Cypher KNN, automatic maintenance, reopen
  persistence, IVF DDL + drop, hybrid text+vector, dimension rejection, int8,
  cosine metric
- [x] 5.3 Full workspace test suite green

## 6. Docs

- [x] 6.1 README quick-start for vector indexes and hybrid search
- [x] 6.2 Openspec proposal/design/specs for the new capability
- [x] 6.3 Deterministic recall/latency benchmark (`vector_bench` example) and
  README results table
- [x] 6.4 Query-time `ef_search` override (`search_with_ef`) for recall/latency
  tuning without a rebuild
