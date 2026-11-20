## 1. Persistence (`gvec-core/src/storage.rs`)

- [x] 1.1 Add `{prefix}_vector_rows` table and property index
- [x] 1.2 Populate rows on `create_node` and `update_node`
- [x] 1.3 Delete rows on `clear_node_vectors` and `delete_node`
- [x] 1.4 Add `vector_row_count`, `vector_rows`, `get_vector_row`,
  `upsert_vector_row`
- [x] 1.5 Add conservative vector detection for numeric arrays and typed objects
- [x] 1.6 Backfill rows from existing vector indexes when the table is created
- [x] 1.7 Make `exact_vector_search` prefer rows with JSON fallback
- [x] 1.8 Keep explicit row upserts in sync with matching ANN indexes

## 2. Query language (`gvec-core/src/query.rs`)

- [x] 2.1 Add `Expr::VectorDistance` separate from `Expr::VectorScore`
- [x] 2.2 Compute similarity scores alongside distances in `extract_vector_nn`
- [x] 2.3 Annotate rows with `__vector_distance__` and `__vector_score__`
- [x] 2.4 Allow `vector_nn` without a label and search all nodes with the
  property
- [x] 2.5 Preserve ascending-distance default ordering

## 3. Fusion and bindings

- [x] 3.1 Implement `fusion::node_vector` from persisted rows with JSON fallback
- [x] 3.2 Export `VectorRow` from `gvec-core`
- [x] 3.3 Add Rust `Graph` APIs for rows and `node_vector`
- [x] 3.4 Add matching Python methods

## 4. Tests and docs

- [x] 4.1 CRUD and reopen persistence tests
- [x] 4.2 Exact KNN from rows without an index
- [x] 4.3 Unlabeled `vector_nn` and distance/score assertions
- [x] 4.4 int8 rows and dimension/type validation
- [x] 4.5 README and Python README examples
- [x] 4.6 Openspec proposal/design/specs
- [x] 4.7 Deterministic persistent-row benchmark (`vector_rows_bench`)
