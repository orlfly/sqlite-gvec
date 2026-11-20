## ADDED Requirements

### Requirement: Persistent vector rows

The system SHALL persist each node vector as a row in a
`{prefix}_vector_rows` shadow table keyed by `(node_id, property)`, storing the
encoded vector, dimension, and element type.

#### Scenario: Create a node with an embedding

- **WHEN** `CREATE (n:Doc {embedding: [0.1, 0.2]})` is executed
- **THEN** `vector_rows("Doc", "embedding")` returns one row for that node

#### Scenario: Update and remove

- **WHEN** a node's `embedding` property is changed with `SET` and later set to
  `null`
- **THEN** the stored row contains the new vector and is removed when the
  property becomes null

#### Scenario: Reopen

- **WHEN** a file-backed database with vector rows is closed and reopened
- **THEN** the same rows and vectors are readable

#### Scenario: Typed vectors

- **WHEN** a node stores `{"type":"int8","values":[1,2,3]}` under a property
- **THEN** the row preserves the int8 element type and values

### Requirement: Vector rows without an index

Vector rows SHALL exist and be searchable even when no ANN index is defined.

#### Scenario: Exact KNN without an index

- **WHEN** `exact_vector_search("Doc", "embedding", query, k, L2)` is called and
  no index exists
- **THEN** it returns the nearest persisted rows

#### Scenario: Label filter

- **WHEN** `vector_rows(Some("Doc"), Some("embedding"))` is called
- **THEN** only rows belonging to nodes with the `Doc` label are returned

### Requirement: Explicit vector rows

The system SHALL allow callers to upsert a vector row independently of the
node's JSON properties.

#### Scenario: Sidecar embedding

- **WHEN** `set_vector_row(node, "embedding", vector)` is called
- **THEN** `node_vector(node, "embedding")` returns that vector even if the
  node has no `embedding` property

#### Scenario: Index synchronization

- **WHEN** an ANN index exists on the node's label/property and
  `set_vector_row` is called
- **THEN** the index row is replaced and subsequent ANN search reflects the new
  vector

#### Scenario: Index created after a sidecar row

- **WHEN** a node has an explicit row but no matching JSON property, and an ANN
  index is later created for that label/property
- **THEN** the backfill includes the explicit row and ANN search can return the
  node

#### Scenario: Dimension or type mismatch

- **WHEN** `set_vector_row` supplies a vector whose dimension or element type
  differs from an existing index
- **THEN** the call fails with a validation error and no row is written

### Requirement: Vector row migration

The system SHALL backfill vector rows from existing ANN index rows when the
vector-row table is first created on an older database.

#### Scenario: Upgrade an indexed database

- **WHEN** a database with `{prefix}_vectors` rows is opened by a version that
  adds `{prefix}_vector_rows`
- **THEN** the indexed vectors appear in `vector_rows` without rewriting nodes
