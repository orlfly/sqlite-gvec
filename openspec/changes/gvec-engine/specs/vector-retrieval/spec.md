## ADDED Requirements

### Requirement: Virtual table vector storage

The system SHALL provide a `gvec0` virtual table that stores vectors in designated columns. Supported element types MUST include float32, int8, and binary (bit-string) at minimum. A newly created 1-column `gvec0` virtual table MUST support `INSERT` of a matching-length vector.

#### Scenario: Create and insert float vectors

- **WHEN** `CREATE VIRTUAL TABLE v USING gvec0(embedding float[8])` is executed and a 8-element float vector is inserted
- **THEN** the vector is stored and retrievable by rowid

#### Scenario: Reject mismatched vector length

- **WHEN** an insert supplies a vector whose length differs from the declared dimension
- **THEN** the insert fails with a dimension mismatch error

### Requirement: KNN search

The system SHALL support exact K-nearest-neighbor search over a virtual table given a query vector, returning the `k` closest rowids ordered by ascending distance. Supported distance metrics MUST include L2 (default), cosine, and inner product.

#### Scenario: L2 KNN returns nearest rowid

- **WHEN** a KNN query with `k=1` and query vector `[0,0]` is run against vectors `[1,0]` and `[3,0]`
- **THEN** the row containing `[1,0]` is returned as the single nearest result

#### Scenario: KNN with result limit

- **WHEN** a KNN query with `k=3` is run
- **THEN** at most 3 results are returned, ordered by ascending distance

#### Scenario: Selectable distance metric

- **WHEN** a KNN query requests cosine distance
- **THEN** results are ordered by ascending cosine distance

### Requirement: Point lookup

The system SHALL support single-row lookup by rowid via the virtual table, with a dedicated query plan distinct from full scan and KNN.

#### Scenario: Point query by rowid

- **WHEN** a virtual table row with rowid `5` is selected by `WHERE rowid = 5`
- **THEN** exactly that row's vector is returned

### Requirement: Metadata and auxiliary storage

The system SHALL allow non-vector scalar metadata columns on virtual tables alongside vector columns, and SHALL allow arbitrary per-row auxiliary data, both retrievable together with the vector.

#### Scenario: Metadata alongside vector

- **WHEN** a virtual table has a metadata column `length` populated on insert alongside the vector
- **THEN** both the vector and `length` are retrievable in a single row query

### Requirement: Encoding and round-trip

The system SHALL encode vectors losslessly in BLOB storage and MUST return elements identical to the original on retrieval, for float32, int8, and binary inputs.

#### Scenario: int8 round-trip

- **WHEN** an int8 vector `[1, -2, 3]` is inserted and retrieved
- **THEN** the retrieved vector equals `[1, -2, 3]`