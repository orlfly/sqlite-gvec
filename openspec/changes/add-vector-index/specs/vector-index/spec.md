## ADDED Requirements

### Requirement: Persistent vector index definition

The system SHALL allow a vector index to be defined over a node label and
property, persisted in SQLite shadow tables, and SHALL retain the index
definition across process restarts.

#### Scenario: Create an index through the Rust API

- **WHEN** `Graph::create_vector_index("Doc", "embedding", "hnsw", Some("l2"), None)` is called
- **THEN** a row exists in the graph's `vector_indexes` shadow table and `list_vector_indexes()` reports algorithm `hnsw` and metric `l2`

#### Scenario: Definition survives reopen

- **WHEN** a file-backed database with a vector index is closed and reopened
- **THEN** `list_vector_indexes()` returns the same definition and stored vector count

### Requirement: HNSW approximate search

The system SHALL provide an HNSW index with configurable `m`,
`ef_construction`, and `ef_search`, assigning levels deterministically from
vector ids so that rebuilding from the same persisted vectors is reproducible.

#### Scenario: High recall

- **WHEN** 500 deterministic 3-dimensional float vectors are indexed with `m=16`, `ef_construction=200`, `ef_search=80`
- **THEN** recall@10 measured against exact search over 25 queries is at least 0.9

#### Scenario: Deterministic rebuild

- **WHEN** the same vectors are indexed twice in the same order
- **THEN** both HNSW instances return identical search results

### Requirement: IVF approximate search

The system SHALL provide an inverted-file index with `nlist` lists and
`nprobe` probes, built with farthest-first initialisation for centroid
selection.

#### Scenario: IVF returns nearest neighbours

- **WHEN** 500 vectors are indexed with `nlist=12`, `nprobe=4`
- **THEN** at least 4 of the exact top-5 results appear in the IVF top-5

#### Scenario: Works for non-float element types

- **WHEN** int8 vectors are indexed with IVF
- **THEN** search returns the requested number of results without error (centroids use medoids)

### Requirement: Flat exact fallback

The system SHALL provide an exact flat scan and SHALL use it when a
`vector_nn` predicate targets a label/property with no vector index.

#### Scenario: No index defined

- **WHEN** `vector_nn(n.embedding, [0,0], 1)` is executed and no index exists on `Doc.embedding`
- **THEN** the nearest node is found by an exact scan

### Requirement: Distance metrics and element types

The system SHALL support L2 (default), cosine, and inner-product distance, and
float32, int8, and binary vectors.

#### Scenario: Cosine ordering

- **WHEN** an index with metric `cosine` contains `[1,0]`, `[0,1]`, and `[-1,0]` and the query is `[2,0]`
- **THEN** the first two results are `[1,0]` then `[0,1]`

#### Scenario: int8 vectors

- **WHEN** int8 vectors are stored via `{"type":"int8","values":[...]}` and queried with an int8 query
- **THEN** nearest results are returned

### Requirement: Automatic index maintenance

The system SHALL keep vector indexes synchronized when nodes are created,
updated, relabelled, or deleted, and SHALL validate vectors before the node
write so a rejected vector does not leave a partial node.

#### Scenario: Index created before nodes

- **WHEN** an index exists and two matching nodes are subsequently created
- **THEN** the index contains two vectors

#### Scenario: Update through SET

- **WHEN** `MATCH (n:Doc {title:"a"}) SET n.embedding = [9,9]` is executed
- **THEN** the index stores the new vector and searches reflect it without an explicit reindex

#### Scenario: Delete

- **WHEN** an indexed node is deleted
- **THEN** its vector row is removed

#### Scenario: Dimension mismatch

- **WHEN** an index has dimension 2 and a node with a 3-element vector is created
- **THEN** the write fails with a dimension error and the node is not persisted

### Requirement: Index lifecycle DDL

The system SHALL support `CREATE VECTOR INDEX ON :Label(property) USING
ALGORITHM(options)` and `DROP VECTOR INDEX ON :Label(property)`.

#### Scenario: Create with IVF options

- **WHEN** `CREATE VECTOR INDEX ON :Doc(embedding) USING IVF(nlist=4, nprobe=2, metric="cosine")` is executed
- **THEN** the index is created with `nlist=4`, `nprobe=2`, and cosine metric

#### Scenario: Idempotent re-create

- **WHEN** `CREATE VECTOR INDEX` is executed twice for the same label/property
- **THEN** the second call succeeds and the index is rebuilt from current nodes

#### Scenario: Drop

- **WHEN** `DROP VECTOR INDEX ON :Doc(embedding)` is executed
- **THEN** the definition and stored vectors are removed

### Requirement: Identifier safety

The system SHALL validate label and property names as plain SQL identifiers
before using them in DDL.

#### Scenario: Reject injection payload

- **WHEN** a label or property containing quotes, semicolons, or comment markers is supplied
- **THEN** index creation fails with a parse error and no DDL is executed
