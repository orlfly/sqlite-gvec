## ADDED Requirements

### Requirement: Vector index DDL statements

The query language SHALL support `CREATE VECTOR INDEX ON :Label(property)
[USING ALGORITHM(option = value, ...)]` and `DROP VECTOR INDEX ON
:Label(property)` as top-level statements alongside `CREATE TEXT INDEX`.

#### Scenario: Algorithm and options

- **WHEN** `CREATE VECTOR INDEX ON :Doc(embedding) USING HNSW(m = 16, ef_construction = 200, metric = "cosine")` is executed
- **THEN** the index is created with those parameters and reported by `list_vector_indexes()`

#### Scenario: Default algorithm

- **WHEN** `CREATE VECTOR INDEX ON :Doc(embedding)` is executed without `USING`
- **THEN** an HNSW index with default parameters and L2 metric is created

#### Scenario: Drop

- **WHEN** `DROP VECTOR INDEX ON :Doc(embedding)` is executed for an existing index
- **THEN** the definition and its vectors are removed

### Requirement: Array literals in expressions

The expression grammar SHALL accept array literals of literals and nested
expressions, allowing vector queries and vector property values to be written
inline.

#### Scenario: Array in RETURN

- **WHEN** `MATCH (n:Doc) RETURN [n.title, n.score] AS pair` is executed
- **THEN** each row contains a two-element JSON array

#### Scenario: Negative numeric literals

- **WHEN** a query contains `[-1.0, 0.0]`
- **THEN** the parser produces a float array with a negative first element
