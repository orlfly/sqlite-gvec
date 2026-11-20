## ADDED Requirements

### Requirement: vector_nn as an ANN rowid-set constraint

The query executor SHALL extract `vector_nn(variable.property, query, k)` from
`WHERE`, run the configured ANN index (or an exact fallback), and restrict the
node scan to the resulting rowid set. It MUST return at most `k` matching
nodes.

#### Scenario: KNN through Cypher

- **WHEN** `MATCH (n:Doc) WHERE vector_nn(n.embedding, [0,0], 2) RETURN n.title` runs against vectors `[0,0]`, `[1,0]`, and `[5,5]`
- **THEN** exactly two rows are returned, nearest first

#### Scenario: Candidate intersection with a label

- **WHEN** `vector_nn` is combined with a label pattern
- **THEN** only nodes carrying that label participate in the ANN result set

### Requirement: Vector distance in results

The system SHALL expose `vector_distance(v)` and its alias `vector_score(v)`
in `RETURN` and `ORDER BY`, returning the distance produced by the ANN search.

#### Scenario: Distance matches the metric

- **WHEN** `RETURN vector_distance(n)` is used with an L2 index
- **THEN** the returned value equals the L2 distance to the query vector

### Requirement: Default distance ordering

When `vector_nn` is present and the query has no explicit `ORDER BY`, results
SHALL be ordered by ascending distance.

#### Scenario: No ORDER BY

- **WHEN** `MATCH (n:Doc) WHERE vector_nn(n.embedding, [0,0], 3) RETURN n.title` is executed
- **THEN** rows are returned from nearest to farthest

#### Scenario: Explicit ORDER BY wins

- **WHEN** the same query adds `ORDER BY n.title ASC`
- **THEN** title ordering overrides distance ordering

### Requirement: Hybrid text + vector intersection

`text_match` and `vector_nn` SHALL both act as rowid-set constraints so that
using both in one `WHERE` clause returns only rows satisfying both.

#### Scenario: Hybrid query

- **WHEN** three documents include two with body matching `"rust"` and the query applies `text_match(n.body, "rust") AND vector_nn(n.embedding, [1,0], 3)`
- **THEN** only the two text-matching documents are returned, ordered by distance

### Requirement: Inline array literals

The query language SHALL parse array literals in expressions, including node
property values and `vector_nn` query arguments.

#### Scenario: Create a node with an embedding

- **WHEN** `CREATE (n:Doc {title: "a", embedding: [0.1, 0.2, 0.3]})` is executed
- **THEN** the node stores the array as its `embedding` property

#### Scenario: Inline query vector

- **WHEN** `vector_nn(n.embedding, [0.1, 0.2], 5)` is parsed
- **THEN** the second argument evaluates to a float32 vector
