## MODIFIED Requirements

### Requirement: Vector distance and score in results

The system SHALL expose `vector_distance(v)` as the metric distance produced by
the ANN or exact search, and `vector_score(v)` as a descending similarity
derived from that distance. For L2, `score = 1 / (1 + distance)`; for cosine,
`score = 1 - distance`; for inner product, `score = -distance`.

#### Scenario: Distance matches the metric

- **WHEN** `RETURN vector_distance(n)` is used with an L2 index
- **THEN** the returned value equals the L2 distance to the query vector

#### Scenario: Score orders most similar first

- **WHEN** a query uses `ORDER BY vector_score(n) DESC` with an L2 index
- **THEN** rows are ordered from smallest distance to largest

### Requirement: vector_nn without a label

`vector_nn(variable.property, query, k)` SHALL be executable when the variable
is not bound to a label, searching every node that has a vector row for the
property.

#### Scenario: Cross-label search

- **WHEN** `MATCH (n) WHERE vector_nn(n.embedding, [0,0], 3)` runs with `Doc`
  and `Other` nodes carrying `embedding`
- **THEN** the result set can contain nodes of either label, nearest first

#### Scenario: Exact fallback

- **WHEN** no ANN index exists for the property
- **THEN** the executor uses an exact scan over persisted vector rows
