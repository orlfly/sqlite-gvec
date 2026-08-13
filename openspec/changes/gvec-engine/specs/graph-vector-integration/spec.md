## ADDED Requirements

### Requirement: Vector predicate in graph queries

The system SHALL support a vector-similarity predicate inside graph queries (e.g., `WHERE vector_nn(n.embedding, :query, 10)`) that filters nodes by KNN distance to a query vector. The executor SHALL be able to use the vector index to plan such predicates as KNN scans rather than full graph scans.

#### Scenario: Vector filter on nodes

- **WHEN** a MATCH query on nodes uses `vector_nn(n.embedding, :query, 10)`
- **THEN** only the 10 nodes nearest to `:query` are matched and returned

#### Scenario: Vector filter combined with label

- **WHEN** a MATCH query restricts by label and applies a vector-nn predicate
- **THEN** results are limited to the nearest nodes among those carrying the label

### Requirement: Vector-anchored traversal

The system SHALL support starting a graph traversal from a vector anchor: a KNN result set seeds a MATCH, and the returned rows SHALL include the KNN distance for ordered, distance-aware traversal.

#### Scenario: Traverse from nearest neighbor

- **WHEN** a query finds the nearest node to a vector then traverses its `KNOWS` relationships
- **THEN** the traversed neighbors are returned together with the anchor's KNN distance

### Requirement: Distance ordering in fused queries

When a vector predicate is combined with graph filtering, the system SHALL preserve distance-based ordering in the result set unless an explicit `ORDER BY` overrides it.

#### Scenario: Distance ordering preserved

- **WHEN** a fused query with `vector_nn` has no explicit `ORDER BY`
- **THEN** results are ordered by ascending KNN distance

### Requirement: Consistent distance semantics

The vector distance computed inside graph queries SHALL match the distance produced by a standalone KNN query on the same vectors and metric.

#### Scenario: Cross-surface distance parity

- **WHEN** the same query vector and metric are used via standalone KNN and via `vector_nn` in a graph query
- **THEN** the reported distances are equal