## ADDED Requirements

### Requirement: Hybrid text + vector fusion

The system SHALL support queries that combine `text_match` and `vector_nn` predicates on the same MATCH. When both predicates are present, the planner SHALL intersect their rowid sets and rank results by reciprocal-rank fusion (RRF) unless an explicit `ORDER BY` overrides.

#### Scenario: Text + vector intersection

- **WHEN** `MATCH (n:Doc) WHERE text_match(n.body, "rust") AND vector_nn(n.embedding, :q, 10) RETURN n.title` is executed
- **THEN** only nodes that satisfy BOTH the text query AND are in the KNN top-10 are returned

#### Scenario: RRF default ordering

- **WHEN** a hybrid query has no explicit `ORDER BY`
- **THEN** results are ordered by descending reciprocal-rank fusion score: `1/(60+text_rank) + 1/(60+vector_rank)`

### Requirement: Three-way text + vector + graph fusion

The system SHALL support queries that start from a text and/or vector anchor and traverse the graph in the same MATCH.

#### Scenario: Text-anchored graph traversal

- **WHEN** `MATCH (d:Doc) WHERE text_match(d.body, "rust") MATCH (d)-[:CITES]->(r) RETURN r.title` is executed
- **THEN** the traversal starts from the text-matched documents and walks CITES edges to referenced documents

### Requirement: Score availability in hybrid results

When hybrid fusion is active, the system SHALL expose both `text_score(n)` and `vector_distance(n)` expressions so users can inspect and combine scores.

#### Scenario: Return both scores

- **WHEN** `MATCH (n:Doc) WHERE text_match(n.body, "rust") AND vector_nn(n.embedding, :q, 10) RETURN n.title, text_score(n) AS ts, vector_distance(n) AS vd` is executed
- **THEN** both score columns are populated in the result set