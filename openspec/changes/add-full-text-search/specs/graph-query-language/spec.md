## ADDED Requirements

### Requirement: text_match predicate

The query language SHALL support a `text_match(variable.property, "query")` predicate usable in `WHERE` clauses. The predicate SHALL be recognized by the planner as a rowid-set constraint (not just a boolean filter), enabling intersection with `vector_nn` and graph traversal.

#### Scenario: text_match in WHERE

- **WHEN** `MATCH (n:Doc) WHERE text_match(n.body, "rust") RETURN n.title` is executed
- **THEN** the executor uses the FTS5 index to constrain the node scan to matching rowids

### Requirement: text_score expression

The query language SHALL support `text_score(variable)` in RETURN and ORDER BY, returning the BM25 score of the most recent `text_match` for that variable.

#### Scenario: Order by text score

- **WHEN** `MATCH (n:Doc) WHERE text_match(n.body, "rust") RETURN n.title ORDER BY text_score(n) DESC` is executed
- **THEN** results are ordered by descending BM25 score

### Requirement: CREATE TEXT INDEX statement

The query language SHALL support `CREATE TEXT INDEX ON :Label(property) [USING fts5(tokenizer)]` as a top-level DDL statement.

#### Scenario: Create index via Cypher-like DDL

- **WHEN** `CREATE TEXT INDEX ON :Doc(body) USING fts5(porter)` is executed via `gvec_query`
- **THEN** the FTS5 index is created and available for `text_match`

### Requirement: DROP TEXT INDEX statement

The query language SHALL support `DROP TEXT INDEX ON :Label(property)` as a top-level DDL statement.

#### Scenario: Drop index via Cypher-like DDL

- **WHEN** `DROP TEXT INDEX ON :Doc(body)` is executed via `gvec_query`
- **THEN** the FTS5 index and triggers are removed