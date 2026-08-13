## ADDED Requirements

### Requirement: Create text index

The system SHALL support creating an FTS5-backed full-text index over a node property via `CREATE TEXT INDEX ON :Label(property) [USING fts5(tokenizer)]`. The default tokenizer SHALL be `unicode61`.

#### Scenario: Create with default tokenizer

- **WHEN** `CREATE TEXT INDEX ON :Person(body)` is executed
- **THEN** an FTS5 virtual table is created and synced with all existing `Person` nodes' `body` property

#### Scenario: Create with porter stemmer

- **WHEN** `CREATE TEXT INDEX ON :Article(abstract) USING fts5(porter)` is executed
- **THEN** the FTS5 index uses the porter tokenizer for stemming

#### Scenario: Reject unknown tokenizer

- **WHEN** `CREATE TEXT INDEX ON :Person(body) USING fts5(unknown)` is executed
- **THEN** the operation fails with an error listing supported tokenizers

#### Scenario: Re-create on existing index is idempotent

- **WHEN** `CREATE TEXT INDEX ON :Person(body)` is executed a second time for an already-indexed `(label, property)` tuple
- **THEN** the operation succeeds without a UNIQUE rowid conflict, the existing FTS5 rows are replaced in place via `INSERT OR REPLACE`, and a subsequent `text_match` returns the same results as after the first call

### Requirement: Idempotent text index lifecycle

`create_text_index` (and the equivalent `CREATE TEXT INDEX` Cypher statement) MUST be idempotent for the same `(graph, label, property)` tuple. Repeated calls MUST NOT fail with a rowid UNIQUE conflict; they MUST leave the FTS5 shadow table and sync triggers in a state consistent with the current set of nodes for that label.

### Requirement: Text search via text_match

The system SHALL support a `text_match(variable.property, "query")` predicate that returns true for nodes whose indexed text property matches the FTS5 query. The predicate SHALL produce a BM25-ranked rowid set consumed by the executor.

#### Scenario: Basic text match

- **WHEN** `MATCH (n:Doc) WHERE text_match(n.body, "rust database") RETURN n.title` is executed against nodes with an FTS index on `body`
- **THEN** only nodes whose `body` matches "rust" OR "database" are returned

#### Scenario: Phrase match

- **WHEN** `text_match(n.body, '"rust database"')` is executed (quoted phrase)
- **THEN** only nodes containing the exact phrase "rust database" are returned

#### Scenario: No index returns error

- **WHEN** `text_match(n.body, "query")` is executed but no text index exists on `:Label.body`
- **THEN** the system returns an error indicating no text index is defined

### Requirement: BM25 score in results

The system SHALL expose the BM25 relevance score via a `text_score(variable)` expression usable in RETURN and ORDER BY.

#### Scenario: Return with score

- **WHEN** `MATCH (n:Doc) WHERE text_match(n.body, "rust") RETURN n.title, text_score(n) AS score ORDER BY score DESC` is executed
- **THEN** results include a `score` column with BM25 values in descending order

### Requirement: Drop text index

The system SHALL support dropping a text index via `DROP TEXT INDEX ON :Label(property)`, which removes the FTS5 virtual table and its sync triggers.

#### Scenario: Drop index

- **WHEN** `DROP TEXT INDEX ON :Person(body)` is executed
- **THEN** the FTS5 table and triggers are removed and subsequent `text_match` on that property fails

### Requirement: Sync on writes

The system SHALL keep the FTS5 index synchronized with node writes via SQLite triggers. INSERT, UPDATE, and DELETE on nodes SHALL automatically update the FTS5 table within the same transaction.

#### Scenario: Insert updates index

- **WHEN** a new node with label `Doc` and property `body = "hello world"` is created after a text index on `:Doc.body` exists
- **THEN** a subsequent `text_match(n.body, "hello")` finds the new node

#### Scenario: Delete removes from index

- **WHEN** a node matching `text_match(n.body, "hello")` is deleted
- **THEN** a subsequent `text_match(n.body, "hello")` no longer returns that node

### Requirement: SQL-level text search function

The system SHALL expose a `gvec_text_search(graph_name, label, property, query, k)` scalar SQL function that returns a JSON array of `{rowid, score}` pairs ranked by BM25.

#### Scenario: SQL-level search

- **WHEN** `SELECT gvec_text_search('g', 'Doc', 'body', 'rust database', 10)` is executed
- **THEN** a JSON array of up to 10 `{"rowid": N, "score": S}` objects is returned