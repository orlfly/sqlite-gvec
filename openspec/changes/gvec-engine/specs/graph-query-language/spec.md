## ADDED Requirements

### Requirement: Query function entry point

The system SHALL expose a `gvec_query()` function that accepts a Cypher-like query string and returns result rows. Results MUST be streamable as rows through a table-valued function, with a scalar variant returning a JSON document.

#### Scenario: Execute a simple match via TVF

- **WHEN** the function `gvec_query('MATCH (n:Person) RETURN n')` is invoked
- **THEN** one result row is returned per matching node

#### Scenario: Execute a simple match via scalar

- **WHEN** `gvec_query('MATCH (n:Person) RETURN n.name')` is invoked as a scalar
- **THEN** a JSON document containing the result set is returned

### Requirement: CREATE clause

The query language SHALL support creating nodes and edges with labels and properties via a `CREATE` clause, including creating a node-edge-node chain in a single statement.

#### Scenario: Create node chain

- **WHEN** `CREATE (a:Person {name: "Alice"})-[:KNOWS]->(b:Person {name: "Bob"})` is executed
- **THEN** two nodes and one typed edge are persisted and queryable

### Requirement: MATCH clause with patterns

The query language SHALL support `MATCH` patterns including node matching by label, node-edge-node relationships with a directed type filter, undirected relationships, and variable-length paths of the form `[*1..3]`.

#### Scenario: Directed relationship match

- **WHEN** `MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN b` is executed on a graph with an Alice-KNOWS-Bob edge
- **THEN** `b` resolves to Bob

#### Scenario: Variable-length path

- **WHEN** `MATCH (a:Person)-[:KNOWS*1..3]->(b) RETURN b` is executed
- **THEN** `b` resolves to all nodes reachable within 1 to 3 KNOWS hops from `a`

### Requirement: WHERE filtering

The query language SHALL support `WHERE` filtering with comparison operators (`=`, `<>`, `<`, `<=`, `>`, `>=`), and MUST support `AND`/`OR` composition and property access on node and relationship variables.

#### Scenario: Filter by property comparison

- **WHEN** `MATCH (n:Person) WHERE n.age > 25 RETURN n.name` is executed
- **THEN** only persons older than 25 are returned

#### Scenario: Compound filter

- **WHEN** `MATCH (n:Person) WHERE n.age > 25 AND n.city = "NYC" RETURN n` is executed
- **THEN** only persons meeting both conditions are returned

### Requirement: RETURN clause

The query language SHALL support `RETURN` of variables, node properties, relationship properties, aliases (`AS`), and ordering with `ORDER BY`, plus `SKIP` and `LIMIT` for pagination.

#### Scenario: Aliased property with order and limit

- **WHEN** `MATCH (n:Person) RETURN n.name AS name ORDER BY n.age DESC LIMIT 10` is executed
- **THEN** up to 10 person names are returned ordered by descending age

### Requirement: SET and DELETE clauses

The query language SHALL support updating properties on matched nodes and relationships via `SET`, and deleting matched nodes and relationships via `DELETE`.

#### Scenario: Update node property

- **WHEN** `MATCH (n:Person {name: "Alice"}) SET n.age = 31` is executed
- **THEN** Alice's `age` property is updated to 31

#### Scenario: Delete a relationship

- **WHEN** `MATCH (a)-[r:KNOWS]->(b) DELETE r` is executed
- **THEN** the matched KNOWS relationship is removed

### Requirement: OPTIONAL MATCH

The query language SHALL support `OPTIONAL MATCH` so that missing matches yield null-valued variables rather than dropping the row.

#### Scenario: Optional relationship

- **WHEN** `MATCH (n:Person) OPTIONAL MATCH (n)-[:KNOWS]->(f) RETURN n.name, f.name` is executed and some persons have no KNOWS relationship
- **THEN** those persons appear with `f.name` as null

### Requirement: Errors on invalid queries

The system SHALL return a clear parse or execution error for malformed or unsupported queries, including the position of a syntax error when determinable.

#### Scenario: Syntax error reporting

- **WHEN** a query with a syntax error such as an unclosed brace is executed
- **THEN** the system returns an error indicating a parse failure and the offending position
