## MODIFIED Requirements

### Requirement: Graph pattern matching

`MATCH` SHALL support fixed, undirected, multi-type, and variable-length
relationship patterns. A node pattern with multiple labels SHALL require every
label. Consecutive `MATCH` clauses SHALL reuse an already-bound variable instead
of rescanning and overwriting it.

#### Scenario: Reuse a bound node

- **WHEN** a query runs `MATCH (a:Person {id: 1}) MATCH (a)-[:KNOWS]->(b)`
- **THEN** the second pattern starts from the node bound to `a`
- **AND** it does not scan every `Person` node again

#### Scenario: Optional match after match

- **WHEN** a query runs `MATCH (a:Person {id: 1}) OPTIONAL MATCH
  (a)-[:MANAGES]->(b)`
- **AND** no `MANAGES` edge exists
- **THEN** one row is returned with `a` bound and `b` null
- **AND** when an edge exists, the row contains the matched `b`

#### Scenario: Multiple labels

- **WHEN** a node has labels `["Person", "Employee"]`
- **THEN** `MATCH (n:Person:Employee)` matches it
- **AND** a node with only `Person` does not match

#### Scenario: Multiple relationship types

- **WHEN** a query uses `-[:KNOWS|LIKES]->`
- **THEN** edges of either type match
- **AND** edges of other types do not match

### Requirement: Variable-length paths

A variable-length relationship pattern SHALL enumerate relationship-unique
paths. `*` SHALL mean `1..inf`, `*3` exactly three, `*..3` `1..3`, `*2..`
`2..inf`, and `*0..2` `0..2`. Every accepted endpoint SHALL satisfy the
destination node's labels and inline properties. The relationship variable
SHALL bind to an array of edge objects. Fixed segments after the variable-length
segment SHALL execute.

#### Scenario: Unbounded path on a cycle

- **WHEN** an unbounded `*` pattern runs on a graph containing a cycle
- **THEN** each path may reuse a node but not a relationship
- **AND** enumeration terminates

#### Scenario: Destination filter

- **WHEN** a variable-length pattern ends at `(b:Person {active: true})`
- **THEN** only endpoints satisfying both the label and the inline property are
  returned

#### Scenario: Zero-length path

- **WHEN** the pattern uses `*0..2`
- **THEN** the zero-length path binds the start node to the destination variable
- **AND** the relationship variable is an empty array

### Requirement: Inline scalar anchors

The executor SHALL push top-level scalar inline properties into a SQL
`json_extract` prefilter and SHALL retain the exact Rust-side property match.
When an exact tag index exists for the label and property, the executor SHALL
use it to resolve an integer or string anchor.

#### Scenario: Indexed anchor

- **WHEN** an exact tag index `CREATE TAG INDEX ON :Person(id)` exists
- **AND** the query uses `MATCH (n:Person {id: 42})`
- **THEN** the executor resolves the anchor through the tag index
- **AND** still applies the exact property comparison

#### Scenario: No index

- **WHEN** no tag index exists on the anchor property
- **THEN** the query remains correct through the SQL prefilter and Rust filter
- **AND** lookup time is linear in the label size
