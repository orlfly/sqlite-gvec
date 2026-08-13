## ADDED Requirements

### Requirement: Node storage

The system SHALL store graph nodes with a unique integer id, zero or more labels, and zero or more properties persisted in SQLite shadow tables. Node writes MUST be atomic within a SQLite transaction.

#### Scenario: Create node with labels and properties

- **WHEN** a node is created via the graph API with labels `["Person"]` and properties `{"name": "Alice", "age": 30}`
- **THEN** the node is persisted in the shadow table and queryable by its integer id

#### Scenario: Node without properties

- **WHEN** a node is created with only a label and no properties
- **THEN** the node persists with an empty property map and remains queryable

### Requirement: Edge storage

The system SHALL store directed edges with a unique integer id, source node id, target node id, an optional edge type, and optional properties persisted in SQLite shadow tables. Edges MUST reference existing node ids.

#### Scenario: Create typed edge between existing nodes

- **WHEN** an edge is created from node 1 to node 2 with type `KNOWS`
- **THEN** the edge is persisted and joinable to both source and target nodes

#### Scenario: Reject edge with missing node

- **WHEN** an edge is created whose source or target id does not exist
- **THEN** the operation fails with a constraint error and no edge is written

### Requirement: Persistence and durability

The system SHALL persist all graph data, including node and edge updates, in the backing SQLite database file. Data written in a committed transaction MUST survive a process restart and a reload of the extension.

#### Scenario: Data survives reload

- **WHEN** nodes and edges are committed, the connection is closed, the extension is reloaded, and the database is reopened
- **THEN** all committed nodes and edges are retrievable

### Requirement: Property values

The system SHALL store node and edge properties as JSON and MUST support scalar types: string, number, boolean, null, arrays, and nested objects. Property access MUST support dotted paths such as `n.name`.

#### Scenario: Nested property access

- **WHEN** a node has properties `{"address": {"city": "NYC"}}`
- **THEN** the query language can access `n.address.city` and returns `"NYC"`

### Requirement: Delete operations

The system SHALL support deleting nodes and edges. Deleting a node MUST delete all incident edges or fail atomically; the behavior SHALL be consistent within a single transaction.

#### Scenario: Delete node removes incident edges

- **WHEN** a node with incident edges is deleted
- **THEN** the node and all of its incident edges are removed atomically

### Requirement: Node identity

The system SHALL expose each node's integer id as a first-class value accessible in queries and usable as an anchor for vector and graph operations.

#### Scenario: Query by id

- **WHEN** a node with id `42` is matched in a query
- **THEN** the returned node record includes id `42`
