## ADDED Requirements

### Requirement: Persisted exact tag indexes

The system SHALL provide `CREATE TAG INDEX ON :Label(property)` and
`DROP TAG INDEX ON :Label(property)`. A tag index SHALL persist one row per
`(index, node, tag)` and SHALL support multiple indexes per graph.

#### Scenario: Create and query a tag index

- **WHEN** documents contain `tags: ["database", "rust"]`
- **AND** the user runs `CREATE TAG INDEX ON :Doc(tags)`
- **THEN** `tag_match(n.tags, "database")` returns those documents
- **AND** matching is exact and case-sensitive

#### Scenario: Scalar string tags

- **WHEN** a document contains `tags: "database"`
- **THEN** the tag index stores `database`
- **AND** `tag_match(n.tags, "database")` matches it

#### Scenario: Scalar numeric tags

- **WHEN** a node contains `id: 42`
- **AND** `CREATE TAG INDEX ON :Person(id)` exists
- **THEN** the tag index stores `"42"` and `tag_search("Person", "id", "42")`
  returns that node
- **AND** an inline anchor `MATCH (n:Person {id: 42})` may use the index

#### Scenario: Multiple tag indexes coexist

- **WHEN** tag indexes exist on `Doc.tags` and `Doc.category`
- **THEN** each has independent metadata, rows, and triggers
- **AND** dropping one does not affect the other

### Requirement: Automatic tag maintenance

Tag indexes SHALL stay synchronized when nodes are inserted, updated, deleted,
or change labels.

#### Scenario: Insert after index creation

- **WHEN** a tag index already exists
- **AND** a new matching node is created
- **THEN** its tags are searchable immediately

#### Scenario: Update and delete

- **WHEN** a node's tag property changes
- **THEN** old tags are removed and new tags are added
- **WHEN** a node is deleted
- **THEN** all of its tag rows are removed

#### Scenario: Label change

- **WHEN** a node loses the indexed label
- **THEN** its tags are removed from that index
- **WHEN** the label is added back
- **THEN** its tags are indexed again

### Requirement: Tag predicate integration

`tag_match(variable.property, "tag")` SHALL act as a rowid-set constraint that
intersects with graph patterns, `text_match`, and `vector_nn`.

#### Scenario: Text and tag

- **WHEN** a query combines `text_match(n.body, "rust")` with
  `tag_match(n.tags, "database")`
- **THEN** only rows satisfying both predicates are returned

#### Scenario: Vector and tag

- **WHEN** a query combines `vector_nn(...)` with `tag_match(...)`
- **THEN** the ANN candidates are filtered by the tag set
- **AND** the executor may return fewer than k rows when the filter is
  selective

#### Scenario: Multiple tags

- **WHEN** two `tag_match` predicates target the same variable under `AND`
- **THEN** their rowid sets are intersected

#### Scenario: Unsupported positions

- **WHEN** `tag_match` appears under `OR` or `NOT`
- **THEN** the query fails with an explicit error
- **WHEN** multiple `tag_match` predicates target different variables
- **THEN** the query fails with an explicit error

### Requirement: Indexed and fallback lookup

The system SHALL answer tag queries with the index when present and with an
exact property scan otherwise.

#### Scenario: No index

- **WHEN** no tag index exists for the property
- **THEN** `tag_match` still returns exact matches by scanning the property

#### Scenario: Drop and fallback

- **WHEN** a tag index is dropped
- **THEN** tag queries remain correct through the fallback path
- **AND** recreating the index restores indexed lookup

### Requirement: Binding surface

The Rust and Python bindings SHALL expose tag index lifecycle and lookup.

#### Scenario: Rust API

- **WHEN** the caller uses `create_tag_index`, `list_tag_indexes`,
  `tag_count`, `tag_search`, or `drop_tag_index`
- **THEN** the calls operate on the persisted index

#### Scenario: Python API

- **WHEN** the caller uses the same methods through PyO3
- **THEN** metadata is returned as dictionaries and search returns node ids
