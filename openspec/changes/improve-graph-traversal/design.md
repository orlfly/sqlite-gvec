## Context

`gvec-core/src/query.rs` executes a Cypher-like subset directly against the
shadow tables. It has no cost-based planner: each `MATCH` statement produces a
row set, each pattern starts from the first node pattern, and fixed segments
extend one relationship at a time. That design is easy to reason about, but it
made several assumptions visible only under realistic graph shapes.

The storage layer keeps nodes in `{prefix}_nodes` (`id`, `labels` JSON,
`properties` JSON) and edges in `{prefix}_edges` (`id`, `source`, `target`,
`edge_type`, `properties` JSON). Vector and tag indexes live in their own shadow
tables with triggers.

## Goals

- Correct semantics for bound variables, `OPTIONAL MATCH`, variable-length
  patterns, multi-label nodes, and multi-type relationships.
- Sub-millisecond traversal hops at 50k nodes and 250k edges, with a documented
  path to fast anchor lookup.
- Preserve exactness and existing APIs. No new dependencies.

## Non-goals

- Aggregation, `DISTINCT`, `COUNT`, shortest-path algorithms, and a general
  cost-based planner.
- `MATCH ... CREATE` (creating relationships between already-matched nodes).
- Automatic secondary indexes for every node property.

## Decisions

### Bound first variable

`execute_match` now checks whether the first node variable is already present in
the incoming rows. If so, it walks those rows, validates each bound node against
the pattern's labels and properties, and calls `extend_pattern` from that node.
It no longer scans the label and overwrites the binding.

`OPTIONAL MATCH` keeps the base row when `extend_pattern` returns no path. This
also fixes the common sequence:

```cypher
MATCH (a:Person {id: 42})
OPTIONAL MATCH (a)-[:MANAGES]->(b)
RETURN a.name, b.name
```

### Relationship-unique variable-length enumeration

`extend_pattern` treats a variable-length segment as a DFS stack of
`(row, node, path_edges, visited_edge_ids)`. At every pop, if the current depth
is at least `lo`, the endpoint is checked against the destination node pattern
(labels and inline properties) and emitted. If the depth is below `hi`, outgoing
edges matching the segment's types are expanded, skipping edges already in
`visited_edge_ids`.

Relationship uniqueness is what makes bare `*` terminate on a cycle: a path can
revisit a node, but not an edge. Fixed segments, by contrast, intentionally
enumerate paths and may revisit both nodes and edges. This matches path
semantics rather than simple-path semantics.

A relationship variable on a variable-length segment binds to a JSON array of
edge objects. On a fixed segment it binds to one edge object.

### Edge endpoint indexes

`ensure_schema` creates:

```sql
CREATE INDEX IF NOT EXISTS {prefix}_edges_source ON {prefix}_edges(source);
CREATE INDEX IF NOT EXISTS {prefix}_edges_target ON {prefix}_edges(target);
```

`list_edges(Some(source), None)` and `list_edges(None, Some(target))` now use
these indexes. Undirected traversal unions the two lists and deduplicates by
edge id.

### Inline property prefilter

`MATCH (n:Person {id: 42})` used to call `list_nodes_by_label("Person")`, decode
every node's labels and properties in Rust, and then filter. At 50k nodes that
cost about 24 ms per query, dwarfing the traversal.

`Storage::list_nodes_by_label_and_properties` pushes top-level scalar literals
into SQL:

```sql
SELECT id, labels, properties FROM {prefix}_nodes
WHERE labels LIKE ?1
  AND json_extract(properties, ?2) = ?3
```

The Rust `properties_match` check still runs as the exactness guard. Nested
paths, arrays, objects, and keys that would need JSON-path escaping are skipped
by the prefilter and handled entirely in Rust. The prefilter alone reduced the
50k-node anchor scan to about 24 ms from 80 ms; it is still linear in the label
size.

### Tag-index anchors

Exact tag indexes now include scalar numbers. Their backfill and trigger
expressions accept JSON `integer` and `real` in addition to `text` and `array`.
Values are stored as `CAST(value AS TEXT)`, and `value_has_tag` matches scalar
numbers for the no-index fallback.

When the first node pattern has an inline literal property and a matching tag
index exists for the label and property, `execute_match` resolves the anchor
with `tag_search` instead of scanning the label. Only integer and string
literals are used, because their text representation is unambiguous. The
resulting id set joins the existing `CandidateConstraints` selection, so the
smallest of vector, text, tag, and inline-anchor candidate sets wins.

This is opt-in: without `CREATE TAG INDEX ON :Person(id)`, the SQL prefilter path
remains. The benchmark shows both.

## Risks

- **Tag text representation**: floats can format differently in Rust and
  SQLite. The planner therefore uses tag anchors only for integers and strings.
  Direct `tag_search` still accepts any tag string.
- **Path explosion**: relationship uniqueness guarantees termination on finite
  graphs but not polynomial time. Dense unbounded patterns can still produce
  huge result sets.
- **No property indexes by default**: a label scan plus SQL prefilter is linear.
  Applications with selective anchor lookups should create a tag index on the
  anchor property.
- **Fixed path duplicates**: a two-hop pattern can return the same endpoint
  twice via different paths. This is intentional and matches path enumeration,
  but callers expecting distinct entities must deduplicate in the application
  until `DISTINCT` exists.

## Migration

No schema migration is required. Existing graphs gain the edge indexes on the
next `ensure_schema`. Existing tag indexes continue to work; rebuilding one adds
numeric values that were previously omitted.
