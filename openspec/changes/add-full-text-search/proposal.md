## Why

gvec currently supports vector search and graph traversal but cannot search text properties by relevance. `sqlite-vec` stores text metadata columns (`metadatatext%02d`) for exact-match filtering on KNN results, but provides no tokenization, inverted index, or relevance ranking. Many real-world use cases need hybrid search — combine semantic (vector), structural (graph), and lexical (full-text) signals in one query. gvec should add full-text search by leveraging SQLite's built-in FTS5 module so users can do `WHERE text_match(n.content, "query terms")` alongside vector and graph predicates.

## What Changes

- Add a new `full-text-search` module to `gvec-core` that manages FTS5 virtual tables as shadow tables alongside existing node/edge tables.
- Introduce a `CREATE TEXT INDEX ON :Label(property)` DDL statement that creates an FTS5 index over a node property.
- Add a `text_match(variable.property, "query")` predicate to the Cypher-like query language, producing BM25-ranked rowid sets that feed into the executor.
- Support hybrid queries: `MATCH (n:Doc) WHERE text_match(n.body, "rust database") AND vector_nn(n.embedding, :q, 10) RETURN n.title, n.score` — the planner intersects FTS hits with KNN anchors.
- Expose `gvec_text_search(graph, property, query, k)` as a scalar SQL function for direct SQL-level FTS access.
- Tokenizer options: `unicode61` (FTS5 default), `porter` (stemming), and `trigram` (for substring matching).

## Capabilities

### New Capabilities

- `full-text-search`: FTS5-backed full-text indexing and search over node text properties, with BM25 ranking and tokenizer selection, integrated into the gvec query language via `text_match()`.

### Modified Capabilities

- `graph-query-language`: add `text_match(variable.property, query)` as a new predicate expression in `WHERE` clauses; add `CREATE TEXT INDEX ON :Label(property)` as a new DDL statement.
- `graph-vector-integration`: extend fusion to three-way intersection of text, vector, and graph constraints in a single MATCH.

## Impact

- New module: `crates/gvec-core/src/fts.rs` — FTS5 shadow-table lifecycle, tokenizer config, BM25 score retrieval.
- Modified: `crates/gvec-core/src/query.rs` — new `Expr::TextMatch` variant, parser support, planner intersection logic.
- Modified: `crates/gvec-core/src/fusion.rs` — three-way fusion (text ∩ vector ∩ graph).
- Modified: `crates/gvec-bindings/src/lib.rs` — expose `Graph::text_search()` and `Graph::create_text_index()`.
- Modified: `crates/gvec-extension/src/lib.rs` — register `gvec_text_search()` scalar.
- Requires SQLite compiled with FTS5 (`SQLITE_ENABLE_FTS5`); the `rusqlite` bundled build includes FTS5 by default.
- No new external crate dependencies; FTS5 is a SQLite built-in module accessed via SQL.