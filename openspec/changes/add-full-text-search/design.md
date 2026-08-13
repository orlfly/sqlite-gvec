## Context

gvec's core engine (`gvec-core`) stores nodes with JSON properties in SQLite shadow tables and provides vector KNN + Cypher-like graph queries. The query executor (`query.rs`) evaluates `WHERE` predicates via an `eval()` function that handles `Expr::Cmp`, `Expr::And`/`Or`/`Not`, and `Expr::VectorNn`. There is currently no text search: property values are compared by exact equality only.

`sqlite-vec` has text metadata columns (`metadatatext%02d` shadow tables) used for `=` / `LIKE` filtering during KNN scans, but offers no tokenization, inverted index, or relevance ranking. This is insufficient for lexical search.

SQLite ships with **FTS5** — a mature full-text search module with BM25 ranking, multiple tokenizers (`unicode61`, `porter`, `trigram`), and phrase/proximity queries. The `rusqlite` bundled build includes FTS5 by default. We leverage FTS5 rather than reimplementing an inverted index in Rust.

## Goals / Non-Goals

**Goals:**
- Index node text properties in FTS5 shadow tables managed by gvec.
- Search via `text_match(n.property, "query")` in the Cypher-like query language.
- BM25 relevance scoring exposed in results.
- Hybrid queries that intersect FTS rowid sets with KNN anchors and graph traversals.
- Multiple tokenizer options per index.
- SQL-level access via `gvec_text_search()`.

**Non-Goals:**
- Reimplementing FTS5 or a custom inverted index in Rust.
- FTS on edge properties (v0.1 — nodes only).
- Custom tokenizers beyond what FTS5 ships with.
- Distributed or replicated FTS.

## Decisions

### D1. FTS5 as shadow tables alongside gvec's existing schema
When a user issues `CREATE TEXT INDEX ON :Person(body) USING fts5(porter)`, gvec creates:
- A shadow FTS5 virtual table: `{prefix}_fts_{label}_{prop}` (e.g., `gvec0_main_fts_Person_body`) with `content_table` pointing at the node shadow table.
- SQLite triggers that keep the FTS5 table in sync on INSERT/UPDATE/DELETE to `gvec0_nodes`.
- Rationale: FTS5 is battle-tested, includes BM25, and ships with SQLite. Reusing it avoids a multi-thousand-line Rust reimplementation and keeps gvec lightweight.
- Alternative considered: Rust-native inverted index (e.g., `tantivy` crate) — rejected for v0.1 because it adds a heavy dependency and duplicates FTS5 capability that SQLite already provides.

### D2. `text_match()` as a special predicate in the planner
`text_match(n.body, "rust database")` evaluates to true for nodes whose `body` property matches the FTS5 query. Unlike a scalar boolean, the planner extracts it as a **rowid set** (the FTS5 `rowid` matches for the query, ranked by BM25). This rowid set constrains the node scan the same way `vector_nn` does — by providing a candidate set.
- Rationale: treating FTS as a rowid-set producer (not just a boolean filter) enables three-way intersection with KNN and graph traversal.
- The `Expr::TextMatch` AST variant carries `(variable, property, query_string)`.

### D3. BM25 score retrieval
When `text_match` is present, the executor joins the FTS5 `bm25()` rank back to result rows. A `score` column is available in RETURN if the user writes `RETURN n.title, text_score(n) AS score`.
- Rationale: BM25 is the standard relevance metric; exposing it lets users sort by text relevance and combine it with vector distance for hybrid ranking.

### D4. Three-way fusion
When a MATCH contains both `text_match` and `vector_nn`, the planner:
1. Collects the FTS5 rowid set (ranked by BM25).
2. Collects the KNN rowid set (ranked by distance).
3. Intersects the two sets.
4. Feeds the intersection as anchors into graph traversal.
5. Default ordering: reciprocal-rank fusion (`1/(60+text_rank) + 1/(60+vector_rank)`) unless `ORDER BY` overrides.
- Rationale: reciprocal-rank fusion is the standard hybrid-search combination method; it needs no score normalization between BM25 and L2.

### D5. Tokenizer selection
The `USING fts5(tokenizer)` clause maps to FTS5's `tokenize=` option. Supported values: `unicode61` (default), `porter`, `trigram`. Unknown tokenizers are rejected with an error.

### D6. Sync strategy: triggers vs. explicit reindex
Use SQLite triggers (`AFTER INSERT/UPDATE/DELETE ON gvec0_nodes`) to keep FTS5 in sync automatically. A `REINDEX TEXT ON :Label(property)` statement supports manual rebuilds.
- Rationale: triggers are ACID with the main transaction; explicit reindex handles schema changes.

## Risks / Trade-offs

- [FTS5 not available in all SQLite builds] → Detect at `CREATE TEXT INDEX` time and error clearly; document that `rusqlite` bundled includes FTS5.
- [Trigger overhead on every node write] → Triggers only fire on tables that have an FTS index; tables without FTS incur zero overhead.
- [FTS5 external-content vs. standalone content] → Use standalone content tables (FTS5 stores its own copy) for simplicity; document the storage overhead.
- [Large result sets from FTS] → Default `text_match` to top-50 BM25 results; users can override with an explicit `k` argument.