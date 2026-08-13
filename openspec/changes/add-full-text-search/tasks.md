## 1. FTS5 Module (`gvec-core/src/fts.rs`)

- [x] 1.1 Create `fts.rs` module with `FtsIndex` struct: manages FTS5 virtual table name, tokenizer, and shadow-table lifecycle
- [x] 1.2 Implement `create_text_index(exec, prefix, label, property, tokenizer)`: create FTS5 virtual table `CREATE VIRTUAL TABLE {prefix}_fts_{label}_{prop} USING fts5(content, tokenize={tokenizer})`
- [x] 1.3 Implement `drop_text_index(exec, prefix, label, property)`: drop FTS5 table and triggers
- [x] 1.4 Implement sync triggers: `AFTER INSERT/UPDATE/DELETE ON {prefix}_nodes` that keep the FTS5 table in sync for matching-label rows
- [x] 1.5 Implement `text_search(exec, prefix, label, property, query, k) -> Vec<(i64, f32)>`: run FTS5 query with `bm25()` ranking, return top-k `(rowid, score)` pairs
- [x] 1.6 Detect FTS5 availability at runtime (try `CREATE VIRTUAL TABLE ... USING fts5(...)` on a temp table; cache result)
- [x] 1.7 Add unit tests: create index, insert nodes, search, BM25 ordering, delete sync
- [x] 1.8 Make `create_text_index` idempotent: switch the backfill `INSERT INTO` to `INSERT OR REPLACE INTO` so re-running for an already-indexed `(label, property)` tuple replaces existing rows instead of failing with a rowid UNIQUE conflict. This makes both the Rust binding and the `CREATE TEXT INDEX` Cypher statement idempotent at the user-visible level.

## 2. Query Language Integration (`graph-query-language` modifications)

- [x] 2.1 Add `Expr::TextMatch { variable, property, query }` and `Expr::TextScore { variable }` AST variants
- [x] 2.2 Add `text_match(var.prop, "query")` and `text_score(var)` to the parser (primary-expression level, like `vector_nn`)
- [x] 2.3 Add `Statement::CreateTextIndex { label, property, tokenizer }` and `Statement::DropTextIndex { label, property }` AST variants
- [x] 2.4 Parse `CREATE TEXT INDEX ON :Label(property) [USING fts5(tokenizer)]` and `DROP TEXT INDEX ON :Label(property)` as top-level statements
- [x] 2.5 Execute `CreateTextIndex` / `DropTextIndex` by delegating to `fts.rs`
- [x] 2.6 Evaluate `Expr::TextMatch` in `eval()` as `Value::Bool(true/false)` for fallback boolean filtering
- [x] 2.7 Evaluate `Expr::TextScore` in `eval()` by looking up the cached BM25 score for the current row
- [x] 2.8 Add parser tests: parse `text_match`, `text_score`, `CREATE TEXT INDEX`, `DROP TEXT INDEX`

## 3. Planner: FTS as Rowid-Set Constraint

- [x] 3.1 In `execute_match`, detect `Expr::TextMatch` in the WHERE clause before generic evaluation
- [x] 3.2 Call `fts::text_search()` to get the ranked rowid set; intersect with the node-scan candidates
- [x] 3.3 Cache BM25 scores per rowid in `ExecState` so `text_score()` can retrieve them during RETURN
- [x] 3.4 Add planner tests: text_match constrains scan, text_score returns BM25, no-index error

## 4. Three-Way Fusion (`graph-vector-integration` modifications)

- [x] 4.1 When both `text_match` and `vector_nn` are present, collect both rowid sets before scanning
- [x] 4.2 Intersect the sets; compute reciprocal-rank fusion score per rowid
- [x] 4.3 Default-order by RRF score when no explicit `ORDER BY`
- [x] 4.4 Expose `vector_distance(variable)` expression for hybrid result inspection
- [x] 4.5 Add fusion tests: text+vector intersection, RRF ordering, three-way with graph traversal

## 5. Bindings & Extension Surface

- [x] 5.1 Add `Graph::create_text_index(label, property, tokenizer?)` and `Graph::drop_text_index(label, property)` to the Rust binding
- [x] 5.2 Add `Graph::text_search(label, property, query, k) -> Vec<(i64, f32)>` to the Rust binding
- [x] 5.3 Register `gvec_text_search(graph, label, property, query, k)` scalar in the extension crate
- [x] 5.4 Add binding tests: create index via Rust API, text_search, hybrid query end-to-end
- [x] 5.5 Add binding tests covering idempotency: `CREATE TEXT INDEX` called twice via Cypher must not fail with a rowid UNIQUE conflict; Rust `Graph::create_text_index` called twice (with same or different tokenizers) must leave the FTS5 table in a consistent, queryable state; `DROP TEXT INDEX` followed by `CREATE TEXT INDEX` must rebuild the index from scratch.

## 6. Docs & Finalize

- [x] 6.1 Update README with FTS quick-start and hybrid-search example
- [x] 6.2 Verify every spec scenario maps to at least one automated test
- [x] 6.3 Run `cargo test --workspace` clean; verify clippy/fmt