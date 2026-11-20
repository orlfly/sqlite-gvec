## 1. Tag index core (`gvec-core/src/tag.rs`)

- [x] 1.1 Add `TagIndexMeta` and shadow tables `{prefix}_tag_indexes` and
  `{prefix}_tags` with lookup indexes
- [x] 1.2 Implement `create_tag_index`, `drop_tag_index`, `list_tag_indexes`,
  `find_tag_index`, `tag_count`, and `tag_search_ids`
- [x] 1.3 Backfill string and array properties through SQLite JSON1
  (`json_each` plus `json_quote` for scalar strings)
- [x] 1.4 Add insert/update/delete triggers, including label changes
- [x] 1.5 Add unit tests for string/array/numeric exact matching and case
  sensitivity
- [x] 1.6 Add `Storage` wrappers and an exact fallback scan when no index exists

## 2. Query language

- [x] 2.1 Add `Expr::TagMatch` and `Statement::CreateTagIndex` /
  `DropTagIndex`
- [x] 2.2 Parse `tag_match(var.property, "tag")`, `CREATE TAG INDEX ON
  :Label(property)`, and `DROP TAG INDEX ON :Label(property)`
- [x] 2.3 Extract top-level conjunctive tag predicates, intersect multiple
  tags on the same variable, and reject OR/NOT/comparison positions
- [x] 2.4 Integrate tag rowid sets with `CandidateConstraints` and fetch the
  smallest matching node set during graph scans
- [x] 2.5 Keep row-level filtering as a final correctness guard
- [x] 2.6 Add parser and executor tests, including text+tag and vector+tag

## 3. Bindings

- [x] 3.1 Expose `Graph::create_tag_index`, `drop_tag_index`,
  `list_tag_indexes`, `tag_count`, and `tag_search` in the Rust binding
- [x] 3.2 Expose the same lifecycle and search surface in the PyO3 binding
- [x] 3.3 Update the Python smoke test with tag index and hybrid retrieval

## 4. Tests and benchmark

- [x] 4.1 E2E: create/search, exactness, multiple tags, string and array values
- [x] 4.2 E2E: insert/update/delete/label synchronization
- [x] 4.3 E2E: text+tag, vector+tag, drop and fallback, reopen persistence
- [x] 4.4 E2E: reject OR/NOT and mixed-variable tag predicates
- [x] 4.5 Add deterministic `tag_bench` with naive-scan verification and
  indexed-vs-fallback latency

## 5. Docs and finalize

- [x] 5.1 README section with syntax, semantics, limitations, and benchmark
- [x] 5.2 Openspec proposal/design/spec
- [x] 5.3 Run fmt, clippy, full workspace tests, and Python build
- [x] 5.4 Commit the tag index work
