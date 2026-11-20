# gvec Python bindings

The Python bindings are implemented in Rust with PyO3. The Python package is
only a thin import shim; graph storage, vector indexes, and FTS all run
in-process through the `gvec` Rust crate.

## Build

Using `maturin` (recommended):

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install maturin
maturin develop --release
```

Or build the extension manually and place it in the package:

```bash
cargo build --release --manifest-path crates/gvec-python/Cargo.toml
cp crates/gvec-python/target/release/lib_gvec.so \
   "bindings/python/gvec/_gvec$(python3-config --extension-suffix)"
```

## Usage

```python
from gvec import Database, Vector, Distance

db = Database(":memory:")
g = db.graph("social")

g.run('CREATE (a:Person {name: "Alice", age: 30})')
g.run('CREATE (b:Person {name: "Bob", age: 25})')
rows = g.run("MATCH (n:Person) WHERE n.age > 20 RETURN n.name AS name")
print(rows)

# Persistent ANN index
g.run('CREATE (d:Doc {title: "hello", body: "hello vector world", embedding: [0.1, 0.2]})')
g.create_vector_index("Doc", "embedding", "hnsw", "l2", {"m": 16})
print(g.vector_search("Doc", "embedding", Vector((0.1, 0.2)), 5))

# Hybrid text + vector query
g.create_text_index("Doc", "body")
print(g.run('''
    MATCH (n:Doc)
    WHERE text_match(n.body, "hello") AND vector_nn(n.embedding, [0.1, 0.2], 5)
    RETURN n.title, text_score(n), vector_distance(n)
'''))

# Exact tag index and tag-aware hybrid query
g.run('CREATE (t:Doc {title: "tagged", body: "hello rust", tags: ["database", "rust"]})')
print(g.create_tag_index("Doc", "tags"))
print(g.tag_search("Doc", "tags", "database"))
print(g.run('''
    MATCH (n:Doc)
    WHERE tag_match(n.tags, "database") AND text_match(n.body, "hello")
    RETURN n.title
'''))
```

## Smoke test

```bash
# After `maturin develop` or installing the wheel:
python bindings/python/tests/test_smoke.py

# After a manual `cargo build` + copy into the source package:
PYTHONPATH=bindings/python python bindings/python/tests/test_smoke.py
```
