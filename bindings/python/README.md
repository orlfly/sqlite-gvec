# gvec — Python bindings

```python
from gvec import Database, Vector, Distance

db = Database(":memory:")  # requires SQLite built with --enable-load-extension
g = db.graph("social")

g.run('CREATE (a:Person {name: "Alice", age: 30})')
g.run('CREATE (b:Person {name: "Bob", age: 25})')
g.run('CREATE (a)-[:KNOWS]->(b)')

rows = g.run("MATCH (n:Person) WHERE n.age > 20 RETURN n.name AS name")
print(rows)

q = Vector((0.0, 0.0))
stored = [(1, Vector((1.0, 0.0))), (2, Vector((3.0, 0.0)))]
print(g.knn(q, stored, k=2))
```

## Smoke test

```bash
PYTHONPATH=bindings/python python3 bindings/python/test_smoke.py
```