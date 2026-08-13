from gvec import Database, Vector, Distance

def smoke():
    db = Database(":memory:")
    g = db.graph("test")
    g.run('CREATE (a:Person {name: "Alice", age: 30})')
    g.run('CREATE (b:Person {name: "Bob", age: 25})')
    rows = g.run("MATCH (n:Person) WHERE n.age > 20 RETURN n.name AS name ORDER BY n.age ASC")
    assert len(rows) == 2, f"expected 2 rows, got {len(rows)}"
    assert rows[0][0] == "Bob", f"expected Bob first, got {rows[0]}"
    assert rows[1][0] == "Alice", f"expected Alice second, got {rows[1]}"
    # KNN parity
    q = Vector((0.0, 0.0))
    stored = [
        (1, Vector((1.0, 0.0))),
        (2, Vector((3.0, 0.0))),
        (3, Vector((0.0, 0.0))),
    ]
    res = g.knn(q, stored, k=2)
    assert res[0][0] == 3
    assert res[1][0] == 1
    print("python smoke OK")

if __name__ == "__main__":
    smoke()
