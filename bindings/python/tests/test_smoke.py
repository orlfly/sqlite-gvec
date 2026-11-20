from gvec import VERSION, Database, Distance, Vector


def smoke():
    assert VERSION.startswith("v"), VERSION
    db = Database(":memory:")
    g = db.graph("test")

    g.run('CREATE (a:Person {name: "Alice", age: 30})')
    g.run('CREATE (b:Person {name: "Bob", age: 25})')
    rows = g.run(
        "MATCH (n:Person) WHERE n.age > 20 RETURN n.name AS name ORDER BY n.age ASC"
    )
    assert rows == [["Bob"], ["Alice"]], rows

    # Brute-force KNN API parity, now executed by the Rust engine.
    q = Vector((0.0, 0.0))
    stored = [
        (1, Vector((1.0, 0.0))),
        (2, Vector((3.0, 0.0))),
        (3, Vector((0.0, 0.0))),
    ]
    res = g.knn(q, stored, k=2)
    assert [r[0] for r in res] == [3, 1], res
    assert Distance.L2 == "l2"

    # Persistent ANN index over node properties.
    docs = db.graph("docs")
    docs.run('CREATE (a:Doc {title: "origin", body: "the rust programming language", tags: ["database", "rust"], embedding: [0.0, 0.0]})')
    docs.run('CREATE (b:Doc {title: "near", body: "python is dynamic", tags: ["python", "web"], embedding: [1.0, 0.0]})')
    docs.run('CREATE (c:Doc {title: "far", body: "unrelated notes", tags: ["notes"], embedding: [5.0, 5.0]})')
    meta = docs.create_vector_index(
        "Doc", "embedding", "hnsw", "l2", {"m": 8, "ef_search": 32}
    )
    assert meta["algorithm"] == "hnsw" and meta["dim"] == 2, meta
    assert docs.vector_count("Doc", "embedding") == 3
    assert len(docs.list_vector_indexes()) == 1

    hits = docs.vector_search("Doc", "embedding", Vector((0.0, 0.0)), 2)
    assert [rowid for rowid, _ in hits] == [1, 2], hits

    rows = docs.run(
        "MATCH (n:Doc) WHERE vector_nn(n.embedding, [0.0, 0.0], 2) "
        "RETURN n.title AS title, vector_distance(n) AS distance"
    )
    assert rows[0][0] == "origin" and rows[1][0] == "near", rows

    # Full-text search and hybrid retrieval.
    docs.create_text_index("Doc", "body")
    text_hits = docs.text_search("Doc", "body", "rust", 5)
    assert len(text_hits) == 1, text_hits
    rows = docs.run(
        'MATCH (n:Doc) WHERE text_match(n.body, "rust") '
        "AND vector_nn(n.embedding, [1.0, 0.0], 3) "
        "RETURN n.title AS title, text_score(n) AS score, vector_distance(n) AS distance"
    )
    assert [r[0] for r in rows] == ["origin"], rows
    assert rows[0][1] > 0.0 and rows[0][2] >= 0.0, rows

    # Exact tag index and tag-aware hybrid retrieval.
    tag_meta = docs.create_tag_index("Doc", "tags")
    assert tag_meta["label"] == "Doc" and tag_meta["property"] == "tags", tag_meta
    assert len(docs.list_tag_indexes()) == 1
    assert docs.tag_count("Doc", "tags") == 3
    assert docs.tag_search("Doc", "tags", "database") == [1]
    rows = docs.run(
        'MATCH (n:Doc) WHERE tag_match(n.tags, "database") '
        'AND text_match(n.body, "rust") RETURN n.title AS title'
    )
    assert rows == [["origin"]], rows

    print("python smoke OK")


if __name__ == "__main__":
    smoke()
