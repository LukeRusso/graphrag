# Copyright (C) 2026 luke russo -- ruff complaining

import pandas as pd
from graphrag.graphs.types import Edge
from graphrag.index.update.relationships import compute_graph_delta


def _relationships(*edges: tuple[str, str, float]) -> pd.DataFrame:
    return pd.DataFrame([
        {"source": source, "target": target, "weight": weight}
        for source, target, weight in edges
    ])


def _entities(*titles: str) -> pd.DataFrame:
    return pd.DataFrame([{"title": title} for title in titles])


def test_identical_graphs_produce_an_empty_delta():
    old = _relationships(("A", "B", 1.0), ("B", "C", 2.0))
    new = _relationships(("B", "C", 2.0), ("A", "B", 1.0))

    delta = compute_graph_delta(
        old, new, _entities("A", "B", "C"), _entities("A", "B", "C")
    )

    assert delta.added_edges == []
    assert delta.removed_edges == []
    assert delta.added_vertices == []
    assert delta.removed_vertices == []


def test_an_edge_in_the_new_graph_only_is_added():
    old = _relationships(("A", "B", 1.0))
    new = _relationships(("A", "B", 1.0), ("B", "C", 1.0))
    entities = _entities("A", "B", "C")

    delta = compute_graph_delta(old, new, entities, entities)

    assert delta.added_edges == [Edge("B", "C", 1.0)]
    assert delta.removed_edges == []


def test_an_edge_in_the_old_graph_only_is_removed():
    old = _relationships(("A", "B", 1.0), ("B", "C", 5.0))
    new = _relationships(("A", "B", 1.0))
    entities = _entities("A", "B", "C")

    delta = compute_graph_delta(old, new, entities, entities)

    assert delta.added_edges == []
    assert delta.removed_edges == [Edge("B", "C", 5.0)]


def test_a_reversed_edge_is_not_a_change():
    old = _relationships(("B", "A", 1.0))
    new = _relationships(("A", "B", 1.0))
    entities = _entities("A", "B")

    delta = compute_graph_delta(old, new, entities, entities)

    assert delta.added_edges == []
    assert delta.removed_edges == []


def test_a_duplicated_pair_is_not_a_change():
    old = _relationships(("A", "B", 1.0))
    new = _relationships(("A", "B", 1.0), ("B", "A", 1.0), ("A", "B", 1.0))
    entities = _entities("A", "B")

    delta = compute_graph_delta(old, new, entities, entities)

    assert delta.added_edges == []
    assert delta.removed_edges == []


def test_an_edgeless_vertex_in_the_new_graph_only_is_added():
    relationships = _relationships(("A", "B", 1.0))

    delta = compute_graph_delta(
        relationships, relationships, _entities("A", "B"), _entities("A", "B", "LONE")
    )

    assert delta.added_vertices == ["LONE"]
    assert delta.added_edges == []


def test_an_edgeless_vertex_in_the_old_graph_only_is_removed():
    relationships = _relationships(("A", "B", 1.0))

    delta = compute_graph_delta(
        relationships, relationships, _entities("A", "B", "LONE"), _entities("A", "B")
    )

    assert delta.removed_vertices == ["LONE"]
    assert delta.removed_edges == []
