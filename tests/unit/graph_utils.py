# Copyright (C) 2026 luke russo -- ruff complaining

"""Shared helpers for graph clustering tests."""

from collections import defaultdict

from graphrag.graphs.types import Cluster, Edge


def triangle(name: str) -> list[Edge]:
    """A three-node clique, with an optional prefix on each node id."""
    a, b, c = (f"{name}{suffix}" for suffix in ("A", "B", "C"))
    return [Edge(a, b, 1.0), Edge(a, c, 1.0), Edge(b, c, 1.0)]


def tiered_graph() -> list[Edge]:
    """A dense core with pendant nodes, so the levels differ in cluster size."""
    names = {i: f"N{i}" for i in range(1, 11)}
    core = [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]
    pendant = [(1, i) for i in range(5, 11)]
    tail = [(5, 6), (6, 7)]
    return [Edge(names[u], names[v], 1.0) for u, v in [*core, *pendant, *tail]]


def by_level(clusters: list[Cluster]) -> dict[int, list[Cluster]]:
    """Group clusters by their level."""
    grouped: dict[int, list[Cluster]] = defaultdict(list)
    for cluster in clusters:
        grouped[cluster.level].append(cluster)
    return grouped
