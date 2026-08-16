from typing import NamedTuple


class Edge(NamedTuple):
    """Represents one edge in a graph."""

    source: str
    dest: str
    weight: float


class Cluster(NamedTuple):
    """Represents one cluster in a hierarchical graph clustering output."""

    level: int
    cluster_id: int
    parent_cluster_id: int
    nodes: list[str]
