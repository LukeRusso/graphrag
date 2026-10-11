# Copyright (C) 2026 luke russo -- ruff complaining

"""Shared helpers for graph clustering tests."""

from collections import defaultdict

import pandas as pd
from graphrag.config.models.cluster_graph_config import (
    ClusterGraphConfig,
    LeidenClusterGraphConfig,
    VDStarClusterGraphConfig,
)
from graphrag.config.models.graph_rag_config import GraphRagConfig
from graphrag.data_model.schemas import (
    ENTITIES_FINAL_COLUMNS,
    RELATIONSHIPS_FINAL_COLUMNS,
)
from graphrag.graphs.types import Cluster, Edge
from graphrag_storage import StorageConfig


def vdstar_config(**overrides) -> VDStarClusterGraphConfig:
    """A deterministic VD-Star config with a single small epsilon level."""
    return VDStarClusterGraphConfig(**{
        "epsilon_levels": [0.5],
        "use_lcc": False,
        "seed": 42,
        **overrides,
    })


def leiden_config(**overrides) -> LeidenClusterGraphConfig:
    """A deterministic Leiden config."""
    return LeidenClusterGraphConfig(**{
        "max_cluster_size": 10,
        "use_lcc": False,
        "seed": 42,
        **overrides,
    })


def cfg(
    cluster_graph: ClusterGraphConfig,
    update_output_storage: StorageConfig | None = None,
) -> GraphRagConfig:
    """Wrap a clustering config in the top-level pipeline config."""
    return GraphRagConfig(
        cluster_graph=cluster_graph,
        update_output_storage=update_output_storage or {"type": "memory"},
    )


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


def node_sets(clusters: list[Cluster]) -> list[set[str]]:
    """The node set of every cluster, in order."""
    return [set(cluster.nodes) for cluster in clusters]


def entities(*titles: str) -> pd.DataFrame:
    """An entities frame, one row per title, with the id derived from it."""
    return pd.DataFrame(
        [
            {
                "id": f"e{title}",
                "human_readable_id": i,
                "title": title,
                "type": "THING",
                "description": f"{title} is a thing.",
                "text_unit_ids": [],
                "frequency": 1,
                "degree": 1,
            }
            for i, title in enumerate(titles)
        ],
        columns=ENTITIES_FINAL_COLUMNS,
    )


def relationships(*edges: tuple[str, str] | tuple[str, str, str]) -> pd.DataFrame:
    """A relationships frame, each edge optionally naming its text unit."""
    return pd.DataFrame(
        [
            {
                "id": f"r{i}",
                "human_readable_id": i,
                "source": edge[0],
                "target": edge[1],
                "description": f"{edge[0]} relates to {edge[1]}.",
                "weight": 1.0,
                "combined_degree": 2,
                "text_unit_ids": [edge[2]] if len(edge) == 3 else [],
            }
            for i, edge in enumerate(edges)
        ],
        columns=RELATIONSHIPS_FINAL_COLUMNS,
    )
