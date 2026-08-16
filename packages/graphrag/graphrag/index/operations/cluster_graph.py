# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A module containing cluster_graph method definition."""

import logging
from collections import defaultdict

import pandas as pd

from graphrag.graphs.hierarchical_leiden import hierarchical_leiden
from graphrag.graphs.stable_lcc import stable_lcc
from packages.graphrag.graphrag.graphs.type_aliases import Edge

Communities = list[tuple[int, int, int, list[str]]]


logger = logging.getLogger(__name__)


def cluster_graph(
    edges: pd.DataFrame,
    max_cluster_size: int,
    use_lcc: bool,
    seed: int | None = None,
) -> Communities:
    """Apply a hierarchical clustering algorithm to a relationships DataFrame."""
    edge_df = _normalize_edges(edges)
    if use_lcc:
        edge_df = stable_lcc(edge_df)
    edge_list = _df_to_edge_list(edge_df)

    node_id_to_community_map, parent_mapping = _compute_leiden_communities(
        edge_list=edge_list,
        max_cluster_size=max_cluster_size,
        seed=seed,
    )

    levels = sorted(node_id_to_community_map.keys())

    clusters: dict[int, dict[int, list[str]]] = {}
    for level in levels:
        result: dict[int, list[str]] = defaultdict(list)
        clusters[level] = result
        for node_id, community_id in node_id_to_community_map[level].items():
            result[community_id].append(node_id)

    results: Communities = []
    for level in clusters:
        for cluster_id, nodes in clusters[level].items():
            results.append((level, cluster_id, parent_mapping[cluster_id], nodes))
    return results


def _normalize_edges(edges: pd.DataFrame) -> pd.DataFrame:
    edge_df = edges.copy()

    # Normalize edge direction and deduplicate (undirected graph).
    # NX deduplicates reversed pairs keeping the last row's attributes,
    # so we replicate that by normalizing direction then keeping last.
    lo = edge_df[["source", "target"]].min(axis=1)
    hi = edge_df[["source", "target"]].max(axis=1)
    edge_df["source"] = lo
    edge_df["target"] = hi
    edge_df.drop_duplicates(subset=["source", "target"], keep="last", inplace=True)

    return edge_df


def _df_to_edge_list(edges: pd.DataFrame) -> list[Edge]:
    weights = (
        edges["weight"].astype(float)
        if "weight" in edges.columns
        else pd.Series(1.0, index=edges.index)
    )

    return sorted(
        zip(
            edges["source"].astype(str),
            edges["target"].astype(str),
            weights,
            strict=True,
        )
    )


# Taken from graph_intelligence & adapted
def _compute_leiden_communities(
    edge_list: list[Edge],
    max_cluster_size: int,
    seed: int | None = None,
) -> tuple[dict[int, dict[str, int]], dict[int, int]]:
    """Return Leiden root communities and their hierarchy mapping."""
    community_mapping = hierarchical_leiden(
        edge_list, max_cluster_size=max_cluster_size, random_seed=seed
    )
    results: dict[int, dict[str, int]] = {}
    hierarchy: dict[int, int] = {}
    for partition in community_mapping:
        results[partition.level] = results.get(partition.level, {})
        results[partition.level][partition.node] = partition.cluster

        hierarchy[partition.cluster] = (
            partition.parent_cluster if partition.parent_cluster is not None else -1
        )

    return results, hierarchy
