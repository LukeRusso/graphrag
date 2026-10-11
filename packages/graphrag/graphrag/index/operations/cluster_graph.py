# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A module containing cluster_graph method definition."""

import logging

import pandas as pd

from graphrag.config.enums import CommunityUpdateMode
from graphrag.config.models.graph_rag_config import GraphRagConfig
from graphrag.graphs.clustering.base_clustering_algorithm import (
    IClusteringAlgorithm,
    IIncrementalClusteringAlgorithm,
)
from graphrag.graphs.clustering.clustering_algorithm_factory import (
    ClusteringAlgorithmFactory,
)
from graphrag.graphs.stable_lcc import stable_lcc
from graphrag.graphs.types import Cluster, Edge
from graphrag.index.update.relationships import GraphDelta

Communities = list[Cluster]

logger = logging.getLogger(__name__)

UNHANDLED_UPDATE_MODE_MSG = "Unhandled community update mode"


async def cluster_graph(
    edges: pd.DataFrame,
    config: GraphRagConfig,
    delta: GraphDelta | None = None,
) -> Communities:
    """Apply a hierarchical clustering algorithm to a relationships DataFrame.

    A build run passes no delta and clusters the whole supplied graph. An update
    run passes the delta it is applying and is handled in different ways depending on update strategy.
    """
    if delta is None:
        return await _cluster_edges(edges, config)

    return await _cluster_update(edges, config, delta)


async def _cluster_update(
    edges: pd.DataFrame, config: GraphRagConfig, delta: GraphDelta
) -> Communities:
    """Cluster the supplied graph according to the configured update mode."""
    match config.cluster_graph.update_mode:
        case CommunityUpdateMode.Incremental:
            return await _cluster_incrementally(config, delta)
        case CommunityUpdateMode.Additive:
            return await _cluster_edges(
                _new_entity_edges(edges, delta.added_vertices), config
            )
        case _:
            error_msg = (
                f"{UNHANDLED_UPDATE_MODE_MSG}: {config.cluster_graph.update_mode}"
            )
            raise ValueError(error_msg)


async def _cluster_incrementally(
    config: GraphRagConfig, delta: GraphDelta
) -> Communities:
    cluster_algo = ClusteringAlgorithmFactory.from_config(config)

    if not isinstance(cluster_algo, IIncrementalClusteringAlgorithm):
        error_msg = (
            f"Cannot cluster incrementally with {type(cluster_algo).__name__}: it "
            "does not implement IIncrementalClusteringAlgorithm."
        )
        raise TypeError(error_msg)

    if not await cluster_algo.has_state():
        error_msg = (
            "Cannot cluster incrementally: no persisted clustering state was "
            "found from a previous run."
        )
        raise RuntimeError(error_msg)

    return await cluster_algo.cluster_incremental(
        delta.added_edges, delta.removed_edges
    )


async def _cluster_edges(edges: pd.DataFrame, config: GraphRagConfig) -> Communities:
    if edges.empty:
        return []

    edge_df = _normalize_edges(edges)
    if config.cluster_graph.use_lcc:
        edge_df = stable_lcc(edge_df)
    edge_list = _df_to_edge_list(edge_df)

    cluster_algo: IClusteringAlgorithm = ClusteringAlgorithmFactory.from_config(config)
    return await cluster_algo.cluster(edges=edge_list)


def _new_entity_edges(edges: pd.DataFrame, titles: list[str]) -> pd.DataFrame:
    """Return the supplied edges whose both endpoints are titles."""
    return edges[edges["source"].isin(titles) & edges["target"].isin(titles)]


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

    return [
        Edge(*tup)
        for tup in sorted(
            zip(
                edges["source"].astype(str),
                edges["target"].astype(str),
                weights,
                strict=True,
            )
        )
    ]
