# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A module containing cluster_graph method definition."""

import logging
from typing import TYPE_CHECKING

import pandas as pd

from graphrag.config.models.cluster_graph_config import (
    ClusterGraphConfig,
)
from graphrag.graphs.clustering.clustering_algorithm_factory import (
    ClusteringAlgorithmFactory,
)
from graphrag.graphs.stable_lcc import stable_lcc
from graphrag.graphs.types import Cluster, Edge

if TYPE_CHECKING:
    from graphrag.graphs.clustering.base_clustering_algorithm import (
        IClusteringAlgorithm,
    )

Communities = list[Cluster]

logger = logging.getLogger(__name__)


def cluster_graph(edges: pd.DataFrame, config: ClusterGraphConfig) -> Communities:
    """Apply a hierarchical clustering algorithm to a relationships DataFrame."""
    edge_df = _normalize_edges(edges)
    if config.use_lcc:
        edge_df = stable_lcc(edge_df)
    edge_list = _df_to_edge_list(edge_df)

    cluster_algo: IClusteringAlgorithm = ClusteringAlgorithmFactory.from_config(config)
    return cluster_algo.cluster(edges=edge_list)


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
