# Copyright (c) 2024 Luke Russo.
# Licensed under the MIT License

"""Interfaces for hierarchical graph clustering algorithms."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from graphrag.graphs.types import Cluster, Edge


@dataclass
class IClusteringAlgorithm[ClusterGraphConfig](ABC):
    """Interface for hierarchical clustering algorithm."""

    config: ClusterGraphConfig

    @abstractmethod
    async def cluster(self, edges: list[Edge]) -> list[Cluster]:
        """Perform hierarchical clustering on input graph."""


@dataclass
class IIncrementalClusteringAlgorithm[ClusterGraphConfig](
    IClusteringAlgorithm[ClusterGraphConfig], ABC
):
    """Clustering algorithms with a persisted engine that can absorb deltas."""

    @abstractmethod
    async def cluster_incremental(
        self,
        edges_added: list[Edge],
        edges_removed: list[Edge],
    ) -> list[Cluster]:
        """Update the clustering given the provided graph changes."""
