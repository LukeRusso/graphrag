from abc import ABC, abstractmethod
from dataclasses import dataclass

from graphrag.graphs.types import Cluster, Edge


@dataclass
class IClusteringAlgorithm[ClusterGraphConfig](ABC):
    """Interface for hierarchical clustering algorithm."""

    config: ClusterGraphConfig

    @abstractmethod
    def cluster(self, edges: list[Edge]) -> list[Cluster]:
        """Perform hierarchical clustering on input graph."""
