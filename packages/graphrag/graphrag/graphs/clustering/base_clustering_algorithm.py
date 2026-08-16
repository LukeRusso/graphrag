from abc import ABC, abstractmethod

from graphrag.config.models.cluster_graph_config import ClusterGraphConfig
from graphrag.graphs.type_aliases import Edge

from .cluster import Cluster


class IClusteringAlgorithm(ABC):
    """Interface for hierarchical clustering algorithm."""

    @abstractmethod
    def cluster(self, edges: list[Edge], config: ClusterGraphConfig) -> list[Cluster]:
        """Perform hierarchical clustering on input graph."""
