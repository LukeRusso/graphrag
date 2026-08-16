from graphrag.config.models.cluster_graph_config import (
    ClusterGraphConfig,
    LeidenClusterGraphConfig,
)
from graphrag.graphs.clustering.base_clustering_algorithm import IClusteringAlgorithm
from graphrag.graphs.hierarchical_leiden import HierarchicalLeiden

UNHANDLED_TYPE_MSG = (
    "Unhandled graph clustering algorithm type during algorithm object instantiation"
)


class ClusteringAlgorithmFactory:
    """Class responsible for instanting objects that execute graph clustering algorithms."""

    @staticmethod
    def from_config(
        config: ClusterGraphConfig,
    ) -> IClusteringAlgorithm:
        """Instantiate an IClusteringAlgorithm object for graph clustering from a user provided config."""
        match config:
            case LeidenClusterGraphConfig() as leiden_config:
                return HierarchicalLeiden(leiden_config)
            case _:
                raise ValueError(UNHANDLED_TYPE_MSG)
