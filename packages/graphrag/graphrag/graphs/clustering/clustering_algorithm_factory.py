# Copyright (C) 2026 luke russo -- ruff complaining

"""Factory for instantiating graph clustering algorithms from config."""

from graphrag.config.models.cluster_graph_config import (
    ClusterGraphConfig,
    LeidenClusterGraphConfig,
    VDStarClusterGraphConfig,
)
from graphrag.graphs.clustering.base_clustering_algorithm import IClusteringAlgorithm
from graphrag.graphs.hierarchical_leiden import HierarchicalLeiden
from graphrag.graphs.hierarchical_vdstar import HierarchicalVDStar

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
            case VDStarClusterGraphConfig() as vdstar_config:
                return HierarchicalVDStar(vdstar_config)
            case _:
                raise ValueError(UNHANDLED_TYPE_MSG)
