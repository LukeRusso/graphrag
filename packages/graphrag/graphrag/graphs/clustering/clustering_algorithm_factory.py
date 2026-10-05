# Copyright (C) 2026 luke russo -- ruff complaining

"""Factory for instantiating graph clustering algorithms from config."""

from graphrag_storage import create_storage

from graphrag.config.models.cluster_graph_config import (
    LeidenClusterGraphConfig,
    VDStarClusterGraphConfig,
)
from graphrag.config.models.graph_rag_config import GraphRagConfig
from graphrag.graphs.clustering.base_clustering_algorithm import IClusteringAlgorithm
from graphrag.graphs.hierarchical_leiden import HierarchicalLeiden
from graphrag.graphs.hierarchical_vdstar import HierarchicalVDStar
from graphrag.graphs.vdstar_state import VDSTAR_STATE_NAMESPACE

UNHANDLED_TYPE_MSG = (
    "Unhandled graph clustering algorithm type during algorithm object instantiation"
)


class ClusteringAlgorithmFactory:
    """Class responsible for instanting objects that execute graph clustering algorithms."""

    @staticmethod
    def from_config(
        config: GraphRagConfig,
    ) -> IClusteringAlgorithm:
        """Instantiate an IClusteringAlgorithm object for graph clustering from a user provided config."""
        match config.cluster_graph:
            case LeidenClusterGraphConfig() as leiden_config:
                return HierarchicalLeiden(leiden_config)
            case VDStarClusterGraphConfig() as vdstar_config:
                state_storage = create_storage(config.update_output_storage).child(
                    VDSTAR_STATE_NAMESPACE
                )
                return HierarchicalVDStar(vdstar_config, state_storage)
            case _:
                raise ValueError(UNHANDLED_TYPE_MSG)
