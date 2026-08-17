# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Parameterization settings for the default configuration."""

from typing import Annotated, Literal

from pydantic import BaseModel, Field

from graphrag.config.defaults import graphrag_config_defaults
from graphrag.config.models.graph_clustering_config_types import (
    GraphClusteringAlgorithmType,
)

INSTANTIATION_ERROR = "BaseClusterGraphConfig cannot be instantiated"


class BaseClusterGraphConfig[GraphClusteringAlgorithmType](BaseModel):
    """General shared configuration section for clustering graphs."""

    algorithm: GraphClusteringAlgorithmType = Field(
        ..., description="The unique identifier of the graph clustering algorithm."
    )

    use_lcc: bool = Field(
        description="Whether to use the largest connected component.",
        default=graphrag_config_defaults.cluster_graph.use_lcc,
    )

    def __init__(self, **data):
        if type(self) is BaseClusterGraphConfig:
            raise TypeError(INSTANTIATION_ERROR)

        super().__init__(**data)


class LeidenClusterGraphConfig(BaseClusterGraphConfig):
    """Configuration section for clustering graphs using the Leiden algorithm."""

    algorithm: Literal[GraphClusteringAlgorithmType.LEIDEN] = Field(
        description="Leiden graph clustering algorithm configuration.",
        default=graphrag_config_defaults.cluster_graph.algorithm,
    )

    max_cluster_size: int = Field(
        description="The maximum cluster size to use.",
        default=graphrag_config_defaults.cluster_graph.max_cluster_size,
    )

    seed: int = Field(
        description="The seed to use for the clustering.",
        default=graphrag_config_defaults.cluster_graph.seed,
    )


class VDStarClusterGraphConfig(BaseClusterGraphConfig):
    """Configuration section for clustering graphs using the VDStar algorithm."""

    algorithm: Literal[GraphClusteringAlgorithmType.VDSTAR] = Field(
        description="VDStar graph clustering algorithm configuration.",
        default=GraphClusteringAlgorithmType.VDSTAR,
    )


ClusterGraphConfig = Annotated[
    LeidenClusterGraphConfig | VDStarClusterGraphConfig,
    Field(discriminator="algorithm"),
]
