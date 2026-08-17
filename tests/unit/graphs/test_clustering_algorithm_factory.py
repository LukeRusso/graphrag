import pytest
from graphrag.config.models.cluster_graph_config import LeidenClusterGraphConfig
from graphrag.graphs.clustering.clustering_algorithm_factory import (
    ClusteringAlgorithmFactory,
)
from graphrag.graphs.hierarchical_leiden import HierarchicalLeiden


def test_leiden_config_returns_leiden():
    config = ClusteringAlgorithmFactory.from_config(LeidenClusterGraphConfig())
    assert isinstance(config, HierarchicalLeiden)


def test_unknown_config_throws():
    with pytest.raises(
        ValueError,
        match="Unhandled graph clustering algorithm type during algorithm object instantiation",
    ):
        ClusteringAlgorithmFactory.from_config({})
