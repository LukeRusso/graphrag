import pytest
from graphrag.config.models.cluster_graph_config import (
    BaseClusterGraphConfig,
    LeidenClusterGraphConfig,
    VDStarClusterGraphConfig,
)
from graphrag.config.models.graph_clustering_config_types import (
    GraphClusteringAlgorithmType,
)


def test_base_cluster_graph_config_cant_be_initialised():
    with pytest.raises(
        TypeError,
        match="BaseClusterGraphConfig cannot be instantiated",
    ):
        BaseClusterGraphConfig(algorithm="Leiden")


def test_leiden_cluster_graph_config_defaults():
    config: LeidenClusterGraphConfig = LeidenClusterGraphConfig()
    assert config.algorithm == GraphClusteringAlgorithmType.LEIDEN
    assert config.use_lcc
    assert config.max_cluster_size == 10
    assert config.seed == 0xDEADBEEF


def test_vdstar_cluster_graph_config_defaults():
    config: VDStarClusterGraphConfig = VDStarClusterGraphConfig()
    assert config.algorithm == GraphClusteringAlgorithmType.VDSTAR
