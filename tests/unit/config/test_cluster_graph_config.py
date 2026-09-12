# Copyright (C) 2026 luke russo -- ruff complaining

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
    expected_epsilon_levels = [0.25, 0.5, 0.75]
    expected_mu = 2
    expected_rho = 0.05

    config: VDStarClusterGraphConfig = VDStarClusterGraphConfig()

    assert config.algorithm == GraphClusteringAlgorithmType.VDSTAR
    assert config.use_lcc
    assert config.epsilon_levels == expected_epsilon_levels
    assert config.mu == expected_mu
    assert config.rho == expected_rho
