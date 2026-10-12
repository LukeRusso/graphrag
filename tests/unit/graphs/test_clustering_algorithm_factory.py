# Copyright (C) 2026 luke russo -- ruff complaining

from typing import cast

import pytest
from graphrag.config.models.cluster_graph_config import (
    ClusterGraphConfig,
    LeidenClusterGraphConfig,
    VDStarClusterGraphConfig,
)
from graphrag.graphs.clustering.clustering_algorithm_factory import (
    ClusteringAlgorithmFactory,
)
from graphrag.graphs.hierarchical_leiden import HierarchicalLeiden
from graphrag.graphs.hierarchical_vdstar import HierarchicalVDStar

from tests.unit.graph_utils import cfg


def test_leiden_config_returns_leiden():
    config = ClusteringAlgorithmFactory.from_config(cfg(LeidenClusterGraphConfig()))
    assert isinstance(config, HierarchicalLeiden)


def test_vdstar_config_returns_vdstar():
    config = ClusteringAlgorithmFactory.from_config(cfg(VDStarClusterGraphConfig()))
    assert isinstance(config, HierarchicalVDStar)


def test_vdstar_config_is_passed_through():
    epsilon_levels = [0.3, 0.6]
    mu = 3
    rho = 0.02

    algorithm = ClusteringAlgorithmFactory.from_config(
        cfg(VDStarClusterGraphConfig(epsilon_levels=epsilon_levels, mu=mu, rho=rho))
    )

    assert algorithm.config.epsilon_levels == epsilon_levels
    assert algorithm.config.mu == mu
    assert algorithm.config.rho == rho


def test_unknown_config_throws():
    config = cfg()
    config.cluster_graph = cast("ClusterGraphConfig", "not-a-clustering-config")

    with pytest.raises(
        ValueError,
        match="Unhandled graph clustering algorithm type during algorithm object instantiation",
    ):
        ClusteringAlgorithmFactory.from_config(config)
