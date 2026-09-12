# Copyright (C) 2026 luke russo -- ruff complaining

"""Tests specific to VDStar clustering adapter.

Everything here goes through the public cluster() entry point. The shared
output contract that every clustering algorithm must satisfy is pinned in
test_clustering_algorithms.py.
"""

import pytest
from graphrag.config.models.cluster_graph_config import VDStarClusterGraphConfig
from graphrag.graphs.hierarchical_vdstar import HierarchicalVDStar
from graphrag.graphs.types import Cluster, Edge

from tests.unit.graph_utils import by_level, triangle

EPSILON_LEVEL_PARAMS = [
    pytest.param([0.5], id="single"),
    pytest.param([0.25, 0.5], id="two"),
    pytest.param([0.25, 0.5, 0.75], id="three"),
    pytest.param([0.1, 0.3, 0.5, 0.7, 0.9], id="five"),
    pytest.param([0.2, 0.8], id="two-spread"),
]


def _two_triangles() -> list[Edge]:
    return triangle("") + triangle("2")


def _run(edges: list[Edge], **config_kwargs) -> list[Cluster]:
    config = VDStarClusterGraphConfig(**config_kwargs)
    return HierarchicalVDStar(config).cluster(edges)


class TestEpsLevelMapping:
    """the eps list determines the level count."""

    @pytest.mark.parametrize("epsilon_levels", EPSILON_LEVEL_PARAMS)
    def test_every_requested_eps_produces_a_level(self, epsilon_levels):
        clusters = _run(_two_triangles(), epsilon_levels=epsilon_levels)

        expected_levels = list(range(len(epsilon_levels)))

        assert sorted(by_level(clusters).keys()) == expected_levels


class TestInvalidInput:
    """Pin the failure modes the vdstar binding raises."""

    def test_empty_edge_list_raises(self):
        with pytest.raises(ValueError, match="n must be a positive integer"):
            _run([], epsilon_levels=[0.5])

    def test_single_node_graph_raises(self):
        with pytest.raises(ValueError, match="mu must be an integer"):
            _run([Edge("A", "A", 1.0)], epsilon_levels=[0.5])
