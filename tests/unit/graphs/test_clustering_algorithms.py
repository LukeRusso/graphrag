# Copyright (C) 2026 luke russo -- ruff complaining

"""Shared output contract for clustering algorithm adapters.

These tests call each adapter's cluster() directly, one layer below the
cluster_graph operation, which is covered separately in
tests/unit/indexing/test_cluster_graph.py.
"""

import pandas as pd
import pytest
from graphrag.config.models.cluster_graph_config import (
    LeidenClusterGraphConfig,
    VDStarClusterGraphConfig,
)
from graphrag.graphs.hierarchical_leiden import HierarchicalLeiden
from graphrag.graphs.hierarchical_vdstar import HierarchicalVDStar
from graphrag.graphs.types import Cluster, Edge
from graphrag_storage.memory_storage import MemoryStorage

from tests.unit.graph_utils import by_level, tiered_graph, triangle

MAX_CLUSTER_SIZE = 10
SEED = 0xDEADBEEF


def _leiden() -> HierarchicalLeiden:
    return HierarchicalLeiden(
        LeidenClusterGraphConfig(
            max_cluster_size=MAX_CLUSTER_SIZE, use_lcc=False, seed=SEED
        )
    )


def _vdstar() -> HierarchicalVDStar:
    return HierarchicalVDStar(
        VDStarClusterGraphConfig(epsilon_levels=[0.25, 0.5, 0.75]), MemoryStorage()
    )


ADAPTERS = [
    pytest.param(_leiden, id="leiden"),
    pytest.param(_vdstar, id="vdstar"),
]


async def _leiden_multi_level() -> list[Cluster]:
    relationships = pd.read_parquet("tests/verbs/data/relationships.parquet")
    edges = [
        Edge(str(source), str(target), float(weight))
        for source, target, weight in zip(
            relationships["source"],
            relationships["target"],
            relationships["weight"],
            strict=False,
        )
    ]
    return await HierarchicalLeiden(
        LeidenClusterGraphConfig(
            max_cluster_size=MAX_CLUSTER_SIZE, use_lcc=False, seed=SEED
        )
    ).cluster(edges)


async def _vdstar_multi_level(epsilon_levels: list[float]) -> list[Cluster]:
    return await HierarchicalVDStar(
        VDStarClusterGraphConfig(epsilon_levels=epsilon_levels), MemoryStorage()
    ).cluster(tiered_graph())


MULTI_LEVEL_ADAPTERS = [
    pytest.param(_leiden_multi_level, id="leiden"),
    pytest.param(lambda: _vdstar_multi_level([0.5]), id="vdstar-single"),
    pytest.param(lambda: _vdstar_multi_level([0.25, 0.5]), id="vdstar-two"),
    pytest.param(lambda: _vdstar_multi_level([0.25, 0.5, 0.75]), id="vdstar-three"),
    pytest.param(
        lambda: _vdstar_multi_level([0.1, 0.3, 0.5, 0.7, 0.9]), id="vdstar-five"
    ),
    pytest.param(lambda: _vdstar_multi_level([0.2, 0.8]), id="vdstar-two-spread"),
]


@pytest.mark.parametrize("make_clusters", MULTI_LEVEL_ADAPTERS)
class TestAdapterOutputContract:
    """Invariants every IClusteringAlgorithm implementation must uphold."""

    async def test_returns_clusters(self, make_clusters):
        clusters = await make_clusters()

        assert isinstance(clusters, list)
        assert clusters
        assert all(isinstance(c, Cluster) for c in clusters)

    async def test_cluster_ids_are_globally_unique(self, make_clusters):
        ids = [c.cluster_id for c in await make_clusters()]

        assert len(ids) == len(set(ids))

    async def test_cluster_ids_are_non_negative_ints(self, make_clusters):
        clusters = await make_clusters()

        assert all(isinstance(c.cluster_id, int) for c in clusters)
        assert min(c.cluster_id for c in clusters) >= 0

    async def test_nodes_are_strings(self, make_clusters):
        clusters = await make_clusters()

        for cluster in clusters:
            assert isinstance(cluster.nodes, list)
            assert cluster.nodes
            assert all(isinstance(node, str) for node in cluster.nodes)

    async def test_node_belongs_to_at_most_one_cluster_per_level(self, make_clusters):
        grouped = by_level(await make_clusters())

        for level, clusters in grouped.items():
            members = [node for c in clusters for node in c.nodes]
            assert len(members) == len(set(members)), f"overlap at level {level}"


class TestAdapterDisconnectionContract:
    """Disconnected components must not be merged by any adapter."""

    @pytest.mark.parametrize("make_adapter", ADAPTERS)
    async def test_disconnected_components_do_not_merge(self, make_adapter):
        clusters = await make_adapter().cluster(triangle("") + triangle("2"))
        node_sets = [set(c.nodes) for c in clusters]

        assert {"A", "B", "C"} in node_sets
        assert {"2A", "2B", "2C"} in node_sets


@pytest.mark.parametrize("make_clusters", MULTI_LEVEL_ADAPTERS)
class TestHierarchyContract:
    """Invariants that hold for any multi-level clustering output."""

    async def test_levels_are_contiguous_from_zero(self, make_clusters):
        levels = sorted(by_level(await make_clusters()))

        assert levels == list(range(len(levels)))

    async def test_level_zero_has_the_largest_mean_cluster_size(self, make_clusters):
        grouped = by_level(await make_clusters())

        mean_sizes = {
            level: sum(len(c.nodes) for c in group) / len(group)
            for level, group in grouped.items()
        }

        assert mean_sizes[0] == max(mean_sizes.values())

    async def test_largest_cluster_shrinks_as_level_increases(self, make_clusters):
        grouped = by_level(await make_clusters())

        if len(grouped) < 2:
            pytest.skip("a single level has nothing to compare against")

        largest_at_level_zero = max(len(c.nodes) for c in grouped[0])
        largest_at_last_level = max(len(c.nodes) for c in grouped[max(grouped)])

        assert largest_at_level_zero > largest_at_last_level

    async def test_parent_ids_reference_coarser_level(self, make_clusters):
        clusters = await make_clusters()
        level_of = {c.cluster_id: c.level for c in clusters}

        for cluster in clusters:
            if cluster.level > 0:
                assert level_of[cluster.parent_cluster_id] == cluster.level - 1

    async def test_parent_minus_one_iff_level_zero(self, make_clusters):
        clusters = await make_clusters()

        for cluster in clusters:
            assert (cluster.parent_cluster_id == -1) == (cluster.level == 0)


class TestAdapterCoverage:
    """Testing node coverage"""

    async def test_leiden_covers_all_nodes_at_level_zero(self):
        edges = triangle("") + triangle("2")
        input_nodes = {e.source for e in edges} | {e.dest for e in edges}

        clusters = await _leiden().cluster(edges)
        covered = {node for c in clusters if c.level == 0 for node in c.nodes}

        assert covered == input_nodes

    ## VDSTAR doesn't assign all. TODO: discuss with junhao about best ways to handle this
