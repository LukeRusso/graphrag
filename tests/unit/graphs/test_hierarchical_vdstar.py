# Copyright (C) 2026 luke russo -- ruff complaining

"""Tests specific to VDStar clustering adapter.

The shared output contract that every clustering algorithm must satisfy is
pinned in test_clustering_algorithms.py.
"""

import pytest
import vdstar_core as vdstar
from graphrag.config.models.cluster_graph_config import VDStarClusterGraphConfig
from graphrag.graphs.hierarchical_vdstar import HierarchicalVDStar
from graphrag.graphs.types import Cluster, Edge
from graphrag.graphs.vdstar_state import GRAPH_BLOB_KEY
from graphrag_storage.memory_storage import MemoryStorage

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


async def _cluster(edges: list[Edge], **config_kwargs) -> list[Cluster]:
    config = VDStarClusterGraphConfig(**config_kwargs)
    return await HierarchicalVDStar(config, MemoryStorage()).cluster(edges)


def _get_vdstar(storage: MemoryStorage, **config_kwargs) -> HierarchicalVDStar:
    config = VDStarClusterGraphConfig(
        epsilon_levels=[0.5], use_lcc=False, **config_kwargs
    )
    return HierarchicalVDStar(config, storage)


def _all_nodes(clusters: list[Cluster]) -> set[str]:
    return {node for cluster in clusters for node in cluster.nodes}


async def _vdstar_vertex_count(storage: MemoryStorage) -> int:
    """The vertex count of the graph the engine last persisted."""
    blob = await storage.get(GRAPH_BLOB_KEY, as_bytes=True)
    return vdstar.GraphStore.read_bytes(blob).get_vertex_num()


class TestEpsLevelMapping:
    """the eps list determines the level count."""

    @pytest.mark.parametrize("epsilon_levels", EPSILON_LEVEL_PARAMS)
    async def test_every_requested_eps_produces_a_level(self, epsilon_levels):
        clusters = await _cluster(_two_triangles(), epsilon_levels=epsilon_levels)

        expected_levels = list(range(len(epsilon_levels)))

        assert sorted(by_level(clusters).keys()) == expected_levels


class TestInvalidInput:
    """Pin the input validation actually enforced."""

    async def test_empty_edge_list_raises(self):
        with pytest.raises(IndexError, match="vertex count must be >= 1"):
            await _cluster([], epsilon_levels=[0.5])


class TestIncrementalClustering:
    """Testing functionality specific to incremental clustering (update flow)."""

    async def test_incremental_without_prior_state_raises(self):
        vdstar = _get_vdstar(MemoryStorage())

        with pytest.raises(RuntimeError, match="no persisted VD-Star state"):
            await vdstar.cluster_incremental([Edge("A", "B", 1.0)], [])

    async def test_incremental_returns_clusters_only(self):
        storage = MemoryStorage()
        vdstar = _get_vdstar(storage)

        await vdstar.cluster(_two_triangles())
        clusters = await vdstar.cluster_incremental([Edge("A", "2A", 1.0)], [])

        assert clusters
        assert all(isinstance(cluster, Cluster) for cluster in clusters)

    async def test_incremental_absorbing_a_delta_keeps_all_nodes(self):
        storage = MemoryStorage()
        vdstar = _get_vdstar(storage)

        await vdstar.cluster(_two_triangles())
        clusters = await vdstar.cluster_incremental([Edge("A", "2A", 1.0)], [])

        assert _all_nodes(clusters) == {edge.source for edge in _two_triangles()} | {
            edge.dest for edge in _two_triangles()
        }

    async def test_resume_keeps_existing_ids_and_adds_new_nodes_on_fresh_ids(self):
        """A resumed update renumbers nothing and gives new nodes fresh ids."""
        storage = MemoryStorage()
        vdstar = _get_vdstar(storage)

        await vdstar.cluster(triangle(""))
        ids_before = vdstar.node_id_map
        highest_before = max(ids_before.values())

        await vdstar.cluster_incremental(
            [Edge("D", "A", 1.0), Edge("D", "B", 1.0), Edge("D", "C", 1.0)], []
        )

        assert vdstar.node_id_map["D"] == highest_before + 1
        for node, engine_id in ids_before.items():
            assert vdstar.node_id_map[node] == engine_id

    async def test_new_node_ids_come_from_the_vdstar_engine(self):
        storage = MemoryStorage()
        vdstar = _get_vdstar(storage)

        await vdstar.cluster(triangle(""))
        ids_before = vdstar.node_id_map

        resumed = _get_vdstar(storage)
        await resumed.cluster_incremental([Edge("D", "A", 1.0)], [])
        await resumed.cluster_incremental([Edge("E", "A", 1.0)], [])

        assert resumed.node_id_map["D"] == ids_before["C"] + 1
        assert resumed.node_id_map["E"] == resumed.node_id_map["D"] + 1
        for node, engine_id in ids_before.items():
            assert resumed.node_id_map[node] == engine_id
        assert resumed.node_id_map["E"] == await _vdstar_vertex_count(storage)
