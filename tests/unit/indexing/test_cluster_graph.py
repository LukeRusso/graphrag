# Copyright (C) 2026 luke russo -- ruff complaining

"""Tests for the cluster_graph operation."""

import asyncio
from pathlib import Path

import pandas as pd
import pytest
from graphrag.config.enums import CommunityUpdateMode
from graphrag.config.models.cluster_graph_config import ClusterGraphConfig
from graphrag.config.models.graph_rag_config import GraphRagConfig
from graphrag.graphs.types import Cluster, Edge
from graphrag.index.operations.cluster_graph import cluster_graph
from graphrag.index.update.relationships import GraphDelta
from graphrag_storage import StorageConfig

from tests.unit.graph_utils import (
    by_level,
    cfg,
    leiden_config,
    node_sets,
    relationships,
    vdstar_config,
)


def _make_edges(
    rows: list[tuple[str, str, float]],
) -> pd.DataFrame:
    """Build a minimal relationships DataFrame from (source, target, weight)."""
    return pd.DataFrame([{"source": s, "target": t, "weight": w} for s, t, w in rows])


def _config(tmp_path: Path, cluster_graph_config: ClusterGraphConfig) -> GraphRagConfig:
    """An update run resolves and persists VD-Star state through this storage."""
    return cfg(
        cluster_graph_config,
        StorageConfig(type="file", base_dir=str(tmp_path / "update")),
    )


def _new_triangle_delta() -> GraphDelta:
    return GraphDelta(
        added_edges=[Edge("D", "E", 1.0), Edge("D", "F", 1.0), Edge("E", "F", 1.0)],
        removed_edges=[],
        added_vertices=["D", "E", "F"],
        removed_vertices=[],
    )


ADAPTER_CONFIGS = [
    pytest.param(lambda: cfg(leiden_config()), id="leiden"),
    pytest.param(lambda: cfg(vdstar_config()), id="vdstar"),
]

OLD_GRAPH = [("A", "B"), ("A", "C"), ("B", "C")]
NEW_GRAPH = [*OLD_GRAPH, ("D", "E"), ("D", "F"), ("E", "F")]


# -------------------------------------------------------------------
# Basic clustering
# -------------------------------------------------------------------


class TestClusterGraphBasic:
    """Verify basic clustering on small synthetic graphs."""

    async def test_single_triangle(self):
        """A single triangle should produce one community at level 0."""
        edges = _make_edges([("X", "Y", 1.0), ("X", "Z", 1.0), ("Y", "Z", 1.0)])
        clusters = await cluster_graph(edges, cfg(leiden_config()))

        assert len(clusters) == 1
        level, _cid, parent, nodes = clusters[0]
        assert level == 0
        assert parent == -1
        assert set(nodes) == {"X", "Y", "Z"}

    async def test_two_disconnected_cliques(self):
        """Two disconnected triangles should produce two communities."""
        edges = _make_edges([
            ("A", "B", 1.0),
            ("A", "C", 1.0),
            ("B", "C", 1.0),
            ("D", "E", 1.0),
            ("D", "F", 1.0),
            ("E", "F", 1.0),
        ])
        clusters = await cluster_graph(edges, cfg(leiden_config()))

        assert len(clusters) == 2
        assert {"A", "B", "C"} in node_sets(clusters)
        assert {"D", "E", "F"} in node_sets(clusters)
        for level, _, parent, _ in clusters:
            assert level == 0
            assert parent == -1

    async def test_lcc_filters_to_largest_component(self):
        """With use_lcc=True, only the largest connected component is kept."""
        edges = _make_edges([
            ("A", "B", 1.0),
            ("A", "C", 1.0),
            ("B", "C", 1.0),
            ("D", "E", 1.0),
            ("D", "F", 1.0),
            ("E", "F", 1.0),
        ])
        clusters = await cluster_graph(edges, cfg(leiden_config(use_lcc=True)))

        assert len(clusters) == 1
        all_nodes = set(clusters[0][3])
        assert len(all_nodes) == 3


# -------------------------------------------------------------------
# Edge normalization
# -------------------------------------------------------------------


@pytest.mark.parametrize("make_config", ADAPTER_CONFIGS)
class TestEdgeNormalization:
    """Verify that direction normalization and deduplication work."""

    async def test_reversed_edges_produce_same_result(self, make_config):
        """Reversing all edge directions should yield identical clusters."""
        forward = _make_edges([
            ("A", "B", 1.0),
            ("A", "C", 1.0),
            ("B", "C", 1.0),
            ("D", "E", 1.0),
            ("D", "F", 1.0),
            ("E", "F", 1.0),
        ])
        backward = _make_edges([
            ("B", "A", 1.0),
            ("C", "A", 1.0),
            ("C", "B", 1.0),
            ("E", "D", 1.0),
            ("F", "D", 1.0),
            ("F", "E", 1.0),
        ])
        clusters_fwd = await cluster_graph(forward, make_config())
        clusters_bwd = await cluster_graph(backward, make_config())

        assert node_sets(clusters_fwd) == node_sets(clusters_bwd)

    async def test_duplicate_edges_are_deduped(self, make_config):
        """A→B and B→A should be treated as one edge after normalization."""
        edges = _make_edges([
            ("A", "B", 1.0),
            ("B", "A", 2.0),
            ("A", "C", 1.0),
            ("B", "C", 1.0),
        ])
        clusters = await cluster_graph(edges, make_config())

        assert len(clusters) == 1
        assert set(clusters[0][3]) == {"A", "B", "C"}

    async def test_a_frame_without_a_weight_column_clusters(self, make_config):
        """Edges without a weight column should still cluster."""
        edges = pd.DataFrame({
            "source": ["A", "A", "B"],
            "target": ["B", "C", "C"],
        })
        clusters = await cluster_graph(edges, make_config())

        assert len(clusters) == 1
        assert set(clusters[0][3]) == {"A", "B", "C"}


# -------------------------------------------------------------------
# Determinism
# -------------------------------------------------------------------


class TestDeterminism:
    """Verify that seeding produces reproducible results."""

    async def test_same_seed_same_result(self):
        """Identical seed should yield identical output."""
        edges = _make_edges([
            ("A", "B", 1.0),
            ("A", "C", 1.0),
            ("B", "C", 1.0),
            ("D", "E", 1.0),
            ("D", "F", 1.0),
            ("E", "F", 1.0),
        ])
        c1 = await cluster_graph(edges, cfg(leiden_config(seed=123)))
        c2 = await cluster_graph(edges, cfg(leiden_config(seed=123)))

        assert c1 == c2

    async def test_does_not_mutate_input(self):
        """cluster_graph should not modify the input DataFrame."""
        edges = _make_edges([
            ("A", "B", 1.0),
            ("A", "C", 1.0),
            ("B", "C", 1.0),
        ])
        original = edges.copy()
        await cluster_graph(edges, cfg(leiden_config()))

        pd.testing.assert_frame_equal(edges, original)


# -------------------------------------------------------------------
# Output structure
# -------------------------------------------------------------------


@pytest.mark.parametrize("make_config", ADAPTER_CONFIGS)
class TestOutputStructure:
    """Verify the shape and types of the Communities output."""

    async def test_output_tuple_structure(self, make_config):
        """Each entry should be (level, community_id, parent, node_list)."""
        edges = _make_edges([("A", "B", 1.0), ("A", "C", 1.0), ("B", "C", 1.0)])
        clusters = await cluster_graph(edges, make_config())

        for entry in clusters:
            assert len(entry) == 4
            level, cid, parent, nodes = entry
            assert isinstance(level, int)
            assert isinstance(cid, int)
            assert isinstance(parent, int)
            assert isinstance(nodes, list)
            assert all(isinstance(n, str) for n in nodes)

    async def test_level_zero_has_parent_minus_one(self, make_config):
        """All level-0 clusters should have parent == -1."""
        edges = _make_edges([
            ("A", "B", 1.0),
            ("A", "C", 1.0),
            ("B", "C", 1.0),
            ("D", "E", 1.0),
            ("D", "F", 1.0),
            ("E", "F", 1.0),
        ])
        clusters = await cluster_graph(edges, make_config())

        for level, _, parent, _ in clusters:
            if level == 0:
                assert parent == -1


class TestOutputStructureLeiden:
    """Output details specific to Leiden, which assigns every node a cluster."""

    async def test_all_nodes_covered_at_each_level(self):
        """At any given level, the union of all community nodes should
        equal exactly the set of all nodes in the graph for that level."""
        edges = _make_edges([
            ("A", "B", 1.0),
            ("A", "C", 1.0),
            ("B", "C", 1.0),
            ("D", "E", 1.0),
            ("D", "F", 1.0),
            ("E", "F", 1.0),
        ])
        clusters = await cluster_graph(edges, cfg(leiden_config()))

        levels = by_level(clusters)
        all_nodes = {"A", "B", "C", "D", "E", "F"}
        for level, grouped in levels.items():
            covered_nodes = {node for c in grouped for node in c.nodes}
            assert covered_nodes == all_nodes, (
                f"Level {level}: expected {all_nodes}, got {covered_nodes}"
            )


# -------------------------------------------------------------------
# Real test data (golden file regression)
# -------------------------------------------------------------------


class TestClusterGraphRealData:
    """Regression tests using the shared test fixture data."""

    @pytest.fixture(scope="module")
    def clusters(self) -> list[Cluster]:
        """The real-data graph clustered once, shared by the regression checks."""
        relationships = pd.read_parquet("tests/verbs/data/relationships.parquet")
        return asyncio.run(
            cluster_graph(
                relationships,
                cfg(leiden_config(use_lcc=True, seed=0xDEADBEEF)),
            )
        )

    def test_cluster_count(self, clusters: list[Cluster]):
        """Pin the expected number of clusters from the fixture data."""
        assert len(clusters) == 122

    def test_level_distribution(self, clusters: list[Cluster]):
        """Pin the expected number of clusters per level."""
        from collections import Counter

        level_counts = Counter(c[0] for c in clusters)
        assert level_counts == {0: 23, 1: 65, 2: 32, 3: 2}

    def test_level_zero_nodes_sample(self, clusters: list[Cluster]):
        """Spot-check a few known nodes in level-0 clusters."""
        level_0 = [c for c in clusters if c[0] == 0]
        all_level_0_nodes = set()
        for _, _, _, nodes in level_0:
            all_level_0_nodes.update(nodes)

        assert "SCROOGE" in all_level_0_nodes
        assert "ABRAHAM" in all_level_0_nodes
        assert "JACOB MARLEY" in all_level_0_nodes


class TestClusterGraphVDStar:
    """VDStar-specific behaviour through the cluster_graph operation."""

    async def test_eps_levels_produce_multiple_levels(self):
        """The eps list is VDStar's only lever for the level count."""
        edges = _make_edges([("A", "B", 1.0), ("A", "C", 1.0), ("B", "C", 1.0)])
        clusters = await cluster_graph(
            edges,
            cfg(vdstar_config(epsilon_levels=[0.25, 0.5, 0.75])),
        )

        levels = {c[0] for c in clusters}
        assert levels == {0, 1, 2}


class TestClusterGraphUpdate:
    """Verify each update mode clusters the delta the way the mode promises."""

    async def test_a_leiden_update_clusters_only_the_new_entities(self, tmp_path):
        clusters = await cluster_graph(
            relationships(*NEW_GRAPH),
            _config(tmp_path, leiden_config()),
            _new_triangle_delta(),
        )

        assert node_sets(clusters) == [{"D", "E", "F"}]

    async def test_a_vdstar_update_without_persisted_state_raises(self, tmp_path):
        with pytest.raises(RuntimeError, match="no persisted clustering state"):
            await cluster_graph(
                relationships(*NEW_GRAPH),
                _config(tmp_path, vdstar_config()),
                _new_triangle_delta(),
            )

    async def test_a_vdstar_update_in_additive_mode_clusters_only_the_new_entities(
        self, tmp_path
    ):
        config = _config(
            tmp_path, vdstar_config(update_mode=CommunityUpdateMode.Additive)
        )

        clusters = await cluster_graph(
            relationships(*NEW_GRAPH), config, _new_triangle_delta()
        )

        assert node_sets(clusters) == [{"D", "E", "F"}]

    async def test_a_vdstar_update_with_persisted_state_absorbs_the_delta(
        self, tmp_path
    ):
        config = _config(tmp_path, vdstar_config())
        await cluster_graph(relationships(*OLD_GRAPH), config)

        delta = GraphDelta(
            added_edges=[Edge("A", "D", 1.0), Edge("B", "D", 1.0)],
            removed_edges=[],
            added_vertices=["D"],
            removed_vertices=[],
        )
        clusters = await cluster_graph(
            relationships(*OLD_GRAPH, ("A", "D"), ("B", "D")), config, delta
        )

        assert node_sets(clusters) == [{"A", "B", "C", "D"}]

    async def test_a_vdstar_update_applies_the_removed_edges(self, tmp_path):
        config = _config(tmp_path, vdstar_config())
        await cluster_graph(relationships(*OLD_GRAPH, ("A", "D"), ("B", "D")), config)

        clusters = await cluster_graph(
            relationships(*OLD_GRAPH),
            config,
            GraphDelta(
                added_edges=[],
                removed_edges=[Edge("A", "D", 1.0), Edge("B", "D", 1.0)],
                added_vertices=[],
                removed_vertices=[],
            ),
        )

        assert node_sets(clusters) == [{"A", "B", "C"}]
