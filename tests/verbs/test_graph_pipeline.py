# Copyright (C) 2026 luke russo -- ruff complaining

"""Tests for the graph indexing pipelines: build and update."""

import json
from pathlib import Path

import pandas as pd
import pytest
from graphrag.api.index import build_index
from graphrag.config.enums import CommunityUpdateMode, IndexingMethod
from graphrag.config.models.cluster_graph_config import ClusterGraphConfig
from graphrag.config.models.graph_rag_config import GraphRagConfig
from graphrag.index.typing.pipeline_run_result import PipelineRunResult

from tests.unit.graph_utils import (
    cfg,
    entities,
    leiden_config,
    relationships,
    vdstar_config,
)

GRAPH_WORKFLOWS = [
    "load_graph",
    "finalize_graph",
    "create_communities",
    "create_community_reports",
    "generate_text_embeddings",
]

GRAPH_UPDATE_WORKFLOWS = [
    "load_graph",
    "finalize_graph",
    "graph_diff",
    "create_communities",
    "create_community_reports",
    "generate_text_embeddings",
    "publish_graph",
    "update_communities",
    "update_clean_state",
]

# the supplied graph: two disconnected triangles, one text unit per edge
SUPPLIED_TITLES = ("A", "B", "C", "D", "E", "F")
SUPPLIED_EDGES = [
    ("A", "B", "t0"),
    ("A", "C", "t1"),
    ("B", "C", "t2"),
    ("D", "E", "t3"),
    ("D", "F", "t4"),
    ("E", "F", "t5"),
]
EXPECTED_COMMUNITIES = {
    frozenset({"A", "B", "C"}): frozenset({"t0", "t1", "t2"}),
    frozenset({"D", "E", "F"}): frozenset({"t3", "t4", "t5"}),
}

# the triangle an update run adds to the supplied graph
ADDED_TITLES = ("G", "H", "I")
ADDED_EDGES = [("G", "H", "t6"), ("G", "I", "t7"), ("H", "I", "t8")]
ADDED_COMMUNITY = frozenset(ADDED_TITLES)

# the mock completion model replays this, so it has to parse as a community report
REPORT = json.dumps({
    "title": "Community report",
    "summary": "A summary.",
    "rating": 5.0,
    "rating_explanation": "Because.",
    "findings": [{"summary": "A finding.", "explanation": "Why."}],
})


CLUSTERING = [
    pytest.param(lambda: leiden_config(), id="leiden"),
    pytest.param(lambda: vdstar_config(), id="vdstar"),
]

# Each clustering method against each update mode it allows, over the same change:
# the supplied graph gains a third triangle. `seeded` builds an index first, so the
# engine has persisted state: additive clusters the added triangle alone, incremental
# re-clusters the whole graph, so every community comes back.
UPDATE_CASES = [
    pytest.param(
        lambda: leiden_config(), False, [ADDED_COMMUNITY], id="leiden-additive"
    ),
    pytest.param(
        lambda: vdstar_config(update_mode=CommunityUpdateMode.Additive),
        False,
        [ADDED_COMMUNITY],
        id="vdstar-additive-no-state",
    ),
    pytest.param(
        lambda: vdstar_config(),
        True,
        [frozenset({"A", "B", "C"}), frozenset({"D", "E", "F"}), ADDED_COMMUNITY],
        id="vdstar-incremental",
    ),
]


def _config(tmp_path: Path, cluster_config: ClusterGraphConfig) -> GraphRagConfig:
    """A config whose every storage lives under tmp_path, with mock models."""
    config = cfg(cluster_config, base_dir=tmp_path)
    config.vector_store.db_uri = str(tmp_path / "lancedb")
    # mock providers, so the model-backed workflows run for real without a model
    completion = config.get_completion_model_config(
        config.community_reports.completion_model_id
    )
    completion.type = "mock"
    completion.mock_responses = [REPORT]
    embedding = config.get_embedding_model_config(config.embed_text.embedding_model_id)
    embedding.type = "mock"
    # length has to match vector_store.index_schema's vector_size
    embedding.mock_responses = [0.0] * 3072
    return config


def _write_graph(
    directory: Path, titles: tuple[str, ...], edges: list[tuple[str, str, str]]
) -> None:
    """Write a graph's entities and relationships tables into a directory."""
    directory.mkdir(parents=True, exist_ok=True)
    entities(*titles).to_parquet(directory / "entities.parquet", index=False)
    relationships(*edges).to_parquet(directory / "relationships.parquet", index=False)


def _read_output(tmp_path: Path, name: str) -> pd.DataFrame:
    """Read a table from the build output storage."""
    return pd.read_parquet(tmp_path / "output" / f"{name}.parquet")


async def _build(
    config: GraphRagConfig, *, is_update_run: bool = False
) -> list[PipelineRunResult]:
    """Run the graph pipeline and return its per-workflow results."""
    return await build_index(
        config, method=IndexingMethod.Graph, is_update_run=is_update_run
    )


async def _run_update(
    tmp_path: Path, cluster_config: ClusterGraphConfig, *, seeded: bool
) -> list[PipelineRunResult]:
    """Run an update whose supplied graph gains a third triangle.

    A seeded case builds the two-triangle graph first, so the clustering engine has
    persisted state; an unseeded case gets that graph written straight to output as
    the previous index. Either way the update is then handed the extended graph.
    """
    config = _config(tmp_path, cluster_config)
    if seeded:
        _write_graph(tmp_path, SUPPLIED_TITLES, SUPPLIED_EDGES)
        await _build(config)
    else:
        _write_graph(tmp_path / "output", SUPPLIED_TITLES, SUPPLIED_EDGES)
    _write_graph(tmp_path, SUPPLIED_TITLES + ADDED_TITLES, SUPPLIED_EDGES + ADDED_EDGES)
    return await _build(config, is_update_run=True)


def _delta_members(tmp_path: Path) -> set[frozenset[str]]:
    """The entity titles of each community the update staged in its delta."""
    delta = next((tmp_path / "update").glob("*/delta"))
    entities_table = pd.read_parquet(delta / "entities.parquet")
    communities = pd.read_parquet(delta / "communities.parquet")
    title_of = dict(zip(entities_table["id"], entities_table["title"], strict=True))
    return {
        frozenset(title_of[entity_id] for entity_id in row["entity_ids"])
        for _, row in communities.iterrows()
    }


class TestGraphBuild:
    """The graph build pipeline: a supplied graph in, an index out."""

    @pytest.mark.parametrize("cluster_config", CLUSTERING)
    async def test_runs_every_workflow_in_order(self, tmp_path, cluster_config):
        _write_graph(tmp_path, SUPPLIED_TITLES, SUPPLIED_EDGES)

        outputs = await _build(_config(tmp_path, cluster_config()))

        assert [output.workflow for output in outputs] == GRAPH_WORKFLOWS
        assert [output.error for output in outputs] == [None] * len(GRAPH_WORKFLOWS)

    @pytest.mark.parametrize("cluster_config", CLUSTERING)
    async def test_communities_are_the_supplied_triangles_and_their_text_units(
        self, tmp_path, cluster_config
    ):
        _write_graph(tmp_path, SUPPLIED_TITLES, SUPPLIED_EDGES)
        await _build(_config(tmp_path, cluster_config()))
        entities_table = _read_output(tmp_path, "entities")
        communities = _read_output(tmp_path, "communities")

        # finalize_graph reissues entity ids, so titles are the stable handle
        title_of = dict(zip(entities_table["id"], entities_table["title"], strict=True))
        by_members = {
            frozenset(title_of[entity_id] for entity_id in row["entity_ids"]): (
                frozenset(row["text_unit_ids"])
            )
            for _, row in communities.iterrows()
        }

        assert len(communities) == 2
        assert by_members == EXPECTED_COMMUNITIES

    @pytest.mark.parametrize("cluster_config", CLUSTERING)
    async def test_writes_a_report_for_each_community(self, tmp_path, cluster_config):
        _write_graph(tmp_path, SUPPLIED_TITLES, SUPPLIED_EDGES)
        await _build(_config(tmp_path, cluster_config()))
        communities = _read_output(tmp_path, "communities")
        reports = _read_output(tmp_path, "community_reports")

        assert len(reports) == 2
        assert set(reports["community"]) == set(communities["community"])

    async def test_an_edgeless_graph_builds_with_an_empty_communities_table(
        self, tmp_path
    ):
        _write_graph(tmp_path, SUPPLIED_TITLES, [])

        outputs = await _build(_config(tmp_path, leiden_config()))

        assert [output.error for output in outputs] == [None] * len(GRAPH_WORKFLOWS)
        assert _read_output(tmp_path, "communities").empty


class TestGraphUpdate:
    """The graph update pipeline: every clustering method against every update mode."""

    async def test_runs_every_workflow_in_order_without_errors(self, tmp_path):
        outputs = await _run_update(tmp_path, leiden_config(), seeded=False)

        assert [output.workflow for output in outputs] == GRAPH_UPDATE_WORKFLOWS
        assert [output.error for output in outputs] == [None] * len(
            GRAPH_UPDATE_WORKFLOWS
        )

    @pytest.mark.parametrize(("cluster_config", "seeded", "expected"), UPDATE_CASES)
    async def test_clusters_according_to_the_update_mode(
        self, tmp_path, cluster_config, seeded, expected
    ):
        outputs = await _run_update(tmp_path, cluster_config(), seeded=seeded)

        assert [output.error for output in outputs] == [None] * len(
            GRAPH_UPDATE_WORKFLOWS
        )
        assert _delta_members(tmp_path) == set(expected)

    async def test_an_incremental_update_without_persisted_state_errors(self, tmp_path):
        outputs = await _run_update(tmp_path, vdstar_config(), seeded=False)

        last = outputs[-1]
        assert last.workflow == "create_communities"
        assert isinstance(last.error, RuntimeError)
        assert "no persisted clustering state" in str(last.error)
