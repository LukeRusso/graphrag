# Copyright (C) 2026 luke russo -- ruff complaining

import json
from pathlib import Path

import pandas as pd
import pytest
from graphrag.api.index import build_index
from graphrag.config.enums import IndexingMethod
from graphrag.config.models.cluster_graph_config import (
    ClusterGraphConfig,
    LeidenClusterGraphConfig,
    VDStarClusterGraphConfig,
)
from graphrag.config.models.graph_rag_config import GraphRagConfig
from graphrag.index.typing.pipeline_run_result import PipelineRunResult
from graphrag.index.workflows.factory import PipelineFactory

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
    "update_clean_state",
]

# two disconnected triangles, one text unit per edge
TRIANGLE_EDGES = [
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

CLUSTERING = [
    pytest.param(
        lambda: LeidenClusterGraphConfig(max_cluster_size=10, use_lcc=False, seed=42),
        id="leiden",
    ),
    pytest.param(
        lambda: VDStarClusterGraphConfig(epsilon_levels=[0.5], use_lcc=False),
        id="vdstar",
    ),
]

# the mock completion model replays this, so it has to parse as a community report
REPORT = json.dumps({
    "title": "Community report",
    "summary": "A summary.",
    "rating": 5.0,
    "rating_explanation": "Because.",
    "findings": [{"summary": "A finding.", "explanation": "Why."}],
})


def _entities() -> pd.DataFrame:
    rows = [
        ("e1", "A"),
        ("e2", "B"),
        ("e3", "C"),
        ("e4", "D"),
        ("e5", "E"),
        ("e6", "F"),
    ]
    return pd.DataFrame([
        {
            "id": entity_id,
            "human_readable_id": i,
            "title": title,
            "type": "THING",
            "description": f"{title} is a thing.",
            "text_unit_ids": ["t1"],
            "frequency": 1,
            "degree": 1,
        }
        for i, (entity_id, title) in enumerate(rows)
    ])


def _relationships() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "id": f"r{i}",
            "human_readable_id": i,
            "source": source,
            "target": target,
            "description": f"{source} relates to {target}.",
            "weight": 1.0,
            "combined_degree": 2,
            "text_unit_ids": [text_unit],
        }
        for i, (source, target, text_unit) in enumerate(TRIANGLE_EDGES)
    ])


def _config(tmp_path: Path, cluster_graph: ClusterGraphConfig) -> GraphRagConfig:
    _entities().to_parquet(tmp_path / "entities.parquet", index=False)
    _relationships().to_parquet(tmp_path / "relationships.parquet", index=False)
    return GraphRagConfig(
        graph_input_storage={"type": "file", "base_dir": str(tmp_path)},
        output_storage={"type": "file", "base_dir": str(tmp_path / "output")},
        update_output_storage={"type": "memory"},
        vector_store={"type": "lancedb", "db_uri": str(tmp_path / "lancedb")},
        cluster_graph=cluster_graph,
        # mock providers, so the model-backed workflows run for real without a model
        completion_models={
            "default_completion_model": {
                "type": "mock",
                "model_provider": "openai",
                "model": "gpt-4o",
                "mock_responses": [REPORT],
            },
        },
        embedding_models={
            "default_embedding_model": {
                "type": "mock",
                "model_provider": "openai",
                "model": "text-embedding-3-large",
                # length has to match vector_store.index_schema's vector_size
                "mock_responses": [0.0] * 3072,
            },
        },
    )


async def _run(
    tmp_path: Path, cluster_graph: ClusterGraphConfig
) -> tuple[list[PipelineRunResult], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    outputs = await build_index(
        _config(tmp_path, cluster_graph), method=IndexingMethod.Graph
    )
    output_dir = tmp_path / "output"
    return (
        outputs,
        pd.read_parquet(output_dir / "entities.parquet"),
        pd.read_parquet(output_dir / "communities.parquet"),
        pd.read_parquet(output_dir / "community_reports.parquet"),
    )


def test_graph_pipeline_is_the_graph_workflows_in_order():
    assert PipelineFactory.pipelines[IndexingMethod.Graph] == GRAPH_WORKFLOWS


@pytest.mark.parametrize("cluster_graph", CLUSTERING)
async def test_runs_every_workflow_in_order(tmp_path, cluster_graph):
    outputs, _, _, _ = await _run(tmp_path, cluster_graph())

    assert [output.workflow for output in outputs] == GRAPH_WORKFLOWS
    assert [output.error for output in outputs] == [None] * len(GRAPH_WORKFLOWS)


@pytest.mark.parametrize("cluster_graph", CLUSTERING)
async def test_communities_are_the_supplied_triangles_and_their_text_units(
    tmp_path, cluster_graph
):
    _, entities, communities, _ = await _run(tmp_path, cluster_graph())

    # finalize_graph reissues entity ids, so titles are the stable handle
    title_of = dict(zip(entities["id"], entities["title"], strict=True))
    by_members = {
        frozenset(title_of[entity_id] for entity_id in row["entity_ids"]): frozenset(
            row["text_unit_ids"]
        )
        for _, row in communities.iterrows()
    }

    assert len(communities) == 2
    assert by_members == EXPECTED_COMMUNITIES


@pytest.mark.parametrize("cluster_graph", CLUSTERING)
async def test_writes_a_report_for_each_community(tmp_path, cluster_graph):
    _, _, communities, reports = await _run(tmp_path, cluster_graph())

    assert len(reports) == 2
    assert set(reports["community"]) == set(communities["community"])


async def test_graph_update_runs_workflows_in_order_without_errors(tmp_path):
    previous = tmp_path / "output"
    previous.mkdir()
    _entities().to_parquet(previous / "entities.parquet", index=False)
    _relationships().to_parquet(previous / "relationships.parquet", index=False)

    outputs = await build_index(
        _config(
            tmp_path,
            LeidenClusterGraphConfig(max_cluster_size=10, use_lcc=False, seed=42),
        ),
        method=IndexingMethod.Graph,
        is_update_run=True,
    )

    assert [output.workflow for output in outputs] == GRAPH_UPDATE_WORKFLOWS
    assert [output.error for output in outputs] == [None] * len(GRAPH_UPDATE_WORKFLOWS)
