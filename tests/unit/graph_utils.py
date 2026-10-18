# Copyright (C) 2026 luke russo -- ruff complaining

"""Shared helpers for graph clustering tests."""

import json
from collections import defaultdict
from pathlib import Path

import pandas as pd
from graphrag.config.models.cluster_graph_config import (
    ClusterGraphConfig,
    LeidenClusterGraphConfig,
    VDStarClusterGraphConfig,
)
from graphrag.config.models.graph_rag_config import GraphRagConfig
from graphrag.data_model.schemas import (
    COMMUNITIES_FINAL_COLUMNS,
    ENTITIES_FINAL_COLUMNS,
    RELATIONSHIPS_FINAL_COLUMNS,
)
from graphrag.graphs.types import Cluster, Edge
from graphrag.index.run.utils import create_run_context
from graphrag.index.typing.context import PipelineRunContext
from graphrag.index.update.relationships import GraphDelta
from graphrag.index.workflows.graph_diff import GRAPH_DELTA_STATE_KEY
from graphrag_storage import StorageConfig, create_storage
from graphrag_storage.tables.parquet_table_provider import ParquetTableProvider
from graphrag_storage.tables.table_provider import TableProvider

from tests.unit.config.utils import get_default_graphrag_config

PERIOD = "2026-10-12"

DEFAULT_REPORT_RESPONSE = json.dumps({
    "title": "Community report",
    "summary": "A summary.",
    "rating": 5.0,
    "rating_explanation": "Because.",
    "findings": [{"summary": "A finding.", "explanation": "Why."}],
})


def vdstar_config(**overrides) -> VDStarClusterGraphConfig:
    """A deterministic VD-Star config with a single small epsilon level."""
    return VDStarClusterGraphConfig(**{
        "epsilon_levels": [0.5],
        "use_lcc": False,
        "seed": 42,
        **overrides,
    })


def leiden_config(**overrides) -> LeidenClusterGraphConfig:
    """A deterministic Leiden config."""
    return LeidenClusterGraphConfig(**{
        "max_cluster_size": 10,
        "use_lcc": False,
        "seed": 42,
        **overrides,
    })


def cfg(
    cluster_graph: ClusterGraphConfig | None = None,
    update_output_storage: StorageConfig | None = None,
    *,
    base_dir: Path | None = None,
) -> GraphRagConfig:
    """The default pipeline config with clustering config and memory storages, or files under base_dir."""
    config = get_default_graphrag_config()
    if cluster_graph is not None:
        config.cluster_graph = cluster_graph
    if base_dir is None:
        config.output_storage = StorageConfig(type="memory")
        config.update_output_storage = StorageConfig(type="memory")
    else:
        config.graph_input_storage = StorageConfig(type="file", base_dir=str(base_dir))
        config.output_storage = StorageConfig(
            type="file", base_dir=str(base_dir / "output")
        )
        config.update_output_storage = StorageConfig(
            type="file", base_dir=str(base_dir / "update")
        )
    if update_output_storage is not None:
        config.update_output_storage = update_output_storage
    completion = config.get_completion_model_config(
        config.community_reports.completion_model_id
    )
    completion.type = "mock"
    completion.mock_responses = [DEFAULT_REPORT_RESPONSE]
    return config


def triangle(name: str) -> list[Edge]:
    """A three-node clique, with an optional prefix on each node id."""
    a, b, c = (f"{name}{suffix}" for suffix in ("A", "B", "C"))
    return [Edge(a, b, 1.0), Edge(a, c, 1.0), Edge(b, c, 1.0)]


def tiered_graph() -> list[Edge]:
    """A dense core with pendant nodes, so the levels differ in cluster size."""
    names = {i: f"N{i}" for i in range(1, 11)}
    core = [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]
    pendant = [(1, i) for i in range(5, 11)]
    tail = [(5, 6), (6, 7)]
    return [Edge(names[u], names[v], 1.0) for u, v in [*core, *pendant, *tail]]


def by_level(clusters: list[Cluster]) -> dict[int, list[Cluster]]:
    """Group clusters by their level."""
    grouped: dict[int, list[Cluster]] = defaultdict(list)
    for cluster in clusters:
        grouped[cluster.level].append(cluster)
    return grouped


def node_sets(clusters: list[Cluster]) -> list[set[str]]:
    """The node set of every cluster, in order."""
    return [set(cluster.nodes) for cluster in clusters]


def entities(*titles: str) -> pd.DataFrame:
    """An entities frame, one row per title, with the id derived from it."""
    return pd.DataFrame(
        [
            {
                "id": f"e{title}",
                "human_readable_id": i,
                "title": title,
                "type": "THING",
                "description": f"{title} is a thing.",
                "text_unit_ids": [],
                "frequency": 1,
                "degree": 1,
            }
            for i, title in enumerate(titles)
        ],
        columns=ENTITIES_FINAL_COLUMNS,
    )


def relationships(*edges: tuple[str, str] | tuple[str, str, str]) -> pd.DataFrame:
    """A relationships frame, each edge optionally naming its text unit."""
    return pd.DataFrame(
        [
            {
                "id": f"r{i}",
                "human_readable_id": i,
                "source": edge[0],
                "target": edge[1],
                "description": f"{edge[0]} relates to {edge[1]}.",
                "weight": 1.0,
                "combined_degree": 2,
                "text_unit_ids": [edge[2]] if len(edge) == 3 else [],
            }
            for i, edge in enumerate(edges)
        ],
        columns=RELATIONSHIPS_FINAL_COLUMNS,
    )


def text_units(*ids: str) -> pd.DataFrame:
    """A text units frame, one row per id."""
    return pd.DataFrame(
        [
            {"id": unit, "text": f"Text for {unit}.", "document_id": None}
            for unit in ids
        ],
        columns=["id", "text", "document_id"],
    )


def communities(
    *groups: tuple[str, ...], level: int = 0, parents: list[int] | None = None
) -> pd.DataFrame:
    """A communities frame, one row per member group, ids numbered from zero."""
    frame = pd.DataFrame(
        [
            {
                "id": f"c{community}",
                "human_readable_id": community,
                "community": community,
                "level": level,
                "parent": -1,
                "children": [],
                "title": f"Community {community}",
                "entity_ids": [f"e{title}" for title in group],
                "relationship_ids": [],
                "text_unit_ids": [],
                "period": PERIOD,
                "size": len(group),
            }
            for community, group in enumerate(groups)
        ],
        columns=COMMUNITIES_FINAL_COLUMNS,
    )
    if parents is not None:
        frame["parent"] = parents
    return frame


def provider(path: Path) -> ParquetTableProvider:
    """Return a parquet table provider rooted at a directory."""
    return ParquetTableProvider(
        storage=create_storage(StorageConfig(base_dir=str(path)))
    )


def update_context(root: Path, timestamp: str) -> PipelineRunContext:
    """Return a run context pointing at a run's output and previous tables."""
    return create_run_context(
        output_table_provider=provider(root / "output"),
        previous_table_provider=provider(root / "update" / timestamp / "previous"),
        state={
            "update_timestamp": timestamp,
            GRAPH_DELTA_STATE_KEY: GraphDelta([], [], [], []),
        },
    )


async def write_tables(
    provider_: TableProvider, tables: dict[str, pd.DataFrame]
) -> None:
    """Write several tables into a table provider."""
    for name, table in tables.items():
        await provider_.write_dataframe(name, table)


def write_table(
    root: Path, timestamp: str, stage: str, name: str, table: pd.DataFrame
) -> None:
    """Write a table into one of a run's staging directories."""
    directory = root / "update" / timestamp / stage
    directory.mkdir(parents=True, exist_ok=True)
    table.to_parquet(directory / f"{name}.parquet", index=False)


def read_output(root: Path, name: str) -> pd.DataFrame:
    """Read a table from a run's output storage."""
    return pd.read_parquet(root / "output" / f"{name}.parquet")
