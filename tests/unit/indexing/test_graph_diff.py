# Copyright (C) 2026 luke russo -- ruff complaining

from pathlib import Path

import pandas as pd
from graphrag.graphs.types import Edge
from graphrag.index.run.utils import create_run_context
from graphrag.index.typing.context import PipelineRunContext
from graphrag.index.workflows.graph_diff import GRAPH_DELTA_STATE_KEY, run_workflow
from graphrag_storage import StorageConfig, create_storage
from graphrag_storage.tables.parquet_table_provider import ParquetTableProvider

from tests.unit.config.utils import get_default_graphrag_config


def _entities(*titles: str) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "id": f"e{title}",
            "human_readable_id": i,
            "title": title,
            "type": "THING",
            "description": f"{title} is a thing.",
            "text_unit_ids": [],
        }
        for i, title in enumerate(titles)
    ])


def _relationships(*edges: tuple[str, str]) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "id": f"r{i}",
            "human_readable_id": i,
            "source": source,
            "target": target,
            "description": f"{source} relates to {target}.",
            "weight": 1.0,
            "text_unit_ids": [],
        }
        for i, (source, target) in enumerate(edges)
    ])


def _provider(path: Path) -> ParquetTableProvider:
    return ParquetTableProvider(
        storage=create_storage(StorageConfig(base_dir=str(path)))
    )


def _context(previous_dir: Path, delta_dir: Path) -> PipelineRunContext:
    return create_run_context(
        output_table_provider=_provider(delta_dir),
        previous_table_provider=_provider(previous_dir),
        state={},
    )


async def test_no_previous_graph_adds_the_whole_supplied_graph(tmp_path):
    context = _context(tmp_path / "previous", tmp_path / "delta")
    await context.output_table_provider.write_dataframe("entities", _entities("A", "B"))
    await context.output_table_provider.write_dataframe(
        "relationships", _relationships(("A", "B"))
    )

    await run_workflow(get_default_graphrag_config(), context)

    delta = context.state[GRAPH_DELTA_STATE_KEY]
    assert delta.added_edges == [Edge("A", "B", 1.0)]
    assert delta.removed_edges == []
    assert delta.added_vertices == ["A", "B"]
    assert delta.removed_vertices == []


async def test_the_delta_is_the_difference_from_the_previous_graph(tmp_path):
    context = _context(tmp_path / "previous", tmp_path / "delta")
    await context.previous_table_provider.write_dataframe(
        "entities", _entities("A", "B", "C")
    )
    await context.previous_table_provider.write_dataframe(
        "relationships", _relationships(("A", "B"), ("B", "C"))
    )
    await context.output_table_provider.write_dataframe(
        "entities", _entities("A", "B", "C", "D")
    )
    await context.output_table_provider.write_dataframe(
        "relationships", _relationships(("A", "B"), ("C", "D"))
    )

    await run_workflow(get_default_graphrag_config(), context)

    delta = context.state[GRAPH_DELTA_STATE_KEY]
    assert delta.added_edges == [Edge("C", "D", 1.0)]
    assert delta.removed_edges == [Edge("B", "C", 1.0)]
    assert delta.added_vertices == ["D"]
    assert delta.removed_vertices == []
