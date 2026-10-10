# Copyright (C) 2026 luke russo -- ruff complaining

import asyncio
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from graphrag.config.models.graph_rag_config import GraphRagConfig
from graphrag.data_model.schemas import (
    ENTITIES_FINAL_COLUMNS,
    RELATIONSHIPS_FINAL_COLUMNS,
)
from graphrag.index.run.utils import create_run_context
from graphrag.index.workflows.load_graph import (
    DEFAULT_EDGE_WEIGHT,
    run_workflow,
)


def _entities() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "id": "e1",
            "human_readable_id": 0,
            "title": "ALICE",
            "type": "PERSON",
            "description": "Alice is a person.",
            "text_unit_ids": ["t1"],
            "frequency": 1,
            "degree": 1,
        },
        {
            "id": "e2",
            "human_readable_id": 1,
            "title": "BOB",
            "type": "PERSON",
            "description": "Bob is a person.",
            "text_unit_ids": ["t1"],
            "frequency": 1,
            "degree": 1,
        },
    ])


def _relationships() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "id": "r1",
            "human_readable_id": 0,
            "source": "ALICE",
            "target": "BOB",
            "description": "Alice knows Bob.",
            "weight": 3.0,
            "combined_degree": 2,
            "text_unit_ids": ["t1"],
        }
    ])


def _write_graph(
    tmp_path: Path,
    entities: pd.DataFrame | None,
    relationships: pd.DataFrame | None,
) -> GraphRagConfig:
    if entities is not None:
        entities.to_parquet(tmp_path / "entities.parquet", index=False)
    if relationships is not None:
        relationships.to_parquet(tmp_path / "relationships.parquet", index=False)

    return GraphRagConfig(
        graph_input_storage={"type": "file", "base_dir": str(tmp_path)}
    )


def _list_cell(value: Any) -> list | None:
    # an empty list comes back as an empty ndarray, a null cell as None
    return None if value is None else list(value)


def _run(config: GraphRagConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    context = create_run_context()

    async def _go() -> tuple[pd.DataFrame, pd.DataFrame]:
        await run_workflow(config, context)
        provider = context.output_table_provider
        return (
            await provider.read_dataframe("entities"),
            await provider.read_dataframe("relationships"),
        )

    return asyncio.run(_go())


def test_supplied_graph_lands_intact_in_final_column_order(tmp_path):
    config = _write_graph(tmp_path, _entities(), _relationships())

    entities, relationships = _run(config)

    assert list(entities.columns) == ENTITIES_FINAL_COLUMNS
    assert list(relationships.columns) == RELATIONSHIPS_FINAL_COLUMNS
    assert entities["title"].tolist() == ["ALICE", "BOB"]
    assert entities["description"].tolist() == [
        "Alice is a person.",
        "Bob is a person.",
    ]
    assert relationships["source"].tolist() == ["ALICE"]
    assert relationships["target"].tolist() == ["BOB"]
    assert relationships["weight"].tolist() == [3.0]


def test_absent_columns_are_left_null_rather_than_missing(tmp_path):
    slim_entities = _entities()[["title", "description"]]
    config = _write_graph(tmp_path, slim_entities, _relationships())

    entities, _ = _run(config)

    assert list(entities.columns) == ENTITIES_FINAL_COLUMNS
    assert entities["title"].tolist() == ["ALICE", "BOB"]
    for column in ("id", "human_readable_id", "type", "frequency", "degree"):
        assert entities[column].isna().all(), column


def test_absent_text_unit_ids_becomes_empty_list(tmp_path):
    entities = _entities().drop(columns=["text_unit_ids"])
    relationships = _relationships().drop(columns=["text_unit_ids"])
    config = _write_graph(tmp_path, entities, relationships)

    entities_out, relationships_out = _run(config)

    assert [_list_cell(v) for v in entities_out["text_unit_ids"]] == [[], []]
    assert [_list_cell(v) for v in relationships_out["text_unit_ids"]] == [[]]


def test_absent_weight_defaults(tmp_path):
    relationships = _relationships().drop(columns=["weight"])
    config = _write_graph(tmp_path, _entities(), relationships)

    _, relationships_out = _run(config)

    assert relationships_out["weight"].tolist() == [DEFAULT_EDGE_WEIGHT]


def test_missing_supplied_table_names_the_path(tmp_path):
    config = _write_graph(tmp_path, _entities(), None)

    with pytest.raises(ValueError, match=r"relationships\.parquet"):
        _run(config)


def test_entities_without_title_raise(tmp_path):
    entities = _entities().drop(columns=["title"])
    config = _write_graph(tmp_path, entities, _relationships())

    with pytest.raises(ValueError, match=r"no 'title' column"):
        _run(config)


def test_relationships_without_an_endpoint_raise(tmp_path):
    relationships = _relationships().drop(columns=["target"])
    config = _write_graph(tmp_path, _entities(), relationships)

    with pytest.raises(ValueError, match=r"missing 'target'"):
        _run(config)
