# Copyright (C) 2026 luke russo -- ruff complaining

"""A module containing run_workflow method definition."""

import io
import logging
from typing import Any

import pandas as pd
import pyarrow.parquet as pq
from graphrag_storage import Storage, create_storage

from graphrag.config.models.graph_rag_config import GraphRagConfig
from graphrag.data_model.schemas import (
    ENTITIES_FINAL_COLUMNS,
    RELATIONSHIPS_FINAL_COLUMNS,
)
from graphrag.index.typing.context import PipelineRunContext
from graphrag.index.typing.workflow import WorkflowFunctionOutput

logger = logging.getLogger(__name__)

DEFAULT_EDGE_WEIGHT = 1.0
SAMPLE_SIZE = 5


async def run_workflow(
    config: GraphRagConfig,
    context: PipelineRunContext,
) -> WorkflowFunctionOutput:
    """Load a supplied graph as the run's entities and relationships tables."""
    logger.info("Workflow started: load_graph")

    graph_storage = create_storage(config.graph_input_storage)

    entities = _normalize_entities(
        await _read_table(graph_storage, config.graph_input.entities_file)
    )
    relationships = _normalize_relationships(
        await _read_table(graph_storage, config.graph_input.relationships_file)
    )

    await context.output_table_provider.write_dataframe("entities", entities)
    await context.output_table_provider.write_dataframe("relationships", relationships)

    logger.info(
        "Supplied graph loaded: %s entities, %s relationships",
        len(entities),
        len(relationships),
    )
    logger.info("Workflow completed: load_graph")
    return WorkflowFunctionOutput(
        result={
            "entities": entities.head(SAMPLE_SIZE).to_dict("records"),
            "relationships": relationships.head(SAMPLE_SIZE).to_dict("records"),
        }
    )


async def _read_table(storage: Storage, path: str) -> pd.DataFrame:
    raw = await storage.get(path, as_bytes=True)

    if not raw:
        msg = f"Cannot load the supplied graph: '{path}' was not found."
        raise ValueError(msg)

    return pq.read_table(io.BytesIO(raw)).to_pandas()


def _normalize_entities(entities: pd.DataFrame) -> pd.DataFrame:
    if "title" not in entities.columns:
        msg = (
            "The supplied entities table has no 'title' column; entities, "
            "relationships and communities all join on it."
        )
        raise ValueError(msg)

    normalized = _select_columns(entities, ENTITIES_FINAL_COLUMNS)
    # create_communities explodes this, so it has to be a list on every row
    normalized["text_unit_ids"] = _fill_missing_list(normalized["text_unit_ids"])
    return normalized


def _normalize_relationships(relationships: pd.DataFrame) -> pd.DataFrame:
    missing = [name for name in ("source", "target") if name not in relationships]

    if missing:
        msg = (
            "The supplied relationships table is missing "
            f"{', '.join(repr(name) for name in missing)}; both endpoints are "
            "required and must match entity titles."
        )
        raise ValueError(msg)

    normalized = _select_columns(relationships, RELATIONSHIPS_FINAL_COLUMNS)
    normalized["text_unit_ids"] = _fill_missing_list(normalized["text_unit_ids"])
    # a missing weight would reach the clusterer as NaN
    normalized["weight"] = _fill_missing_weight(normalized["weight"])
    return normalized


def _select_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    selected = pd.DataFrame(index=df.index)

    for column in columns:
        selected[column] = df[column] if column in df.columns else None

    return selected


def _is_missing(value: Any) -> bool:
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        # array-like, so isnan isn't a scalar question
        return False


def _fill_missing_list(series: pd.Series) -> pd.Series:
    return series.map(lambda value: [] if _is_missing(value) else value)


def _fill_missing_weight(series: pd.Series) -> pd.Series:
    return series.map(
        lambda value: DEFAULT_EDGE_WEIGHT if _is_missing(value) else value
    )
