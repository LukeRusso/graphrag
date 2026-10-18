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
    TEXT_UNITS_FINAL_COLUMNS,
)
from graphrag.index.typing.context import PipelineRunContext
from graphrag.index.typing.workflow import WorkflowFunctionOutput
from graphrag.tokenizer.get_tokenizer import get_tokenizer

logger = logging.getLogger(__name__)

DEFAULT_EDGE_WEIGHT = 1.0
SAMPLE_SIZE = 5


async def run_workflow(
    config: GraphRagConfig,
    context: PipelineRunContext,
) -> WorkflowFunctionOutput:
    """Load a supplied graph as the run's graph tables."""
    logger.info("Workflow started: load_graph")

    graph_storage = create_storage(config.graph_input_storage)

    entities = _normalize_entities(
        await _read_table(graph_storage, config.graph_input.entities_file)
    )
    relationships = _normalize_relationships(
        await _read_table(graph_storage, config.graph_input.relationships_file)
    )
    text_units = _normalize_text_units(
        await _read_table(graph_storage, config.graph_input.text_units_file), config
    )

    _validate_graph(entities, relationships, text_units, config)

    await context.output_table_provider.write_dataframe("entities", entities)
    await context.output_table_provider.write_dataframe("relationships", relationships)
    await context.output_table_provider.write_dataframe("text_units", text_units)

    logger.info(
        "Supplied graph loaded: %s entities, %s relationships, %s text units",
        len(entities),
        len(relationships),
        len(text_units),
    )
    logger.info("Workflow completed: load_graph")
    return WorkflowFunctionOutput(
        result={
            "entities": entities.head(SAMPLE_SIZE).to_dict("records"),
            "relationships": relationships.head(SAMPLE_SIZE).to_dict("records"),
            "text_units": text_units.head(SAMPLE_SIZE).to_dict("records"),
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
            f"{', '.join(missing)}; both endpoints are "
            "required and must match entity titles."
        )
        raise ValueError(msg)

    normalized = _select_columns(relationships, RELATIONSHIPS_FINAL_COLUMNS)
    normalized["text_unit_ids"] = _fill_missing_list(normalized["text_unit_ids"])
    # a missing weight would reach the clusterer as NaN
    normalized["weight"] = _fill_missing_weight(normalized["weight"])
    return normalized


def _normalize_text_units(
    text_units: pd.DataFrame, config: GraphRagConfig
) -> pd.DataFrame:
    missing = [name for name in ("id", "text") if name not in text_units]

    if missing:
        msg = (
            "The supplied text units table is missing "
            f"{', '.join(missing)}; a unit is identified by "
            "its id and quoted by its text."
        )
        raise ValueError(msg)

    if text_units["text"].isna().any():
        msg = f"The supplied text units table in '{config.graph_input.text_units_file}' has a row with no 'text'."
        raise ValueError(msg)

    normalized = _select_columns(text_units, TEXT_UNITS_FINAL_COLUMNS)
    tokenizer = get_tokenizer(encoding_model=config.chunking.encoding_model)
    normalized["human_readable_id"] = range(len(normalized))
    normalized["n_tokens"] = [
        len(tokenizer.encode(text)) for text in normalized["text"].astype(str)
    ]

    for column in ("entity_ids", "relationship_ids", "covariate_ids"):
        normalized[column] = _fill_missing_list(normalized[column])
    return normalized


def _validate_graph(
    entities: pd.DataFrame,
    relationships: pd.DataFrame,
    text_units: pd.DataFrame,
    config: GraphRagConfig,
) -> None:
    entities_file = config.graph_input.entities_file
    relationships_file = config.graph_input.relationships_file
    text_units_file = config.graph_input.text_units_file

    duplicated_title = entities["title"][entities["title"].duplicated()]
    if not duplicated_title.empty:
        msg = (
            f"Duplicate entity title {duplicated_title.iloc[0]!r} in '{entities_file}'."
        )
        raise ValueError(msg)

    duplicated_unit = text_units["id"][text_units["id"].duplicated()]
    if not duplicated_unit.empty:
        msg = f"Duplicate text unit id {duplicated_unit.iloc[0]!r} in '{text_units_file}'."
        raise ValueError(msg)

    unit_ids = set(text_units["id"])
    _check_text_units(
        entities["title"],
        entities["text_unit_ids"],
        unit_ids,
        entities_file,
        text_units_file,
        "Entity",
    )
    _check_text_units(
        relationships["source"].astype(str)
        + " -> "
        + relationships["target"].astype(str),
        relationships["text_unit_ids"],
        unit_ids,
        relationships_file,
        text_units_file,
        "Relationship",
    )

    endpoints = set(relationships["source"]) | set(relationships["target"])
    dangling = sorted(endpoints - set(entities["title"]))
    if dangling:
        msg = (
            f"The supplied relationships table names {dangling[0]!r}, which is not "
            f"an entity title in '{entities_file}'."
        )
        raise ValueError(msg)

    referenced: set[Any] = set()
    for ids in [*entities["text_unit_ids"], *relationships["text_unit_ids"]]:
        referenced.update(ids)

    unclaimed = len(unit_ids - referenced)
    if unclaimed:
        logger.info(
            "%s supplied text units are named by no entity or relationship.",
            unclaimed,
        )


def _check_text_units(
    labels: pd.Series,
    references: pd.Series,
    unit_ids: set[Any],
    table_file: str,
    units_file: str,
    label: str,
) -> None:
    for name, referenced in zip(labels, references, strict=True):
        if len(referenced) == 0:
            msg = f"{label} {name!r} names no text unit in '{table_file}'."
            raise ValueError(msg)

        missing = [unit for unit in referenced if unit not in unit_ids]
        if missing:
            msg = (
                f"{label} {name!r} names text unit {missing[0]!r}, which is not in "
                f"'{units_file}'."
            )
            raise ValueError(msg)


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
