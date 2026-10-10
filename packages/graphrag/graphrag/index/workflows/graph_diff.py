# Copyright (C) 2026 luke russo -- ruff complaining

"""A module containing run_workflow method definition."""

import logging

import pandas as pd
from graphrag_storage.tables.table_provider import TableProvider

from graphrag.config.models.graph_rag_config import GraphRagConfig
from graphrag.data_model.schemas import (
    ENTITIES_FINAL_COLUMNS,
    RELATIONSHIPS_FINAL_COLUMNS,
)
from graphrag.index.typing.context import PipelineRunContext
from graphrag.index.typing.workflow import WorkflowFunctionOutput
from graphrag.index.update.relationships import GraphDelta, compute_graph_delta

logger = logging.getLogger(__name__)

GRAPH_DELTA_STATE_KEY = "incremental_update_graph_delta"


async def run_workflow(
    _config: GraphRagConfig,
    context: PipelineRunContext,
) -> WorkflowFunctionOutput:
    """Diff the supplied graph against the previous run's output."""
    logger.info("Workflow started: graph_diff")

    delta = await _diff_against_previous(context)

    context.state[GRAPH_DELTA_STATE_KEY] = delta

    logger.info(
        "Graph delta: %s edges added, %s removed; %s vertices added, %s removed",
        len(delta.added_edges),
        len(delta.removed_edges),
        len(delta.added_vertices),
        len(delta.removed_vertices),
    )
    logger.info("Workflow completed: graph_diff")
    return WorkflowFunctionOutput(
        result={
            "added_edges": len(delta.added_edges),
            "removed_edges": len(delta.removed_edges),
            "added_vertices": len(delta.added_vertices),
            "removed_vertices": len(delta.removed_vertices),
        }
    )


async def _diff_against_previous(context: PipelineRunContext) -> GraphDelta:
    previous = context.previous_table_provider
    supplied = context.output_table_provider

    return compute_graph_delta(
        await _read_or_empty(previous, "relationships", RELATIONSHIPS_FINAL_COLUMNS),
        await supplied.read_dataframe("relationships"),
        await _read_or_empty(previous, "entities", ENTITIES_FINAL_COLUMNS),
        await supplied.read_dataframe("entities"),
    )


async def _read_or_empty(
    provider: TableProvider | None, table: str, columns: list[str]
) -> pd.DataFrame:
    if provider is None or not await provider.has(table):
        return pd.DataFrame(columns=columns)
    return await provider.read_dataframe(table)
