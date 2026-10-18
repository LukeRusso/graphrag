# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A module containing run_workflow method definition."""

import logging

import pandas as pd
from graphrag_storage.tables.table_provider import TableProvider

from graphrag.config.enums import CommunityUpdateMode
from graphrag.config.models.graph_rag_config import GraphRagConfig
from graphrag.data_model.data_reader import DataReader
from graphrag.data_model.schemas import COMMUNITY_REPORTS_FINAL_COLUMNS
from graphrag.index.operations.cluster_graph import UNHANDLED_UPDATE_MODE_MSG
from graphrag.index.run.utils import get_update_table_providers
from graphrag.index.typing.context import PipelineRunContext
from graphrag.index.typing.workflow import WorkflowFunctionOutput
from graphrag.index.update.communities import _update_and_merge_community_reports
from graphrag.index.workflows.create_community_reports import (
    REUSED_REPORTS_STATE_KEY,
)
from graphrag.index.workflows.graph_diff import GRAPH_DELTA_STATE_KEY

logger = logging.getLogger(__name__)


async def run_workflow(
    config: GraphRagConfig,
    context: PipelineRunContext,
) -> WorkflowFunctionOutput:
    """Update the community reports from a incremental index run."""
    logger.info("Workflow started: update_community_reports")
    output_table_provider, previous_table_provider, delta_table_provider = (
        get_update_table_providers(config, context.state["update_timestamp"])
    )

    community_id_mapping = context.state["incremental_update_community_id_mapping"]
    reused_reports_map = context.state.get(REUSED_REPORTS_STATE_KEY, {})

    merged_community_reports = await _update_community_reports(
        previous_table_provider,
        delta_table_provider,
        output_table_provider,
        community_id_mapping,
        reused_reports_map,
        _keep_previous_reports(context, config),
    )

    context.state["incremental_update_merged_community_reports"] = (
        merged_community_reports
    )

    logger.info("Workflow completed: update_community_reports")
    return WorkflowFunctionOutput(result=None)


def _keep_previous_reports(context: PipelineRunContext, config: GraphRagConfig) -> bool:
    if GRAPH_DELTA_STATE_KEY not in context.state:
        return True

    match config.cluster_graph.update_mode:
        case CommunityUpdateMode.Additive:
            return True
        case CommunityUpdateMode.Incremental:
            return False
        case _:
            error_msg = (
                f"{UNHANDLED_UPDATE_MODE_MSG}: {config.cluster_graph.update_mode}"
            )
            raise ValueError(error_msg)


async def _update_community_reports(
    previous_table_provider: TableProvider,
    delta_table_provider: TableProvider,
    output_table_provider: TableProvider,
    community_id_mapping: dict,
    reused_reports_map: dict[int, int],
    keep_previous_reports: bool,
) -> pd.DataFrame:
    """Update the community reports output."""
    old_community_reports = (
        await DataReader(previous_table_provider).community_reports()
        if await previous_table_provider.has("community_reports")
        else pd.DataFrame(columns=COMMUNITY_REPORTS_FINAL_COLUMNS)
    )
    delta_community_reports = await DataReader(delta_table_provider).community_reports()
    communities = await DataReader(output_table_provider).communities()
    parent_by_community = dict(
        zip(
            communities["community"].astype(int),
            communities["parent"].astype(int),
            strict=True,
        )
    )
    merged_community_reports = _update_and_merge_community_reports(
        old_community_reports,
        delta_community_reports,
        community_id_mapping,
        reused_reports_map,
        keep_previous_reports,
        parent_by_community,
    )

    await output_table_provider.write_dataframe(
        "community_reports", merged_community_reports
    )

    return merged_community_reports
