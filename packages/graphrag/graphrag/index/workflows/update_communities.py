# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A module containing run_workflow method definition."""

import logging

import pandas as pd
from graphrag_storage.tables.table_provider import TableProvider

from graphrag.config.enums import CommunityUpdateMode
from graphrag.config.models.graph_rag_config import GraphRagConfig
from graphrag.data_model.data_reader import DataReader
from graphrag.index.operations.cluster_graph import UNHANDLED_UPDATE_MODE_MSG
from graphrag.index.run.utils import get_update_table_providers
from graphrag.index.typing.context import PipelineRunContext
from graphrag.index.typing.workflow import WorkflowFunctionOutput
from graphrag.index.update.communities import _update_and_merge_communities

logger = logging.getLogger(__name__)


async def run_workflow(
    config: GraphRagConfig,
    context: PipelineRunContext,
) -> WorkflowFunctionOutput:
    """Update the communities from a incremental index run."""
    logger.info("Workflow started: update_communities")
    output_table_provider, previous_table_provider, delta_table_provider = (
        get_update_table_providers(config, context.state["update_timestamp"])
    )

    community_id_mapping = await _update_communities(
        config, previous_table_provider, delta_table_provider, output_table_provider
    )

    context.state["incremental_update_community_id_mapping"] = community_id_mapping

    logger.info("Workflow completed: update_communities")
    return WorkflowFunctionOutput(result=None)


async def _update_communities(
    config: GraphRagConfig,
    previous_table_provider: TableProvider,
    delta_table_provider: TableProvider,
    output_table_provider: TableProvider,
) -> dict:
    """Update the communities output."""
    match config.cluster_graph.update_mode:
        case CommunityUpdateMode.Incremental:
            return await _replace_communities(
                delta_table_provider, output_table_provider
            )
        case CommunityUpdateMode.Additive:
            old_communities = await _read_previous_communities(
                previous_table_provider
            )
            if old_communities.empty:
                return await _replace_communities(
                    delta_table_provider, output_table_provider
                )
            delta_communities = await DataReader(delta_table_provider).communities()
            merged_communities, community_id_mapping = _update_and_merge_communities(
                old_communities, delta_communities
            )
            await output_table_provider.write_dataframe(
                "communities", merged_communities
            )
            return community_id_mapping
        case _:
            error_msg = (
                f"{UNHANDLED_UPDATE_MODE_MSG}: {config.cluster_graph.update_mode}"
            )
            raise ValueError(error_msg)


async def _read_previous_communities(
    previous_table_provider: TableProvider,
) -> pd.DataFrame:
    """Return the previous communities, or an empty frame when there are none."""
    if not await previous_table_provider.has("communities"):
        return pd.DataFrame()
    return await DataReader(previous_table_provider).communities()


async def _replace_communities(
    delta_table_provider: TableProvider,
    output_table_provider: TableProvider,
) -> dict:
    """Promote the delta communities to the output, ids unchanged."""
    communities = await DataReader(delta_table_provider).communities()
    await output_table_provider.write_dataframe("communities", communities)
    community_id_mapping = {
        int(community_id): int(community_id)
        for community_id in communities["community"].dropna()
    }
    community_id_mapping.update({-1: -1})
    return community_id_mapping
