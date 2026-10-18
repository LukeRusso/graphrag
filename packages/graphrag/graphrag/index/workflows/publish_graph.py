# Copyright (C) 2026 luke russo -- ruff complaining


"""A module containing run_workflow method definition."""

import logging

from graphrag_storage.tables.table_provider import TableProvider

from graphrag.config.models.graph_rag_config import GraphRagConfig
from graphrag.data_model.data_reader import DataReader
from graphrag.index.run.utils import get_update_table_providers
from graphrag.index.typing.context import PipelineRunContext
from graphrag.index.typing.workflow import WorkflowFunctionOutput

logger = logging.getLogger(__name__)


async def run_workflow(
    config: GraphRagConfig,
    context: PipelineRunContext,
) -> WorkflowFunctionOutput:
    """Publish the supplied graph to the output location."""
    logger.info("Workflow started: publish_graph")
    output_table_provider, _, delta_table_provider = get_update_table_providers(
        config, context.state["update_timestamp"]
    )

    await _publish_graph(delta_table_provider, output_table_provider)

    logger.info("Workflow completed: publish_graph")
    return WorkflowFunctionOutput(result=None)


async def _publish_graph(
    delta_table_provider: TableProvider,
    output_table_provider: TableProvider,
) -> None:
    """Write the delta's graph tables to the output."""
    reader = DataReader(delta_table_provider)
    entities = await reader.entities()
    relationships = await reader.relationships()
    text_units = await reader.text_units()

    await output_table_provider.write_dataframe("entities", entities)
    await output_table_provider.write_dataframe("relationships", relationships)
    await output_table_provider.write_dataframe("text_units", text_units)
