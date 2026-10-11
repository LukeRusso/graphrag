# Copyright (C) 2026 luke russo -- ruff complaining

"""Tests for the create_community_reports_text workflow."""

import pandas as pd
from graphrag.data_model.schemas import (
    COMMUNITIES_FINAL_COLUMNS,
    COMMUNITY_REPORTS_FINAL_COLUMNS,
)
from graphrag.index.run.utils import create_run_context
from graphrag.index.workflows.create_community_reports_text import run_workflow
from graphrag_storage.memory_storage import MemoryStorage
from graphrag_storage.tables.parquet_table_provider import ParquetTableProvider

from tests.unit.graph_utils import cfg, entities, leiden_config


async def test_no_communities_writes_an_empty_reports_table():
    provider = ParquetTableProvider(storage=MemoryStorage())
    await provider.write_dataframe("entities", entities("A"))
    await provider.write_dataframe(
        "communities", pd.DataFrame(columns=COMMUNITIES_FINAL_COLUMNS)
    )
    context = create_run_context(output_table_provider=provider)

    await run_workflow(cfg(leiden_config()), context)

    reports = await provider.read_dataframe("community_reports")
    assert reports.empty
    assert list(reports.columns) == COMMUNITY_REPORTS_FINAL_COLUMNS
