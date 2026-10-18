# Copyright (C) 2026 luke russo -- ruff complaining

"""Tests for the community reports reconcile workflow."""

import pandas as pd
from graphrag.config.enums import CommunityUpdateMode
from graphrag.data_model.schemas import COMMUNITY_REPORTS_FINAL_COLUMNS
from graphrag.index.workflows.create_community_reports import (
    REUSED_REPORTS_STATE_KEY,
)
from graphrag.index.workflows.update_community_reports import run_workflow

from tests.unit.graph_utils import (
    PERIOD,
    cfg,
    communities,
    read_output,
    update_context,
    vdstar_config,
    write_table,
)

TIMESTAMP = "20261012-000000"


def _write_reports(tmp_path, stage: str, *rows: tuple[int, str]) -> None:
    """Write a community reports table into one of the run's staging directories."""
    reports = pd.DataFrame(
        [
            {
                "id": f"r{community}",
                "human_readable_id": community,
                "community": community,
                "level": 0,
                "parent": -1,
                "children": [],
                "title": f"Community {community}",
                "summary": summary,
                "full_content": summary,
                "rank": 5.0,
                "rating_explanation": "Because.",
                "findings": [],
                "full_content_json": None,
                "period": PERIOD,
                "size": 2,
            }
            for community, summary in rows
        ],
        columns=COMMUNITY_REPORTS_FINAL_COLUMNS,
    )
    write_table(tmp_path, TIMESTAMP, stage, "community_reports", reports)


async def test_an_additive_update_keeps_every_previous_report(tmp_path):
    _write_reports(tmp_path, "previous", (0, "old zero"), (1, "old one"))
    _write_reports(tmp_path, "delta", (0, "new three"))
    context = update_context(tmp_path, TIMESTAMP)
    context.state["incremental_update_community_id_mapping"] = {0: 3, -1: -1}
    await context.output_table_provider.write_dataframe(
        "communities", communities(("A", "B"), ("C", "D"), ("E", "F"))
    )
    config = cfg(
        vdstar_config(update_mode=CommunityUpdateMode.Additive), base_dir=tmp_path
    )

    await run_workflow(config, context)

    output = read_output(tmp_path, "community_reports")
    assert dict(zip(output["community"], output["summary"], strict=True)) == {
        0: "old zero",
        1: "old one",
        3: "new three",
    }
    assert (output["human_readable_id"] == output["community"]).all()


async def test_an_incremental_update_keeps_only_the_reused_report(tmp_path):
    _write_reports(tmp_path, "previous", (0, "old zero"), (1, "old one"))
    _write_reports(tmp_path, "delta", (0, "new zero"))
    context = update_context(tmp_path, TIMESTAMP)
    context.state["incremental_update_community_id_mapping"] = {0: 0, 1: 1, -1: -1}
    context.state[REUSED_REPORTS_STATE_KEY] = {1: 1}
    await context.output_table_provider.write_dataframe(
        "communities", communities(("A", "B"), ("C", "D"), parents=[-1, 0])
    )
    config = cfg(
        vdstar_config(update_mode=CommunityUpdateMode.Incremental), base_dir=tmp_path
    )

    await run_workflow(config, context)

    output = read_output(tmp_path, "community_reports")
    assert dict(zip(output["community"], output["summary"], strict=True)) == {
        0: "new zero",
        1: "old one",
    }
    assert dict(zip(output["community"], output["parent"], strict=True)) == {
        0: -1,
        1: 0,
    }


async def test_an_incremental_update_with_nothing_reused_writes_the_delta(tmp_path):
    _write_reports(tmp_path, "previous", (0, "old zero"))
    _write_reports(tmp_path, "delta", (0, "new zero"), (1, "new one"))
    context = update_context(tmp_path, TIMESTAMP)
    context.state["incremental_update_community_id_mapping"] = {0: 0, 1: 1, -1: -1}
    await context.output_table_provider.write_dataframe(
        "communities", communities(("A", "B"), ("C", "D"))
    )
    config = cfg(
        vdstar_config(update_mode=CommunityUpdateMode.Incremental), base_dir=tmp_path
    )

    await run_workflow(config, context)

    output = read_output(tmp_path, "community_reports")
    assert dict(zip(output["community"], output["summary"], strict=True)) == {
        0: "new zero",
        1: "new one",
    }
