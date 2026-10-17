# Copyright (C) 2026 luke russo -- ruff complaining

"""Tests for the communities reconcile workflow."""

from typing import cast

import pandas as pd
import pytest
from graphrag.config.enums import CommunityUpdateMode
from graphrag.data_model.schemas import COMMUNITIES_FINAL_COLUMNS
from graphrag.index.workflows.update_communities import run_workflow
from pandas.testing import assert_frame_equal

from tests.unit.graph_utils import (
    cfg,
    leiden_config,
    read_output,
    update_context,
    vdstar_config,
    write_table,
)

TIMESTAMP = "20261012-000000"
PERIOD = "2026-10-12"


def _communities(*groups: tuple[str, ...]) -> pd.DataFrame:
    """A communities frame,  row per member group, ids numbered from zero."""
    return pd.DataFrame(
        [
            {
                "id": f"c{community}",
                "human_readable_id": community,
                "community": community,
                "level": 0,
                "parent": -1,
                "children": [],
                "title": f"Community {community}",
                "entity_ids": [f"e{title}" for title in group],
                "relationship_ids": [],
                "text_unit_ids": [],
                "period": PERIOD,
                "size": len(group),
            }
            for community, group in enumerate(groups)
        ],
        columns=COMMUNITIES_FINAL_COLUMNS,
    )


async def test_an_incremental_update_replaces_the_table_with_ids_unchanged(tmp_path):
    previous = _communities(("A",), ("B",), ("C",))
    delta = _communities(("A", "D"), ("B",))
    write_table(tmp_path, TIMESTAMP, "previous", "communities", previous)
    write_table(tmp_path, TIMESTAMP, "delta", "communities", delta)
    context = update_context(tmp_path, TIMESTAMP)
    config = cfg(
        vdstar_config(update_mode=CommunityUpdateMode.Incremental), base_dir=tmp_path
    )

    await run_workflow(config, context)

    assert_frame_equal(read_output(tmp_path, "communities"), delta, check_dtype=False)
    assert context.state["incremental_update_community_id_mapping"] == {
        0: 0,
        1: 1,
        -1: -1,
    }


@pytest.mark.parametrize(
    "previous", ["absent", "empty"], ids=["no-previous-table", "empty-previous-table"]
)
async def test_additive_replaces_when_there_are_no_previous_communities(
    tmp_path, previous
):
    delta = _communities(("A",), ("B",))
    if previous == "empty":
        write_table(tmp_path, TIMESTAMP, "previous", "communities", _communities())
    write_table(tmp_path, TIMESTAMP, "delta", "communities", delta)

    await run_workflow(
        cfg(leiden_config(), base_dir=tmp_path),
        update_context(tmp_path, TIMESTAMP),
    )

    assert_frame_equal(read_output(tmp_path, "communities"), delta, check_dtype=False)


async def test_an_additive_update_renumbers_the_delta_past_the_old_ids(tmp_path):
    previous = _communities(("A",), ("B",))
    delta = _communities(("C",))
    write_table(tmp_path, TIMESTAMP, "previous", "communities", previous)
    write_table(tmp_path, TIMESTAMP, "delta", "communities", delta)
    context = update_context(tmp_path, TIMESTAMP)

    await run_workflow(cfg(leiden_config(), base_dir=tmp_path), context)

    output = read_output(tmp_path, "communities")
    assert list(output["community"]) == [0, 1, 2]
    assert list(output["entity_ids"]) == [["eA"], ["eB"], ["eC"]]
    assert context.state["incremental_update_community_id_mapping"] == {0: 2, -1: -1}


async def test_an_unhandled_update_mode_raises(tmp_path):
    cluster_config = vdstar_config()
    cluster_config.update_mode = cast("CommunityUpdateMode", "nonsense")

    with pytest.raises(ValueError, match="Unhandled community update mode"):
        await run_workflow(
            cfg(cluster_config, base_dir=tmp_path),
            update_context(tmp_path, TIMESTAMP),
        )
