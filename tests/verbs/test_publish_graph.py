# Copyright (C) 2026 luke russo -- ruff complaining

"""Tests for the publish_graph workflow."""

from graphrag.index.workflows.publish_graph import run_workflow
from pandas.testing import assert_frame_equal

from tests.unit.graph_utils import (
    cfg,
    entities,
    read_output,
    relationships,
    text_units,
    update_context,
    write_table,
)

TIMESTAMP = "20261012-000000"


def _units():
    return text_units("t1").assign(human_readable_id=0, n_tokens=4)


async def test_the_supplied_graph_is_published_to_output(tmp_path):
    entity_table = entities("A", "B")
    relationship_table = relationships(("A", "B"))
    unit_table = _units()
    write_table(tmp_path, TIMESTAMP, "delta", "entities", entity_table)
    write_table(tmp_path, TIMESTAMP, "delta", "relationships", relationship_table)
    write_table(tmp_path, TIMESTAMP, "delta", "text_units", unit_table)

    await run_workflow(cfg(base_dir=tmp_path), update_context(tmp_path, TIMESTAMP))

    assert_frame_equal(
        read_output(tmp_path, "entities"), entity_table, check_dtype=False
    )
    assert_frame_equal(
        read_output(tmp_path, "relationships"), relationship_table, check_dtype=False
    )
    assert_frame_equal(
        read_output(tmp_path, "text_units"), unit_table, check_dtype=False
    )
