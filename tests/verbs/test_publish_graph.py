# Copyright (C) 2026 luke russo -- ruff complaining

"""Tests for the publish_graph workflow."""

from graphrag.index.workflows.publish_graph import run_workflow
from pandas.testing import assert_frame_equal

from tests.unit.graph_utils import (
    cfg,
    entities,
    read_output,
    relationships,
    update_context,
    write_table,
)

TIMESTAMP = "20261012-000000"


async def test_the_supplied_graph_is_published_to_output(tmp_path):
    entity_table = entities("A", "B")
    relationship_table = relationships(("A", "B"))
    write_table(tmp_path, TIMESTAMP, "delta", "entities", entity_table)
    write_table(tmp_path, TIMESTAMP, "delta", "relationships", relationship_table)

    await run_workflow(cfg(base_dir=tmp_path), update_context(tmp_path, TIMESTAMP))

    assert_frame_equal(
        read_output(tmp_path, "entities"), entity_table, check_dtype=False
    )
    assert_frame_equal(
        read_output(tmp_path, "relationships"), relationship_table, check_dtype=False
    )


async def test_an_edge_with_a_missing_endpoint_is_dropped(tmp_path):
    write_table(tmp_path, TIMESTAMP, "delta", "entities", entities("A", "B"))
    write_table(
        tmp_path,
        TIMESTAMP,
        "delta",
        "relationships",
        relationships(("A", "B"), ("A", "Z")),
    )

    await run_workflow(cfg(base_dir=tmp_path), update_context(tmp_path, TIMESTAMP))

    edges = read_output(tmp_path, "relationships")[["source", "target"]]
    assert list(edges.itertuples(index=False, name=None)) == [("A", "B")]
