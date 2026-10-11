# Copyright (C) 2026 Microsoft

"""Tests for the create_communities pure function.

These tests pin down the behavior of the create_communities function
independently of the workflow runner, so that refactoring (vectorizing
the per-level loop, streaming entity reads, streaming writes, etc.)
can be verified against known output.
"""

import asyncio
import uuid
from typing import Any

import numpy as np
import pandas as pd
import pytest
from graphrag.config.models.cluster_graph_config import ClusterGraphConfig
from graphrag.data_model.schemas import COMMUNITIES_FINAL_COLUMNS
from graphrag.index.workflows.create_communities import (
    _sanitize_row,
    create_communities,
)
from graphrag_storage.memory_storage import MemoryStorage
from graphrag_storage.tables.csv_table import CSVTable
from graphrag_storage.tables.parquet_table_provider import ParquetTableProvider
from graphrag_storage.tables.table import Table

from tests.unit.graph_utils import cfg, leiden_config


class FakeTable(CSVTable):
    """In-memory table that collects written rows for test assertions."""

    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    async def write(self, row: dict[str, Any]) -> None:
        """Append a row to the in-memory store."""
        self.rows.append(row)


class FakeEntitiesTable(Table):
    """In-memory read-only table that supports async iteration."""

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows
        self._index = 0

    def __aiter__(self):
        """Return an async iterator over the rows."""
        self._index = 0
        return self

    async def __anext__(self) -> dict[str, Any]:
        """Yield the next row or stop."""
        if self._index >= len(self._rows):
            raise StopAsyncIteration
        row = self._rows[self._index]
        self._index += 1
        return row

    async def length(self) -> int:
        """Return number of rows."""
        return len(self._rows)

    async def has(self, row_id: str) -> bool:
        """Check if a row with the given ID exists."""
        return any(r.get("id") == row_id for r in self._rows)

    async def write(self, row: dict[str, Any]) -> None:
        """Not supported for read-only table."""
        raise NotImplementedError

    async def close(self) -> None:
        """No-op."""


async def _run_create_communities(
    title_to_entity_id: dict[str, str],
    relationships: pd.DataFrame,
    graph_cluster_config: ClusterGraphConfig,
) -> pd.DataFrame:
    """Helper that runs create_communities with fake tables and returns all rows as a DataFrame."""
    communities_table = FakeTable()
    entity_rows = [
        {"id": eid, "title": title} for title, eid in title_to_entity_id.items()
    ]
    entities_table = FakeEntitiesTable(entity_rows)
    await create_communities(
        communities_table,
        entities_table,
        relationships,
        cfg(graph_cluster_config),
        ParquetTableProvider(storage=MemoryStorage()),
    )
    return pd.DataFrame(communities_table.rows)


def _make_title_to_entity_id(
    rows: list[tuple[str, str]],
) -> dict[str, str]:
    """Build a title-to-entity-id mapping from (id, title) pairs."""
    return {title: eid for eid, title in rows}


def _make_relationships(
    rows: list[tuple[str, str, str, float, list[str]]],
) -> pd.DataFrame:
    """Build a minimal relationships DataFrame.

    Each row is (id, source, target, weight, text_unit_ids).
    """
    return pd.DataFrame([
        {
            "id": rid,
            "source": src,
            "target": tgt,
            "weight": w,
            "text_unit_ids": tuids,
            "human_readable_id": i,
        }
        for i, (rid, src, tgt, w, tuids) in enumerate(rows)
    ])


def _two_triangles() -> tuple[dict[str, str], pd.DataFrame]:
    """Two disconnected triangles: {A,B,C} and {D,E,F}."""
    title_to_entity_id = _make_title_to_entity_id([
        ("e1", "A"),
        ("e2", "B"),
        ("e3", "C"),
        ("e4", "D"),
        ("e5", "E"),
        ("e6", "F"),
    ])
    relationships = _make_relationships([
        ("r1", "A", "B", 1.0, ["t1"]),
        ("r2", "A", "C", 1.0, ["t1", "t2"]),
        ("r3", "B", "C", 1.0, ["t2"]),
        ("r4", "D", "E", 1.0, ["t3"]),
        ("r5", "D", "F", 1.0, ["t3", "t4"]),
        ("r6", "E", "F", 1.0, ["t4"]),
    ])
    return title_to_entity_id, relationships


@pytest.fixture(scope="module")
def two_triangle_result() -> pd.DataFrame:
    """One run of the operation over the two-triangle fixture."""
    title_to_entity_id, relationships = _two_triangles()
    return asyncio.run(
        _run_create_communities(title_to_entity_id, relationships, leiden_config())
    )


# -------------------------------------------------------------------
# Column schema
# -------------------------------------------------------------------


class TestOutputSchema:
    """Verify the output DataFrame has the expected column schema."""

    def test_has_all_final_columns(self, two_triangle_result: pd.DataFrame):
        """Output must have exactly the COMMUNITIES_FINAL_COLUMNS."""
        assert list(two_triangle_result.columns) == COMMUNITIES_FINAL_COLUMNS


# -------------------------------------------------------------------
# Metadata fields
# -------------------------------------------------------------------


class TestMetadataFields:
    """Verify computed metadata fields like id, title, size, period."""

    def test_uuid_ids(self, two_triangle_result: pd.DataFrame):
        """Each community id should be a valid UUID4."""
        for _, row in two_triangle_result.iterrows():
            parsed = uuid.UUID(row["id"])
            assert parsed.version == 4

    def test_title_format(self, two_triangle_result: pd.DataFrame):
        """Title should be 'Community N' where N is the community id."""
        for _, row in two_triangle_result.iterrows():
            assert row["title"] == f"Community {row['community']}"

    def test_size_equals_entity_count(self, two_triangle_result: pd.DataFrame):
        """size should equal the length of entity_ids."""
        for _, row in two_triangle_result.iterrows():
            assert row["size"] == len(row["entity_ids"])

    def test_period_is_iso_date(self, two_triangle_result: pd.DataFrame):
        """period should be a valid ISO date string."""
        from datetime import date

        for _, row in two_triangle_result.iterrows():
            date.fromisoformat(row["period"])


# -------------------------------------------------------------------
# Entity aggregation
# -------------------------------------------------------------------


class TestEntityAggregation:
    """Verify that entity_ids are correctly aggregated per community."""

    def test_entity_ids_per_community(self, two_triangle_result: pd.DataFrame):
        """Each community should contain exactly the entities matching
        its cluster nodes."""
        comm_0 = two_triangle_result[two_triangle_result["community"] == 0].iloc[0]
        comm_1 = two_triangle_result[two_triangle_result["community"] == 1].iloc[0]

        assert sorted(comm_0["entity_ids"]) == ["e1", "e2", "e3"]
        assert sorted(comm_1["entity_ids"]) == ["e4", "e5", "e6"]

    def test_entity_ids_are_lists(self, two_triangle_result: pd.DataFrame):
        """entity_ids should be Python lists, not numpy arrays."""
        for _, row in two_triangle_result.iterrows():
            assert isinstance(row["entity_ids"], list)


# -------------------------------------------------------------------
# Relationship and text_unit aggregation
# -------------------------------------------------------------------


class TestRelationshipAggregation:
    """Verify that relationship_ids and text_unit_ids are correctly
    aggregated (intra-community only) and deduplicated."""

    def test_relationship_ids_per_community(self, two_triangle_result: pd.DataFrame):
        """Each community should only include relationships where both
        endpoints are in the same community."""
        comm_0 = two_triangle_result[two_triangle_result["community"] == 0].iloc[0]
        comm_1 = two_triangle_result[two_triangle_result["community"] == 1].iloc[0]

        assert sorted(comm_0["relationship_ids"]) == ["r1", "r2", "r3"]
        assert sorted(comm_1["relationship_ids"]) == ["r4", "r5", "r6"]

    def test_text_unit_ids_per_community(self, two_triangle_result: pd.DataFrame):
        """text_unit_ids should be the deduplicated union of text units
        from the community's intra-community relationships."""
        comm_0 = two_triangle_result[two_triangle_result["community"] == 0].iloc[0]
        comm_1 = two_triangle_result[two_triangle_result["community"] == 1].iloc[0]

        assert sorted(comm_0["text_unit_ids"]) == ["t1", "t2"]
        assert sorted(comm_1["text_unit_ids"]) == ["t3", "t4"]

    async def test_cross_community_relationships_excluded(self):
        """A relationship spanning two communities must not appear in
        either community's relationship_ids."""
        title_to_entity_id = _make_title_to_entity_id([
            ("e1", "A"),
            ("e2", "B"),
            ("e3", "C"),
            ("e4", "D"),
            ("e5", "E"),
            ("e6", "F"),
        ])
        relationships = _make_relationships([
            ("r1", "A", "B", 1.0, ["t1"]),
            ("r2", "A", "C", 1.0, ["t1"]),
            ("r3", "B", "C", 1.0, ["t1"]),
            ("r_cross", "C", "D", 0.1, ["t_cross"]),
            ("r4", "D", "E", 1.0, ["t2"]),
            ("r5", "D", "F", 1.0, ["t2"]),
            ("r6", "E", "F", 1.0, ["t2"]),
        ])
        result = await _run_create_communities(
            title_to_entity_id,
            relationships,
            leiden_config(),
        )
        all_rel_ids = []
        for _, row in result.iterrows():
            all_rel_ids.extend(row["relationship_ids"])
        assert "r_cross" not in all_rel_ids
        assert "t_cross" not in [
            tid for _, row in result.iterrows() for tid in row["text_unit_ids"]
        ]


# -------------------------------------------------------------------
# Parent / children tree
# -------------------------------------------------------------------


class TestParentChildTree:
    """Verify the parent-child tree structure is consistent."""

    async def test_parent_child_bidirectional_consistency_real_data(self):
        """For real test data: if community X lists Y as child,
        then Y's parent must be X."""
        entities_df = pd.read_parquet("tests/verbs/data/entities.parquet")
        title_to_entity_id = dict(
            zip(entities_df["title"], entities_df["id"], strict=False)
        )
        relationships = pd.read_parquet("tests/verbs/data/relationships.parquet")
        result = await _run_create_communities(
            title_to_entity_id,
            relationships,
            leiden_config(use_lcc=True, seed=0xDEADBEEF),
        )
        for _, row in result.iterrows():
            children = row["children"]
            if hasattr(children, "__len__") and len(children) > 0:
                for child_id in children:
                    child_row = result[result["community"] == child_id]
                    assert len(child_row) == 1, (
                        f"Child {child_id} not found or duplicated"
                    )
                    assert child_row.iloc[0]["parent"] == row["community"]


# -------------------------------------------------------------------
# LCC filtering
# -------------------------------------------------------------------


class TestLccFiltering:
    """Verify LCC filtering interaction with create_communities."""

    async def test_lcc_reduces_community_count(self):
        """With use_lcc=True and two disconnected components, only the
        larger component's communities should appear."""
        title_to_entity_id = _make_title_to_entity_id([
            ("e1", "A"),
            ("e2", "B"),
            ("e3", "C"),
            ("e4", "D"),
            ("e5", "E"),
            ("e6", "F"),
        ])
        relationships = _make_relationships([
            ("r1", "A", "B", 1.0, ["t1"]),
            ("r2", "A", "C", 1.0, ["t1"]),
            ("r3", "B", "C", 1.0, ["t1"]),
            ("r4", "D", "E", 1.0, ["t2"]),
            ("r5", "D", "F", 1.0, ["t2"]),
            ("r6", "E", "F", 1.0, ["t2"]),
        ])
        result_no_lcc = await _run_create_communities(
            title_to_entity_id,
            relationships,
            leiden_config(),
        )
        result_lcc = await _run_create_communities(
            title_to_entity_id,
            relationships,
            leiden_config(use_lcc=True),
        )
        assert len(result_lcc) < len(result_no_lcc)
        assert len(result_lcc) == 1


# -------------------------------------------------------------------
# Golden file regression (real test data)
# -------------------------------------------------------------------


class TestRealDataRegression:
    """Regression tests using the shared test fixture data.

    These pin exact values so any behavioral change during refactoring
    is caught immediately.
    """

    @pytest.fixture(scope="module")
    def real_result(self) -> pd.DataFrame:
        """One run of the operation over the shared real test data."""
        entities_df = pd.read_parquet("tests/verbs/data/entities.parquet")
        title_to_entity_id = dict(
            zip(entities_df["title"], entities_df["id"], strict=False)
        )
        relationships = pd.read_parquet("tests/verbs/data/relationships.parquet")
        return asyncio.run(
            _run_create_communities(
                title_to_entity_id,
                relationships,
                leiden_config(use_lcc=True, seed=0xDEADBEEF),
            )
        )

    def test_values_match_golden_file(self, real_result: pd.DataFrame):
        """The output should match the golden Parquet file for all
        columns except id (UUID) and period (date-dependent)."""
        expected = pd.read_parquet("tests/verbs/data/communities.parquet")

        assert len(real_result) == len(expected)

        skip_columns = {"id", "period", "children"}
        for col in COMMUNITIES_FINAL_COLUMNS:
            if col in skip_columns:
                continue
            pd.testing.assert_series_equal(
                real_result[col],
                expected[col],
                check_dtype=False,
                check_index=False,
                check_names=False,
                obj=f"Column '{col}'",
            )

        # children requires special handling: the golden file stores
        # numpy arrays, the function may return lists or arrays
        for i in range(len(real_result)):
            actual_children = list(real_result.iloc[i]["children"])
            expected_children = list(expected.iloc[i]["children"])
            assert actual_children == expected_children, (
                f"Row {i} children mismatch: {actual_children} != {expected_children}"
            )


# -------------------------------------------------------------------
# Row sanitization
# -------------------------------------------------------------------


class TestSanitizeRow:
    """Verify numpy types are converted to native Python types."""

    def test_ndarray_to_list(self):
        """np.ndarray values should become plain lists."""
        row = {"children": np.array([1, 2, 3])}
        result = _sanitize_row(row)
        assert result["children"] == [1, 2, 3]
        assert isinstance(result["children"], list)

    def test_empty_ndarray_to_empty_list(self):
        """An empty np.ndarray should become an empty list."""
        row = {"children": np.array([])}
        assert _sanitize_row(row)["children"] == []

    def test_np_integer_to_int(self):
        """np.integer values should become native int."""
        row = {"community": np.int64(42)}
        result = _sanitize_row(row)
        assert result["community"] == 42
        assert type(result["community"]) is int

    def test_np_floating_to_float(self):
        """np.floating values should become native float."""
        row = {"weight": np.float64(3.14)}
        result = _sanitize_row(row)
        assert result["weight"] == pytest.approx(3.14)
        assert type(result["weight"]) is float

    def test_mixed_row(self):
        """A row with a mix of numpy and native types."""
        row = {
            "community": np.int64(7),
            "children": np.array([1, 2]),
            "title": "Community 7",
            "weight": np.float64(0.5),
        }
        result = _sanitize_row(row)
        assert result == {
            "community": 7,
            "children": [1, 2],
            "title": "Community 7",
            "weight": pytest.approx(0.5),
        }
        assert type(result["community"]) is int
        assert type(result["children"]) is list
        assert type(result["weight"]) is float


async def test_no_edges_writes_an_empty_communities_table():
    provider = ParquetTableProvider(storage=MemoryStorage())

    await create_communities(
        FakeTable(),
        FakeEntitiesTable([{"id": "e1", "title": "A"}]),
        pd.DataFrame(),
        cfg(leiden_config()),
        provider,
    )

    communities = await provider.read_dataframe("communities")
    assert communities.empty
    assert list(communities.columns) == COMMUNITIES_FINAL_COLUMNS
