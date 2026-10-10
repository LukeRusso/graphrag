# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Relationship related operations and utils for Incremental Indexing."""

import itertools
from typing import NamedTuple

import numpy as np
import pandas as pd

from graphrag.data_model.schemas import RELATIONSHIPS_FINAL_COLUMNS
from graphrag.graphs.types import Edge


class GraphDelta(NamedTuple):
    """The edges and vertices a new graph adds to and removes from an old one."""

    added_edges: list[Edge]
    removed_edges: list[Edge]
    added_vertices: list[str]
    removed_vertices: list[str]


def compute_graph_delta(
    old_relationships: pd.DataFrame,
    new_relationships: pd.DataFrame,
    old_entities: pd.DataFrame,
    new_entities: pd.DataFrame,
) -> GraphDelta:
    """Diff a new graph against the old one, returning what changed.

    Edges are compared as undirected (min, max) title pairs, so a reversed row
    is not a change. Vertices are compared by title, so a vertex with no edges
    still counts. Both sides are sorted, so the delta is reproducible.
    """
    old_edges = _undirected_edges(old_relationships)
    new_edges = _undirected_edges(new_relationships)

    old_titles = set(old_entities["title"])
    new_titles = set(new_entities["title"])

    return GraphDelta(
        added_edges=[
            new_edges[pair] for pair in sorted(new_edges.keys() - old_edges.keys())
        ],
        removed_edges=[
            old_edges[pair] for pair in sorted(old_edges.keys() - new_edges.keys())
        ],
        added_vertices=sorted(new_titles - old_titles),
        removed_vertices=sorted(old_titles - new_titles),
    )


def _undirected_edges(relationships: pd.DataFrame) -> dict[tuple[str, str], Edge]:
    """Index edges by their undirected pair, a duplicate row keeping the last weight."""
    edges: dict[tuple[str, str], Edge] = {}

    for source, target, weight in zip(
        relationships["source"].astype(str),
        relationships["target"].astype(str),
        relationships["weight"].astype(float),
        strict=True,
    ):
        pair = (min(source, target), max(source, target))
        edges[pair] = Edge(pair[0], pair[1], weight)

    return edges


def _update_and_merge_relationships(
    old_relationships: pd.DataFrame, delta_relationships: pd.DataFrame
) -> pd.DataFrame:
    """Update and merge relationships.

    Parameters
    ----------
    old_relationships : pd.DataFrame
        The old relationships.
    delta_relationships : pd.DataFrame
        The delta relationships.

    Returns
    -------
    pd.DataFrame
        The updated relationships.
    """
    # Increment the human readable id in b by the max of a
    # Ensure both columns are integers
    delta_relationships["human_readable_id"] = delta_relationships[
        "human_readable_id"
    ].astype(int)
    old_relationships["human_readable_id"] = old_relationships[
        "human_readable_id"
    ].astype(int)

    # Adjust delta_relationships IDs to be greater than any in old_relationships
    initial_id = old_relationships["human_readable_id"].max() + 1
    delta_relationships["human_readable_id"] = np.arange(
        initial_id, initial_id + len(delta_relationships)
    )

    # Merge the DataFrames without copying if possible
    merged_relationships = pd.concat(
        [old_relationships, delta_relationships], ignore_index=True
    )

    # Group by title and resolve conflicts
    aggregated = (
        merged_relationships
        .groupby(["source", "target"])
        .agg({
            "id": "first",
            "human_readable_id": "first",
            "description": lambda x: list(x.astype(str)),  # Ensure str
            # Concatenate nd.array into a single list
            "text_unit_ids": lambda x: list(itertools.chain(*x.tolist())),
            "weight": "mean",
            "combined_degree": "sum",
        })
        .reset_index()
    )

    # Force the result into a DataFrame
    final_relationships: pd.DataFrame = pd.DataFrame(aggregated)

    # Recalculate target and source degrees
    final_relationships["source_degree"] = final_relationships.groupby("source")[
        "target"
    ].transform("count")
    final_relationships["target_degree"] = final_relationships.groupby("target")[
        "source"
    ].transform("count")

    # Recalculate the combined_degree of the relationships (source degree + target degree)
    final_relationships["combined_degree"] = (
        final_relationships["source_degree"] + final_relationships["target_degree"]
    )

    return final_relationships.loc[
        :,
        RELATIONSHIPS_FINAL_COLUMNS,
    ]
