# Copyright (C) 2026 luke russo -- ruff complaining

"""Tests for the community report reuse helpers."""

import pandas as pd
from graphrag.index.update.communities import (
    fingerprint_report_inputs,
    reusable_reports,
)

from tests.unit.graph_utils import communities, entities, relationships


def _claims(*claims: tuple[str, str]) -> pd.DataFrame:
    """A minimal covariates frame: (subject title, claim description) pairs."""
    return pd.DataFrame([
        {
            "subject_id": subject,
            "type": "CLAIM",
            "status": "TRUE",
            "description": text,
        }
        for subject, text in claims
    ])


def test_a_changed_description_makes_only_that_community_unusable():
    communities_frame = communities(("A", "B"), ("C",))
    relationships_frame = relationships(("A", "B"))
    previous_entities = entities("A", "B", "C")
    changed_entities = previous_entities.copy()
    changed_entities.loc[changed_entities["title"] == "B", "description"] = "B changed."

    assert reusable_reports(
        fingerprint_report_inputs(
            communities_frame, changed_entities, relationships_frame
        ),
        fingerprint_report_inputs(
            communities_frame, previous_entities, relationships_frame
        ),
    ) == {1: 1}


def test_an_added_member_makes_a_community_unusable():
    relationships_frame = relationships(("A", "B"))
    entities_frame = entities("A", "B", "C", "D")

    assert reusable_reports(
        fingerprint_report_inputs(
            communities(("A", "B", "D"), ("C",)), entities_frame, relationships_frame
        ),
        fingerprint_report_inputs(
            communities(("A", "B"), ("C",)), entities_frame, relationships_frame
        ),
    ) == {1: 1}


def test_a_changed_degree_makes_a_community_unusable():
    communities_frame = communities(("A", "B"), ("C",))
    relationships_frame = relationships(("A", "B"))
    previous_entities = entities("A", "B", "C")
    changed_entities = previous_entities.copy()
    changed_entities.loc[changed_entities["title"] == "A", "degree"] = 7

    assert reusable_reports(
        fingerprint_report_inputs(
            communities_frame, changed_entities, relationships_frame
        ),
        fingerprint_report_inputs(
            communities_frame, previous_entities, relationships_frame
        ),
    ) == {1: 1}


def test_a_changed_internal_edge_makes_only_its_community_unusable():
    communities_frame = communities(("A", "B"), ("C", "D"))
    entities_frame = entities("A", "B", "C", "D")
    previous_relationships = relationships(("A", "B"), ("C", "D"))
    changed_relationships = previous_relationships.copy()
    changed_relationships.loc[1, "description"] = "C and D, reworded."

    assert reusable_reports(
        fingerprint_report_inputs(
            communities_frame, entities_frame, changed_relationships
        ),
        fingerprint_report_inputs(
            communities_frame, entities_frame, previous_relationships
        ),
    ) == {0: 0}


def test_an_edge_between_communities_is_not_an_input():
    communities_frame = communities(("A", "B"), ("C", "D"))
    entities_frame = entities("A", "B", "C", "D")

    assert reusable_reports(
        fingerprint_report_inputs(
            communities_frame, entities_frame, relationships(("A", "B"), ("B", "C"))
        ),
        fingerprint_report_inputs(
            communities_frame, entities_frame, relationships(("A", "B"))
        ),
    ) == {0: 0, 1: 1}


def test_a_changed_claim_makes_a_community_unusable():
    communities_frame = communities(("A", "B"), ("C",))
    entities_frame = entities("A", "B", "C")
    relationships_frame = relationships(("A", "B"))

    assert reusable_reports(
        fingerprint_report_inputs(
            communities_frame,
            entities_frame,
            relationships_frame,
            _claims(("A", "A is Y")),
        ),
        fingerprint_report_inputs(
            communities_frame,
            entities_frame,
            relationships_frame,
            _claims(("A", "A is X")),
        ),
    ) == {1: 1}


def test_a_community_at_another_level_is_not_reusable():
    entities_frame = entities("A", "B")
    relationships_frame = relationships(("A", "B"))

    assert (
        reusable_reports(
            fingerprint_report_inputs(
                communities(("A", "B"), level=1), entities_frame, relationships_frame
            ),
            fingerprint_report_inputs(
                communities(("A", "B"), level=0), entities_frame, relationships_frame
            ),
        )
        == {}
    )
