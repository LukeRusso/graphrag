# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Dataframe operations and utils for Incremental Indexing."""

import pandas as pd

from graphrag.data_model.schemas import (
    CLAIM_STATUS,
    CLAIM_SUBJECT,
    COMMUNITIES_FINAL_COLUMNS,
    COMMUNITY_ID,
    COMMUNITY_LEVEL,
    COMMUNITY_REPORTS_FINAL_COLUMNS,
    DESCRIPTION,
    EDGE_DEGREE,
    EDGE_SOURCE,
    EDGE_TARGET,
    ENTITY_IDS,
    ID,
    NODE_DEGREE,
    TITLE,
    TYPE,
)


def _update_and_merge_communities(
    old_communities: pd.DataFrame,
    delta_communities: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """Update and merge communities.

    Parameters
    ----------
    old_communities : pd.DataFrame
        The old communities.
    delta_communities : pd.DataFrame
        The delta communities.
    community_id_mapping : dict
        The community id mapping.

    Returns
    -------
    pd.DataFrame
        The updated communities.
    """
    # Check if size and period columns exist in the old_communities. If not, add them
    if "size" not in old_communities.columns:
        old_communities["size"] = None
    if "period" not in old_communities.columns:
        old_communities["period"] = None

    # Same for delta_communities
    if "size" not in delta_communities.columns:
        delta_communities["size"] = None
    if "period" not in delta_communities.columns:
        delta_communities["period"] = None

    # Increment all community ids by the max of the old communities
    old_max_community_id = old_communities["community"].fillna(0).astype(int).max()
    # Increment only the non-NaN values in delta_communities["community"]
    community_id_mapping = {
        v: v + old_max_community_id + 1
        for v in delta_communities["community"].dropna().astype(int)
    }
    community_id_mapping.update({-1: -1})

    # Look for community ids in community and replace them with the corresponding id in the mapping
    delta_communities["community"] = (
        delta_communities["community"]
        .astype(int)
        .apply(lambda x: community_id_mapping.get(x, x))
    )

    delta_communities["parent"] = (
        delta_communities["parent"]
        .astype(int)
        .apply(lambda x: community_id_mapping.get(x, x))
    )

    old_communities["community"] = old_communities["community"].astype(int)

    # Merge the final communities
    merged_communities = pd.concat(
        [old_communities, delta_communities], ignore_index=True
    )

    # Rename title
    merged_communities["title"] = "Community " + merged_communities["community"].astype(
        str
    )
    # Re-assign the human_readable_id
    merged_communities["human_readable_id"] = merged_communities["community"]

    merged_communities = merged_communities.loc[
        :,
        COMMUNITIES_FINAL_COLUMNS,
    ]
    return merged_communities, community_id_mapping


def _update_and_merge_community_reports(
    old_community_reports: pd.DataFrame,
    delta_community_reports: pd.DataFrame,
    community_id_mapping: dict,
    reused_reports_map: dict[int, int] | None = None,
    keep_previous_reports: bool = True,
    parent_by_community: dict[int, int] | None = None,
) -> pd.DataFrame:
    """Update and merge community reports.

    Parameters
    ----------
    old_community_reports : pd.DataFrame
        The old community reports.
    delta_community_reports : pd.DataFrame
        The delta community reports.
    community_id_mapping : dict
        The community id mapping.
    reused_reports_map : dict[int, int]
        Delta community id -> the previous community whose report it reuses.
    keep_previous_reports : bool
        Whether the previous reports still describe communities in the output.
    parent_by_community : dict[int, int]
        Output community id -> parent id, used to re-parent the reused reports.

    Returns
    -------
    pd.DataFrame
        The updated community reports.
    """
    # Check if size and period columns exist in the old_community_reports. If not, add them
    if "size" not in old_community_reports.columns:
        old_community_reports["size"] = None
    if "period" not in old_community_reports.columns:
        old_community_reports["period"] = None

    # Same for delta_community_reports
    if "size" not in delta_community_reports.columns:
        delta_community_reports["size"] = None
    if "period" not in delta_community_reports.columns:
        delta_community_reports["period"] = None

    # Look for community ids in community and replace them with the corresponding id in the mapping
    delta_community_reports["community"] = (
        delta_community_reports["community"]
        .astype(int)
        .apply(lambda x: community_id_mapping.get(x, x))
    )

    delta_community_reports["parent"] = (
        delta_community_reports["parent"]
        .astype(int)
        .apply(lambda x: community_id_mapping.get(x, x))
    )

    old_community_reports["community"] = old_community_reports["community"].astype(int)

    kept_reports = [old_community_reports] if keep_previous_reports else []
    if reused_reports_map:
        old_to_output = {
            old_id: community_id_mapping[new_id]
            for new_id, old_id in reused_reports_map.items()
        }
        reused_rows = old_community_reports[
            old_community_reports["community"].isin(old_to_output)
        ].copy()
        reused_rows["community"] = reused_rows["community"].map(old_to_output)
        if parent_by_community is not None:
            reused_rows["parent"] = reused_rows["community"].map(parent_by_community)
        kept_reports.append(reused_rows)

    # Merge the final community reports
    merged_community_reports = pd.concat(
        [*kept_reports, delta_community_reports], ignore_index=True
    )

    # Maintain type compat with query
    merged_community_reports["community"] = merged_community_reports[
        "community"
    ].astype(int)
    # Re-assign the human_readable_id
    merged_community_reports["human_readable_id"] = merged_community_reports[
        "community"
    ]

    return merged_community_reports.loc[:, COMMUNITY_REPORTS_FINAL_COLUMNS]


def fingerprint_report_inputs(
    communities: pd.DataFrame,
    entities: pd.DataFrame,
    relationships: pd.DataFrame,
    claims: pd.DataFrame | None = None,
) -> dict[int, tuple]:
    """Return one report-input fingerprint per community."""
    membership = communities.explode(ENTITY_IDS).loc[
        :, [COMMUNITY_ID, COMMUNITY_LEVEL, ENTITY_IDS]
    ]
    membership = membership.merge(
        entities.loc[:, [ID, TITLE, DESCRIPTION, NODE_DEGREE]],
        left_on=ENTITY_IDS,
        right_on=ID,
        how="left",
    )

    levels: dict[int, int] = {}
    members: dict[int, set] = {}
    by_title: dict[str, set[int]] = {}
    for community, level, title, description, degree in zip(
        membership[COMMUNITY_ID].to_numpy(),
        membership[COMMUNITY_LEVEL].to_numpy(),
        membership[TITLE].to_numpy(),
        _missing_as_none(membership[DESCRIPTION]).to_numpy(),
        membership[NODE_DEGREE].to_numpy(),
        strict=True,
    ):
        community = int(community)
        levels[community] = int(level)
        members.setdefault(community, set()).add((title, description, degree))
        by_title.setdefault(title, set()).add(community)

    edges: dict[int, set] = {}
    for source, target, description, degree in zip(
        relationships[EDGE_SOURCE].to_numpy(),
        relationships[EDGE_TARGET].to_numpy(),
        _missing_as_none(relationships[DESCRIPTION]).to_numpy(),
        relationships[EDGE_DEGREE].to_numpy(),
        strict=True,
    ):
        pair = (min(source, target), max(source, target))
        shared_communities = by_title.get(source, set()).intersection(
            by_title.get(target, set())
        )
        for community in shared_communities:
            edges.setdefault(community, set()).add((*pair, description, degree))

    claims_by_community: dict[int, set] = {}
    if claims is not None and not claims.empty:
        for subject, claim_type, status, description in zip(
            claims[CLAIM_SUBJECT].to_numpy(),
            claims[TYPE].to_numpy(),
            claims[CLAIM_STATUS].to_numpy(),
            _missing_as_none(claims[DESCRIPTION]).to_numpy(),
            strict=True,
        ):
            for community in by_title.get(subject, set()):
                claims_by_community.setdefault(community, set()).add((
                    subject,
                    claim_type,
                    status,
                    description,
                ))

    return {
        community: (
            levels[community],
            frozenset(members[community]),
            frozenset(edges.get(community, ())),
            frozenset(claims_by_community.get(community, ())),
        )
        for community in levels
    }


def reusable_reports(
    new_fingerprints: dict[int, tuple],
    old_fingerprints: dict[int, tuple],
) -> dict[int, int]:
    """If possible, map new communities to a previous community whose report it can reuse."""
    previous_by_fingerprint: dict[tuple, int] = {}
    for community in sorted(old_fingerprints):
        previous_by_fingerprint.setdefault(old_fingerprints[community], community)
    return {
        community: previous_by_fingerprint[fingerprint]
        for community, fingerprint in new_fingerprints.items()
        if fingerprint in previous_by_fingerprint
    }


def _missing_as_none(values: pd.Series) -> pd.Series:
    return values.astype(object).where(values.notna(), None)
