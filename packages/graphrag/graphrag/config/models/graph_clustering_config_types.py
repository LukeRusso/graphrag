from enum import StrEnum


class GraphClusteringAlgorithmType(StrEnum):
    """Enum for graph clustering algorithm identifiers."""

    LEIDEN = "Leiden"
    VDSTAR = "VDStar"
