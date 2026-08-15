from dataclasses import dataclass, field


@dataclass
class Cluster:
    """Represents one cluster in a hierarchical graph clustering output."""

    level: int
    cluster_id: int
    parent_cluster_id: int
    nodes: list[str] = field(default_factory=list)