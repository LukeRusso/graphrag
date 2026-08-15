from dataclasses import dataclass, field

@dataclass
class Cluster:
    level: int
    clusterId: int
    parentClusterId: int
    nodes: list[str] = field(default_factory=list) 