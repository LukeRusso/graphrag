from dataclasses import dataclass


@dataclass
class Edge:
    """Represents an edge in a graph."""

    source_node: str
    destination_node: str
    weight: float