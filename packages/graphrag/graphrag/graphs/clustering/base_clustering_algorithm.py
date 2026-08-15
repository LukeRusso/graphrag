from abc import ABC, abstractmethod

import pandas as pd

from .cluster import Cluster


class IClusteringAlgorithm(ABC):
    """Interface for hierarchical clustering algorithm."""

    @abstractmethod
    def cluster(self, edges: pd.DataFrame) -> list[Cluster]:
        """Perform hierarchical clustering on input graph."""
