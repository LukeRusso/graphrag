# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License


from typing import Any

import graspologic_native as gn

from graphrag.config.models.cluster_graph_config import LeidenClusterGraphConfig
from graphrag.graphs.types import Cluster, Edge
from packages.graphrag.graphrag.graphs.clustering.base_clustering_algorithm import (
    IClusteringAlgorithm,
)


class HierarchicalLeiden(IClusteringAlgorithm[LeidenClusterGraphConfig]):
    """Hierarchical Leiden clustering on edge lists."""

    def cluster(
        self,
        edges: list[Edge],
    ) -> list[Cluster]:
        """Run hierarchical leiden on an edge list."""
        return self._gn_hierarchical_clusters_to_clusters(
            gn.hierarchical_leiden(
                edges=[(e.source, e.dest, e.weight) for e in edges],
                max_cluster_size=self.config.max_cluster_size,
                seed=self.config.seed,
                starting_communities=None,
                resolution=1.0,
                randomness=0.001,
                use_modularity=True,
                iterations=1,
            )
        )

    def _gn_hierarchical_clusters_to_clusters(
        self, gn_cluster: list[gn.HierarchicalCluster]
    ) -> list[Cluster]:
        cluster_map: dict[int, Cluster] = {}

        for node_assignment in gn_cluster:
            if cluster_map[node_assignment.cluster] is None:
                cluster_map[node_assignment.cluster] = Cluster(
                    level=node_assignment.level,
                    cluster_id=node_assignment.cluster,
                    parent_cluster_id=(
                        node_assignment.parent_cluster
                        if node_assignment.parent_cluster is not None
                        else -1
                    ),
                    nodes=[node_assignment.node],
                )
            else:
                cluster_map[node_assignment.cluster].nodes.append(node_assignment.node)

        return list(cluster_map.values())

    def first_level_hierarchical_clustering(
        self,
        hcs: list[gn.HierarchicalCluster],
    ) -> dict[Any, int]:
        """Return the initial leiden clustering as a dict of node id to community id.

        Returns
        -------
        dict[Any, int]
            The initial leiden algorithm clustering results as a dictionary
            of node id to community id.
        """
        return {entry.node: entry.cluster for entry in hcs if entry.level == 0}

    def final_level_hierarchical_clustering(
        self,
        hcs: list[gn.HierarchicalCluster],
    ) -> dict[Any, int]:
        """Return the final leiden clustering as a dict of node id to community id.

        Returns
        -------
        dict[Any, int]
            The last leiden algorithm clustering results as a dictionary
            of node id to community id.
        """
        return {entry.node: entry.cluster for entry in hcs if entry.is_final_cluster}
