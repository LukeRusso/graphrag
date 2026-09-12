# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Hierarchical VDStar clustering on edge lists."""

import vdstar

from graphrag.config.models.cluster_graph_config import VDStarClusterGraphConfig
from graphrag.graphs.clustering.base_clustering_algorithm import IClusteringAlgorithm
from graphrag.graphs.types import Cluster, Edge


class HierarchicalVDStar(IClusteringAlgorithm[VDStarClusterGraphConfig]):
    """Hierarchical VDStar clustering on edge lists."""

    def cluster(
        self,
        edges: list[Edge],
    ) -> list[Cluster]:
        """Run VDStar clustering on an edge list."""
        vdstar_edges: list[tuple[int, int]] = self._convert_graphrag_to_vdstar_edges(
            edges
        )

        vdstar_cluster_results: list[list[list[int]]] = [
            vdstar.cluster(
                self._next_node_id - 1,
                vdstar_edges,
                eps,
                self.config.mu,
                self.config.rho,
            )
            for eps in sorted(self.config.epsilon_levels)
        ]

        return self._convert_vdstar_cluster_results_to_hierarchical_graphrag_clusters(
            vdstar_cluster_results
        )

    def _convert_vdstar_cluster_results_to_hierarchical_graphrag_clusters(
        self, vdstar_cluster_results: list[list[list[int]]]
    ) -> list[Cluster]:
        hierarchical_graphrag_clusters: list[Cluster] = []
        levels: int = len(vdstar_cluster_results)
        node_to_cluster_id_maps: list[dict[int, int]] = [{} for _ in range(levels)]
        next_cluster_id: int = 0

        for level, clusters in enumerate(vdstar_cluster_results):
            for cluster in clusters:
                parent_id: int = -1

                if level > 0:
                    parent_ids: list[int] = [
                        node_to_cluster_id_maps[level - 1][node] for node in cluster
                    ]

                    if any(id != parent_ids[0] for id in parent_ids):
                        error_msg = (
                            "Cluster has multiple different parents: "
                            f"expected parent {parent_ids[0]}, got {parent_ids}"
                        )
                        raise RuntimeError(error_msg)

                    parent_id = parent_ids[0]

                hierarchical_graphrag_clusters.append(
                    Cluster(
                        level=level,
                        cluster_id=next_cluster_id,
                        parent_cluster_id=parent_id,
                        nodes=[self._int_to_str_id_map[node] for node in cluster],
                    )
                )

                for node in cluster:
                    node_to_cluster_id_maps[level][node] = next_cluster_id

                next_cluster_id += 1

        return hierarchical_graphrag_clusters

    def _convert_graphrag_to_vdstar_edges(
        self, edges: list[Edge]
    ) -> list[tuple[int, int]]:
        self._str_to_int_id_map: dict[str, int] = {}
        self._int_to_str_id_map: dict[int, str] = {}
        self._next_node_id: int = 1

        for e in edges:
            self._add_new_id_mappings_for_edge(e)

        return [
            (self._str_to_int_id_map[e.source], self._str_to_int_id_map[e.dest])
            for e in edges
        ]

    def _add_new_id_mappings_for_edge(self, edge: Edge) -> None:
        """Add id mappings for the nodes of an edge."""
        self._add_new_id_mapping_for_node(edge.source)
        self._add_new_id_mapping_for_node(edge.dest)

    def _add_new_id_mapping_for_node(self, node_id: str) -> None:
        """Add an id mapping for a single node id."""
        if node_id not in self._str_to_int_id_map:
            self._str_to_int_id_map[node_id] = self._next_node_id
            self._int_to_str_id_map[self._next_node_id] = node_id
            self._next_node_id += 1
