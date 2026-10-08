# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Hierarchical VDStar clustering on edge lists."""

import vdstar
from graphrag_storage import Storage

from graphrag.config.models.cluster_graph_config import VDStarClusterGraphConfig
from graphrag.graphs.clustering.base_clustering_algorithm import (
    IIncrementalClusteringAlgorithm,
)
from graphrag.graphs.types import Cluster, Edge
from graphrag.graphs.vdstar_state import VDStarState


class HierarchicalVDStar(IIncrementalClusteringAlgorithm[VDStarClusterGraphConfig]):
    """Hierarchical VDStar clustering on edge lists."""

    def __init__(self, config: VDStarClusterGraphConfig, storage: Storage) -> None:
        """Create the adapter over a VD-Star state namespace."""
        super().__init__(config)
        self._state = VDStarState(storage)

    async def cluster(
        self,
        edges: list[Edge],
    ) -> list[Cluster]:
        """Run VDStar clustering on an edge list, publishing state for a later update."""
        self._reset_id_maps()

        for edge in edges:
            self._vdstar_vertex_id_for_node(None, edge.source)
            self._vdstar_vertex_id_for_node(None, edge.dest)

        node_count = len(self._str_to_int_id_map)

        graph = vdstar.Graph(node_count, self.config.rho, self.config.seed)

        for edge in edges:
            graph.insert_edge(
                self._str_to_int_id_map[edge.source],
                self._str_to_int_id_map[edge.dest],
            )

        return await self._query_and_save(graph)

    async def cluster_incremental(
        self,
        edges_added: list[Edge],
        edges_removed: list[Edge],
    ) -> list[Cluster]:
        """Update the clustering given the provided graph changes."""
        graph = await self._load_graph()

        for edge in edges_removed:
            graph.remove_edge(
                self._str_to_int_id_map[edge.source], self._str_to_int_id_map[edge.dest]
            )

        for edge in edges_added:
            self._add_edge(graph, edge)

        return await self._query_and_save(graph)

    async def _load_graph(self) -> vdstar.Graph:
        """Read the persisted graph and rebuild the id maps."""
        state = await self._state.load()

        if state is None:
            error_msg = (
                "Cannot run incremental clustering: no persisted VD-Star state. "
            )
            raise RuntimeError(error_msg)

        blob, str_to_int_id_map = state

        graph = vdstar.GraphStore.read_bytes(blob)

        self._str_to_int_id_map = str_to_int_id_map
        self._int_to_str_id_map = {
            int_id: node_id for node_id, int_id in str_to_int_id_map.items()
        }
        # VDSTAR is the authority on the id space: ids are 1-based and it
        # never reuses them, so the next id is one past its vertex count.
        self._next_node_id = graph.get_vertex_num() + 1

        return graph

    async def _query_and_save(self, graph: vdstar.Graph) -> list[Cluster]:
        """Query every epsilon level, persist the state pair, and convert the results."""
        vdstar_cluster_results: list[list[list[int]]] = [
            graph.cluster(eps, self.config.mu)
            for eps in sorted(self.config.epsilon_levels)
        ]

        await self._state.save(
            vdstar.GraphStore.write_bytes(graph), dict(self._str_to_int_id_map)
        )

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

    def _reset_id_maps(self) -> None:
        """Start empty, for a fresh build."""
        self._str_to_int_id_map: dict[str, int] = {}
        self._int_to_str_id_map: dict[int, str] = {}
        self._next_node_id: int = 1

    @property
    def node_id_map(self) -> dict[str, int]:
        """A copy of the GraphRAG-node-id -> VDSTAR-id map."""
        return dict(self._str_to_int_id_map)

    def _add_edge(self, graph: vdstar.Graph, edge: Edge) -> None:
        """Register both endpoints and insert the edge into the engine."""
        source_id = self._vdstar_vertex_id_for_node(graph, edge.source)
        dest_id = self._vdstar_vertex_id_for_node(graph, edge.dest)

        graph.insert_edge(source_id, dest_id)

    def _vdstar_vertex_id_for_node(
        self, graph: vdstar.Graph | None, node_id: str
    ) -> int:
        """Return the vdstar vertex id for a given GraphRAG node, adding a vertex when new.

        Pre-existing nodes keep their persisted vertex id. New
        nodes draw a fresh id (and a fresh vertex) from the VD-STAR engine.

        ``graph`` may be ``None`` for the pre-pass that only assigns GraphRAG
        node -> vdstar vertex ids before the graph exists.
        """
        if node_id in self._str_to_int_id_map:
            return self._str_to_int_id_map[node_id]

        int_id: int = self._next_node_id

        if graph is not None:
            int_id = graph.add_vertex()

        self._str_to_int_id_map[node_id] = int_id
        self._int_to_str_id_map[int_id] = node_id
        self._next_node_id += 1

        return int_id
