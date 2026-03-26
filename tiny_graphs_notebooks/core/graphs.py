"""Shared tiny-graph constructors used by notebook helper modules."""

from __future__ import annotations

from typing import Mapping

import networkx as nx

FIXED_IRREGULAR_NODE_COUNT = 16
FIXED_IRREGULAR_UNDIRECTED_EDGES: tuple[tuple[int, int], ...] = (
    (12, 15),
    (5, 8),
    (5, 6),
    (8, 9),
    (7, 8),
    (1, 3),
    (13, 14),
    (13, 15),
    (12, 14),
    (12, 13),
    (1, 2),
    (4, 5),
    (3, 4),
    (10, 11),
    (0, 1),
    (2, 3),
    (6, 7),
    (11, 0),
    (9, 10),
    (14, 15),
)
FIXED_IRREGULAR_EDGE_COUNT = len(FIXED_IRREGULAR_UNDIRECTED_EDGES)


def build_tiny_star_graph(star_degree: int, path_length: int) -> tuple[nx.Graph, int | None]:
    """Builds a path-star graph with root node `0`.

    Args:
        star_degree: Number of branches from root.
        path_length: Number of nodes per branch path including root.

    Returns:
        tuple[nx.Graph, int | None]: Graph and root node index.
    """
    branch_depth = max(int(path_length) - 1, 1)
    degree = max(int(star_degree), 1)

    graph = nx.Graph()
    graph.add_node(0)
    next_node = 1
    for _ in range(degree):
        prev = 0
        for _ in range(branch_depth):
            graph.add_node(next_node)
            graph.add_edge(prev, next_node)
            prev = next_node
            next_node += 1
    return graph, 0


def build_tiny_grid_graph(rows: int, cols: int) -> tuple[nx.Graph, int | None]:
    """Builds a row-major relabeled 2D grid graph.

    Args:
        rows: Grid row count.
        cols: Grid column count.

    Returns:
        tuple[nx.Graph, int | None]: Graph and `None` root.
    """
    r = max(int(rows), 2)
    c = max(int(cols), 2)
    graph = nx.grid_2d_graph(r, c)
    row_major_nodes = sorted(graph.nodes(), key=lambda n: (n[0], n[1]))
    node_map = {node: idx for idx, node in enumerate(row_major_nodes)}
    graph = nx.relabel_nodes(graph, node_map)
    return graph, None


def build_tiny_cycle_graph(total_nodes: int) -> tuple[nx.Graph, int | None]:
    """Builds a cycle graph.

    Args:
        total_nodes: Cycle node count.

    Returns:
        tuple[nx.Graph, int | None]: Graph and `None` root.
    """
    n = max(int(total_nodes), 3)
    return nx.cycle_graph(n), None


def build_tiny_irregular_graph(
    total_nodes: int,
    irregular_edge_count: int,
) -> tuple[nx.Graph, int | None]:
    """Builds the fixed irregular graph used in tiny-graph experiments.

    Args:
        total_nodes: Node count (must be 16).
        irregular_edge_count: Undirected edge count (must be 20).

    Returns:
        tuple[nx.Graph, int | None]: Graph and `None` root.
    """
    if int(total_nodes) != FIXED_IRREGULAR_NODE_COUNT:
        raise ValueError(
            'Irregular tiny graph is fixed to 16 nodes; '
            f'received total_nodes={total_nodes}.'
        )
    if int(irregular_edge_count) != FIXED_IRREGULAR_EDGE_COUNT:
        raise ValueError(
            'Irregular tiny graph is fixed to 20 undirected edges; '
            f'received irregular_edge_count={irregular_edge_count}.'
        )

    graph = nx.Graph()
    graph.add_nodes_from(range(FIXED_IRREGULAR_NODE_COUNT))
    graph.add_edges_from(FIXED_IRREGULAR_UNDIRECTED_EDGES)
    return graph, None


def build_tiny_graph_from_config(
    config: Mapping[str, object],
) -> tuple[nx.Graph, int | None]:
    """Builds a tiny graph from section config.

    Args:
        config: Tiny graph configuration mapping.

    Returns:
        tuple[nx.Graph, int | None]: Graph and optional root node index.
    """
    graph_type = str(config.get('graph_type', 'star'))
    if graph_type == 'star':
        return build_tiny_star_graph(
            star_degree=int(config.get('star_degree', 4)),
            path_length=int(config.get('path_length', 5)),
        )
    if graph_type == 'grid':
        return build_tiny_grid_graph(
            rows=int(config.get('grid_rows', 4)),
            cols=int(config.get('grid_cols', 4)),
        )
    if graph_type == 'cycle':
        return build_tiny_cycle_graph(total_nodes=int(config.get('total_nodes', 15)))
    if graph_type == 'irregular':
        return build_tiny_irregular_graph(
            total_nodes=int(config.get('total_nodes', FIXED_IRREGULAR_NODE_COUNT)),
            irregular_edge_count=int(
                config.get('irregular_edge_count', FIXED_IRREGULAR_EDGE_COUNT)
            ),
        )
    raise ValueError(f'Unsupported graph_type: {graph_type}')


def build_bidirectional_edge_list(graph: nx.Graph) -> list[tuple[int, int]]:
    """Returns directed edges by adding both directions per undirected edge.

    Args:
        graph: Undirected graph.

    Returns:
        list[tuple[int, int]]: Sorted directed edge list.
    """
    edges: list[tuple[int, int]] = []
    for u, v in graph.edges():
        edges.append((int(u), int(v)))
        edges.append((int(v), int(u)))
    return sorted(edges)
