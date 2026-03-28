"""Embedding, evolution, and heatmap plotting helpers for tiny-graph analysis."""

from __future__ import annotations

from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np

from tiny_graphs_notebooks.notebook_utils.graphing_utils import (
    EDGE_NODE_COLOR,
    ROOT_NODE_COLOR,
    fix_axes_style,
    plot_embedding_graph_3d,
    plot_node_similarity_and_adjacency,
)

def plot_stylized_embedding_graph(
    reduced_xyz: np.ndarray,
    edge_list: Sequence[tuple[int, int]],
    *,
    title: str,
    view: Mapping[str, float],
    root_node_index: int | None,
    axis_permutation: tuple[int, int, int],
):
    """Delegates styled 3D plotting to shared graphing utility.

    Args:
        reduced_xyz: Reduced embeddings `[num_nodes, 3]`.
        edge_list: Graph edges.
        title: Plot title.
        view: View config dict with `elev`, `azim`, `roll`.
        root_node_index: Optional star root index.
        axis_permutation: Axis permutation tuple.

    Returns:
        tuple[object, object]: Matplotlib `(fig, ax)` handles.
    """
    return plot_embedding_graph_3d(
        reduced_embeddings=reduced_xyz,
        edges=edge_list,
        title=title,
        view=view,
        root_node_index=root_node_index,
        axis_permutation=axis_permutation,
    )

def plot_three_snapshot_evolution(
    reduced_by_step: Mapping[int, np.ndarray],
    edge_list: Sequence[tuple[int, int]],
    *,
    title: str,
    view: Mapping[str, float],
    axis_permutation: tuple[int, int, int],
    root_node_index: int | None = None,
):
    """Plots three reduced embedding snapshots side-by-side.

    Args:
        reduced_by_step: Mapping `step -> reduced_xyz`.
        edge_list: Graph edges.
        title: Figure title.
        view: View config with `elev`, `azim`, `roll`.
        axis_permutation: Axis permutation tuple.
        root_node_index: Optional root index for path-star root highlighting.

    Returns:
        tuple[object, list[object]]: Matplotlib figure and axes list.
    """
    steps = sorted(reduced_by_step.keys())
    if not steps:
        raise ValueError('No reduced snapshots to plot.')

    x_idx, y_idx, z_idx = axis_permutation
    fig = plt.figure(figsize=(6 * len(steps), 5))
    axes = []
    for i, step in enumerate(steps, start=1):
        coords = reduced_by_step[step]
        ax = fig.add_subplot(1, len(steps), i, projection='3d')
        axes.append(ax)
        if root_node_index is None:
            ax.scatter(
                coords[:, x_idx],
                coords[:, y_idx],
                coords[:, z_idx],
                s=120,
                alpha=0.95,
                color=EDGE_NODE_COLOR,
                edgecolor='black',
                linewidth=0.3,
            )
        else:
            node_mask = np.ones(len(coords), dtype=bool)
            node_mask[root_node_index] = False
            non_root_nodes = coords[node_mask]
            root_node = coords[root_node_index]

            ax.scatter(
                non_root_nodes[:, x_idx],
                non_root_nodes[:, y_idx],
                non_root_nodes[:, z_idx],
                s=120,
                alpha=0.95,
                color=EDGE_NODE_COLOR,
                edgecolor='black',
                linewidth=0.3,
            )
            ax.scatter(
                root_node[x_idx],
                root_node[y_idx],
                root_node[z_idx],
                s=140,
                alpha=1.0,
                color=ROOT_NODE_COLOR,
                edgecolor='black',
                linewidth=0.3,
                zorder=20,
            )

        for src, dst in edge_list:
            ax.plot(
                [coords[src, x_idx], coords[dst, x_idx]],
                [coords[src, y_idx], coords[dst, y_idx]],
                [coords[src, z_idx], coords[dst, z_idx]],
                color='k',
                linewidth=0.8,
            )
        ax.view_init(
            elev=float(view['elev']),
            azim=float(view['azim']),
            roll=float(view.get('roll', 0.0)),
        )
        fix_axes_style(ax)
        ax.set_title(f'step={step}')

    if title:
        fig.suptitle(title)
    plt.tight_layout()
    plt.show()
    return fig, axes

def _build_star_branch_paths(
    edge_list: Sequence[tuple[int, int]],
    node_count: int,
    root_node_index: int | None,
) -> tuple[int, list[list[int]]]:
    """Builds ordered root-to-leaf branch paths for a path-star style graph.

    Args:
        edge_list: Directed edge list.
        node_count: Number of nodes.
        root_node_index: Optional root override.

    Returns:
        tuple[int, list[list[int]]]: Root index and list of branch paths.
    """
    undirected_adj = {node: set() for node in range(node_count)}
    for src, dst in edge_list:
        undirected_adj[src].add(dst)
        undirected_adj[dst].add(src)

    if root_node_index is None:
        root_node_index = max(
            range(node_count),
            key=lambda node: len(undirected_adj[node]),
            default=0,
        )

    branch_paths: list[list[int]] = []
    for branch_start in sorted(undirected_adj[root_node_index]):
        path = [root_node_index, branch_start]
        prev = root_node_index
        cur = branch_start
        while True:
            next_nodes = sorted(
                node for node in undirected_adj[cur] if node not in {prev, root_node_index}
            )
            if not next_nodes:
                break
            nxt = next_nodes[0]
            path.append(nxt)
            prev, cur = cur, nxt
        branch_paths.append(path)
    return root_node_index, branch_paths

def get_star_branch_layout(
    edge_list: Sequence[tuple[int, int]],
    node_count: int,
    root_node_index: int | None,
) -> list[list[int]]:
    """Returns explicit root-to-leaf branch rows for a path-star graph.

    Args:
        edge_list: Directed edge list.
        node_count: Number of nodes.
        root_node_index: Optional root override.

    Returns:
        list[list[int]]: Branch rows such as `[0, 1, 2, 3, 4]`.
    """
    _, branch_paths = _build_star_branch_paths(
        edge_list=edge_list,
        node_count=node_count,
        root_node_index=root_node_index,
    )
    return branch_paths

def compute_node_ordering(
    edge_list: Sequence[tuple[int, int]],
    node_count: int,
    *,
    graph_type: str,
    root_node_index: int | None,
):
    """Computes graph-aware node ordering for cleaner heatmaps.

    Args:
        edge_list: Directed edge list.
        node_count: Number of nodes.
        graph_type: Graph family name.
        root_node_index: Optional star root.

    Returns:
        list[int]: Node ordering.
    """
    if graph_type == 'star':
        root_node_index, branch_paths = _build_star_branch_paths(
            edge_list=edge_list,
            node_count=node_count,
            root_node_index=root_node_index,
        )
        ordering = [root_node_index]
        for path in branch_paths:
            ordering.extend(path[1:])

        seen = set(ordering)
        ordering.extend(node for node in range(node_count) if node not in seen)
        return ordering

    if graph_type in {'grid', 'cycle'}:
        return list(range(node_count))

    return list(range(node_count))

def plot_ordered_heatmaps(
    embeddings: np.ndarray,
    edge_list: Sequence[tuple[int, int]],
    *,
    graph_type: str,
    root_node_index: int | None,
    cmap_name: str = 'plasma',
    wspace: float = 0.5,
    show_titles: bool = False,
    custom_order: Sequence[int] | None = None,
):
    """Plots similarity/adjacency heatmaps with graph-aware node ordering.

    Args:
        embeddings: Node embeddings `[num_nodes, dim]`.
        edge_list: Directed edge list.
        graph_type: Graph family name.
        root_node_index: Optional star root.
        cmap_name: Matplotlib colormap name.
        wspace: Horizontal spacing between heatmap panels.
        show_titles: Whether to render subplot titles.
        custom_order: Optional precomputed node ordering.

    Returns:
        tuple: Return tuple from `plot_node_similarity_and_adjacency`.
    """
    node_order = (
        list(custom_order)
        if custom_order is not None
        else compute_node_ordering(
            edge_list=edge_list,
            node_count=embeddings.shape[0],
            graph_type=graph_type,
            root_node_index=root_node_index,
        )
    )
    return plot_node_similarity_and_adjacency(
        embeddings=embeddings,
        edges=edge_list,
        metric='cosine',
        directed=True,
        add_self_loops=False,
        cmap_name=cmap_name,
        order=node_order,
        wspace=wspace,
        show_titles=show_titles,
    )
