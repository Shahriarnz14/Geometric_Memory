"""Shared plotting and embedding-analysis helpers for tiny-graph notebooks."""

from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, to_rgb
import numpy as np
from sklearn.decomposition import PCA

try:
    import umap
except Exception:  # pragma: no cover - optional dependency for notebook workflows
    umap = None


EDGE_NODE_COLOR = "#0072B2"
ROOT_NODE_COLOR = "#44A5E5"
ASSOCIATIVE_CURVE_COLOR = "#E69F00"
ASSOCIATIVE_AXIS_COLOR = "#D55E00"
GEOMETRIC_CURVE_COLOR = "#0072B2"
PANE_COLOR = (0.9, 0.9, 0.9, 0.65)


def fix_axes_style(ax, pane_color=PANE_COLOR):
    """Applies the existing notebook 3D style for tiny-graph scatter plots.

    Args:
        ax: Matplotlib 3D axis object.
        pane_color: RGBA pane color used for all axis panes.

    Returns:
        object: The same axis object, styled in-place.
    """
    ax.tick_params(axis="both", which="major", labelsize=18, pad=10)

    ax.xaxis.set_pane_color(pane_color)
    ax.yaxis.set_pane_color(pane_color)
    ax.zaxis.set_pane_color(pane_color)

    ax.xaxis.line.set_alpha(0)
    ax.yaxis.line.set_alpha(0)
    ax.zaxis.line.set_alpha(0)

    ax.xaxis.set_tick_params(colors="white")
    ax.yaxis.set_tick_params(colors="white")
    ax.zaxis.set_tick_params(colors="white")

    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    return ax


def reduce_embeddings_pca_initialized_umap(
    embeddings: np.ndarray,
    n_neighbors: int,
    min_dist: float,
    random_state: int,
):
    """Reduces node embeddings to 3D using PCA-initialized UMAP.

    Args:
        embeddings: Node embedding matrix of shape `[num_nodes, dim]`.
        n_neighbors: UMAP neighborhood size.
        min_dist: UMAP minimum distance parameter.
        random_state: Random seed used by UMAP.

    Returns:
        object: Reduced embedding matrix of shape `[num_nodes, 3]`.
    """
    if umap is None:
        return PCA(n_components=3).fit_transform(embeddings)

    reducer = umap.UMAP(
        n_components=3,
        init="pca",
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
    )
    return reducer.fit_transform(embeddings)


def plot_embedding_graph_3d(
    reduced_embeddings: np.ndarray,
    edges: Sequence[Tuple[int, int]],
    title: str,
    view: Mapping[str, float],
    root_node_index: int | None = None,
    axis_permutation: Tuple[int, int, int] = (0, 1, 2),
    save_path: str | None = None,
):
    """Plots reduced node embeddings and graph edges in the notebook house style.

    Args:
        reduced_embeddings: Reduced coordinates with shape `[num_nodes, 3]`.
        edges: Edge list as `(u, v)` node-id tuples.
        title: Plot title.
        view: Dict containing `elev`, `azim`, and `roll`.
        root_node_index: Optional root node index (path-star special coloring).
        axis_permutation: Axis ordering used for plotting (e.g. `(0, 2, 1)`).
        save_path: Optional path to save the figure as PDF.

    Returns:
        object: `(fig, ax)` matplotlib objects.
    """
    x_idx, y_idx, z_idx = axis_permutation
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    if root_node_index is None:
        ax.scatter(
            reduced_embeddings[:, x_idx],
            reduced_embeddings[:, y_idx],
            reduced_embeddings[:, z_idx],
            s=200,
            color=EDGE_NODE_COLOR,
            alpha=1.0,
            edgecolor="black",
            linewidth=0.5,
        )
    else:
        node_mask = np.ones(len(reduced_embeddings), dtype=bool)
        node_mask[root_node_index] = False
        path_nodes = reduced_embeddings[node_mask]
        root_node = reduced_embeddings[root_node_index]

        ax.scatter(
            path_nodes[:, x_idx],
            path_nodes[:, y_idx],
            path_nodes[:, z_idx],
            s=200,
            color=EDGE_NODE_COLOR,
            alpha=1.0,
            edgecolor="black",
            linewidth=0.5,
        )
        ax.scatter(
            root_node[x_idx],
            root_node[y_idx],
            root_node[z_idx],
            s=200,
            color=ROOT_NODE_COLOR,
            alpha=1.0,
            edgecolor="black",
            linewidth=0.5,
            zorder=18,
        )

    for edge_start, edge_end in edges:
        x_values = [reduced_embeddings[edge_start, x_idx], reduced_embeddings[edge_end, x_idx]]
        y_values = [reduced_embeddings[edge_start, y_idx], reduced_embeddings[edge_end, y_idx]]
        z_values = [reduced_embeddings[edge_start, z_idx], reduced_embeddings[edge_end, z_idx]]
        ax.plot(x_values, y_values, z_values, color="k")

    fix_axes_style(ax)
    ax.view_init(
        elev=float(view["elev"]),
        azim=float(view["azim"]),
        roll=float(view.get("roll", 0.0)),
    )
    ax.set_title(title, fontsize=18)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    return fig, ax


def _cosine_similarity(a: np.ndarray, b: np.ndarray):
    """Computes cosine similarity between two vectors.

    Args:
        a: First vector.
        b: Second vector.

    Returns:
        object: Scalar cosine similarity.
    """
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def mean_edge_cosine_similarity(embeddings: np.ndarray, edges: Sequence[Tuple[int, int]]):
    """Computes mean cosine similarity across graph edges.

    Args:
        embeddings: Node embeddings of shape `[num_nodes, dim]`.
        edges: Edge list as `(u, v)` tuples.

    Returns:
        object: Mean cosine similarity across all edges.
    """
    edge_sims = [_cosine_similarity(embeddings[u], embeddings[v]) for u, v in edges]
    return float(np.mean(edge_sims))


def compute_geometric_level_curve(
    embedding_history: Mapping[int, np.ndarray],
    edges: Sequence[Tuple[int, int]],
):
    """Computes geometric memorization scores over recorded training steps.

    Args:
        embedding_history: Dict mapping step -> node embeddings.
        edges: Graph edge list.

    Returns:
        object: `(steps, geometric_scores)` as numpy arrays.
    """
    steps = np.array(sorted(embedding_history.keys()))
    geom_scores = np.array([mean_edge_cosine_similarity(embedding_history[s], edges) for s in steps])
    return steps, geom_scores


def compute_associative_curve(
    steps: Sequence[int],
    topk_history: Mapping[int, float] | None = None,
):
    """Builds associative memorization curve for plotting.

    Args:
        steps: Ordered training steps used on the x-axis.
        topk_history: Optional dict mapping step -> top-k hit rate in percent.

    Returns:
        object: Associative memorization scores (percent) aligned with `steps`.
    """
    if topk_history:
        return np.array([float(topk_history.get(int(s), 0.0)) for s in steps], dtype=float)

    assoc_scores = np.zeros(len(steps), dtype=float)
    if len(assoc_scores) > 1:
        assoc_scores[1:] = 100.0
    return assoc_scores


def plot_associative_vs_geometric(
    steps: Sequence[int],
    geometric_scores: Sequence[float],
    associative_scores: Sequence[float],
    title: str,
):
    """Plots associative and geometric memorization with dual y-axes.

    Args:
        steps: X-axis training steps.
        geometric_scores: Geometric level score per step.
        associative_scores: Associative memorization percentage per step.
        title: Figure title.

    Returns:
        object: `(fig, ax_left, ax_right)` matplotlib objects.
    """
    fig, ax_left = plt.subplots(figsize=(8, 5))

    line_left, = ax_left.plot(
        steps,
        associative_scores,
        linestyle="-",
        linewidth=3,
        label="Associative memorization (top-k)",
        color=ASSOCIATIVE_CURVE_COLOR,
    )
    ax_left.set_xlabel("Training step", fontsize=24)
    ax_left.tick_params(axis="x", labelsize=18)
    ax_left.set_ylabel("Associative\nMemorization[%]", color=ASSOCIATIVE_AXIS_COLOR, fontsize=24)
    ax_left.tick_params(axis="y", labelcolor=ASSOCIATIVE_AXIS_COLOR, labelsize=18)
    ax_left.set_ylim(-0.05, 105)

    ax_right = ax_left.twinx()
    line_right, = ax_right.plot(
        steps,
        geometric_scores,
        linestyle="-",
        linewidth=3,
        label="Geometric memorization (mean cosine similarity)",
        color=GEOMETRIC_CURVE_COLOR,
    )
    ax_right.set_ylabel("Geometric Level", color=GEOMETRIC_CURVE_COLOR, fontsize=24, labelpad=15)
    ax_right.tick_params(axis="y", labelcolor=GEOMETRIC_CURVE_COLOR, labelsize=18)
    y_pad = max(0.05, 0.1 * (np.max(geometric_scores) - np.min(geometric_scores) + 1e-8))
    ax_right.set_ylim(np.min(geometric_scores) - y_pad, np.max(geometric_scores) + y_pad)

    lines = [line_left, line_right]
    labels = [line.get_label() for line in lines]
    ax_left.legend(lines, labels, fontsize=14, frameon=False, loc="upper left")
    ax_left.set_title(title, fontsize=18)
    ax_left.grid(alpha=0.25)
    fig.tight_layout()
    plt.show()
    return fig, ax_left, ax_right


def plot_node_similarity_and_adjacency(
    embeddings: np.ndarray,
    edges: Iterable[Tuple[int, int]],
    metric: str = "cosine",
    directed: bool = True,
    add_self_loops: bool = False,
    cmap_name: str = "coolwarm",
    wspace: float = 0.04,
    order: Sequence[int] | None = None,
    show_titles: bool = False,
    save_path: str | None = None,
):
    """Plots node-node similarity and adjacency matrices side-by-side.

    Args:
        embeddings: Node embedding matrix `[num_nodes, dim]`.
        edges: Edge iterable as `(u, v)` tuples.
        metric: Similarity metric (`\"cosine\"` or dot-product fallback).
        directed: Whether adjacency should remain directed.
        add_self_loops: Whether to set adjacency diagonal to 1.
        cmap_name: Matplotlib colormap name.
        wspace: Horizontal subplot spacing.
        order: Optional node reordering applied to both matrices.
        show_titles: Whether to display subplot titles.
        save_path: Optional path to save the figure as PDF.

    Returns:
        object: `(fig, (ax_similarity, ax_adjacency), S_masked, A_masked)`.
    """
    E = np.asarray(embeddings, dtype=float)
    if order is not None:
        E = E[np.asarray(order)]

    if metric == "cosine":
        X = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)
        S = X @ X.T
        sim_vmin, sim_vmax = -1.0, 1.0
    else:
        S = E @ E.T
        sim_vmin, sim_vmax = float(np.min(S)), float(np.max(S))
    S_masked = S.copy()
    np.fill_diagonal(S_masked, np.nan)

    N = E.shape[0]
    A = np.zeros((N, N), dtype=float)
    for u, v in edges:
        A[int(u), int(v)] = 1.0

    if not directed:
        A = np.maximum(A, A.T)
    if add_self_loops:
        np.fill_diagonal(A, 1.0)
    if order is not None:
        idx = np.asarray(order)
        A = A[np.ix_(idx, idx)]

    A_masked = A.copy()
    np.fill_diagonal(A_masked, np.nan)

    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad(color="0.85")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"wspace": wspace})
    im1 = ax1.imshow(
        S_masked,
        cmap=cmap,
        norm=Normalize(vmin=sim_vmin, vmax=sim_vmax),
        interpolation="nearest",
    )
    if show_titles:
        ax1.set_title("Node-Node Similarity Matrix", fontsize=22)
    ax1.set_xlabel("Node Embedding", fontsize=26)
    ax1.set_ylabel("Node Embedding", fontsize=26)
    ax1.set_xticks([])
    ax1.set_yticks([])
    cb1 = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.02)
    cb1.set_label("Similarity", fontsize=22)
    cb1.ax.tick_params(labelsize=18)

    im2 = ax2.imshow(
        A_masked,
        cmap=cmap,
        norm=Normalize(vmin=0.0, vmax=1.0),
        interpolation="nearest",
    )
    if show_titles:
        ax2.set_title("Adjacency Matrix", fontsize=22)
    ax2.set_xlabel("Node Index", fontsize=26)
    ax2.set_ylabel("Node Index", fontsize=26)
    ax2.set_xticks([])
    ax2.set_yticks([])
    cb2 = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.01)
    cb2.set_label("Adjacency", fontsize=22)
    cb2.ax.tick_params(labelsize=18)

    # fig.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    return fig, (ax1, ax2), S_masked, A_masked


def _interpolate_color(color1: str, color2: str, fraction: float):
    """Interpolates between two hex colors.

    Args:
        color1: Start color.
        color2: End color.
        fraction: Blend fraction in `[0, 1]`.

    Returns:
        object: RGB numpy array.
    """
    rgb1 = np.array(to_rgb(color1))
    rgb2 = np.array(to_rgb(color2))
    return rgb1 + (rgb2 - rgb1) * fraction


def plot_embedding_evolution_3d(
    reduced_history: Mapping[int, np.ndarray],
    edges: Sequence[Tuple[int, int]],
    title: str,
    view: Mapping[str, float],
    root_node_index: int | None = None,
    axis_permutation: Tuple[int, int, int] = (0, 1, 2),
    max_snapshots: int = 4,
):
    """Plots multi-step 3D embedding evolution snapshots.

    Args:
        reduced_history: Dict mapping step -> reduced embeddings `[num_nodes, 3]`.
        edges: Edge list as `(u, v)` tuples.
        title: Figure title.
        view: Dict containing `elev`, `azim`, and `roll`.
        root_node_index: Optional root node index for path-star coloring.
        axis_permutation: Axis order for plotting.
        max_snapshots: Maximum number of time snapshots to display.

    Returns:
        object: `(fig, axes)` matplotlib objects.
    """
    if not reduced_history:
        raise ValueError("Cannot plot embedding evolution: reduced_history is empty.")

    all_steps = sorted(reduced_history.keys())
    if len(all_steps) <= max_snapshots:
        snapshot_steps = all_steps
    else:
        snapshot_indices = np.linspace(0, len(all_steps) - 1, num=max_snapshots).astype(int)
        snapshot_steps = [all_steps[idx] for idx in snapshot_indices]

    x_idx, y_idx, z_idx = axis_permutation
    fig = plt.figure(figsize=(6 * len(snapshot_steps), 6))
    axes = []

    for panel_index, step in enumerate(snapshot_steps, start=1):
        reduced_embeddings = reduced_history[step]
        ax = fig.add_subplot(1, len(snapshot_steps), panel_index, projection="3d")
        axes.append(ax)

        if root_node_index is None:
            ax.scatter(
                reduced_embeddings[:, x_idx],
                reduced_embeddings[:, y_idx],
                reduced_embeddings[:, z_idx],
                s=160,
                color=EDGE_NODE_COLOR,
                alpha=1.0,
                edgecolor="black",
                linewidth=0.4,
            )
        else:
            node_mask = np.ones(len(reduced_embeddings), dtype=bool)
            node_mask[root_node_index] = False
            path_nodes = reduced_embeddings[node_mask]
            root_node = reduced_embeddings[root_node_index]

            ax.scatter(
                path_nodes[:, x_idx],
                path_nodes[:, y_idx],
                path_nodes[:, z_idx],
                s=160,
                color=EDGE_NODE_COLOR,
                alpha=1.0,
                edgecolor="black",
                linewidth=0.4,
            )
            ax.scatter(
                root_node[x_idx],
                root_node[y_idx],
                root_node[z_idx],
                s=170,
                color=ROOT_NODE_COLOR,
                alpha=1.0,
                edgecolor="black",
                linewidth=0.4,
                zorder=20,
            )

        for edge_start, edge_end in edges:
            x_values = [reduced_embeddings[edge_start, x_idx], reduced_embeddings[edge_end, x_idx]]
            y_values = [reduced_embeddings[edge_start, y_idx], reduced_embeddings[edge_end, y_idx]]
            z_values = [reduced_embeddings[edge_start, z_idx], reduced_embeddings[edge_end, z_idx]]
            ax.plot(x_values, y_values, z_values, color="k", linewidth=0.8)

        fix_axes_style(ax)
        ax.view_init(
            elev=float(view["elev"]),
            azim=float(view["azim"]),
            roll=float(view.get("roll", 0.0)),
        )
        ax.set_title(f"step={step}", fontsize=14)

    fig.suptitle(title, fontsize=18)
    plt.tight_layout()
    plt.show()
    return fig, axes


def reduce_embedding_history_joint(
    embedding_history: Mapping[int, np.ndarray],
    n_neighbors: int,
    min_dist: float,
    random_state: int,
    step_stride: int = 10,
):
    """Jointly reduces sampled embedding snapshots into one consistent 3D space.

    Args:
        embedding_history: Dict mapping step -> embeddings `[num_nodes, dim]`.
        n_neighbors: UMAP neighborhood size.
        min_dist: UMAP minimum distance parameter.
        random_state: UMAP random seed.
        step_stride: Keep every Nth recorded step, always keeping first/last.

    Returns:
        object: Dict mapping selected step -> reduced embeddings `[num_nodes, 3]`.
    """
    all_steps = sorted(embedding_history.keys())
    if not all_steps:
        return {}

    selected_steps = []
    first_step, last_step = all_steps[0], all_steps[-1]
    for step in all_steps:
        if step in {first_step, last_step} or (step - first_step) % max(1, step_stride) == 0:
            selected_steps.append(step)
    selected_steps = sorted(set(selected_steps))

    combined = np.vstack([embedding_history[s] for s in selected_steps])
    reduced_combined = reduce_embeddings_pca_initialized_umap(
        embeddings=combined,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
    )

    reduced_history: Dict[int, np.ndarray] = {}
    cursor = 0
    for step in selected_steps:
        n = embedding_history[step].shape[0]
        reduced_history[step] = reduced_combined[cursor : cursor + n]
        cursor += n
    return reduced_history
