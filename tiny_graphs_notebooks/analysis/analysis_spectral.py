"""Graph spectral-analysis helpers for tiny-graph notebooks."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np

from tiny_graphs_notebooks.notebook_utils.figure_saving import save_figure_for_context


def _slugify_plot_filename(filename: str) -> str:
    """Builds a filesystem-safe stem for saved plot filenames."""
    normalized = re.sub(r'[^A-Za-z0-9._-]+', '_', filename.strip())
    normalized = normalized.strip('._-')
    return normalized or 'spectral_bias'


def _build_dense_undirected_adjacency(
    edge_list: Sequence[tuple[int, int]],
    node_count: int,
) -> np.ndarray:
    """Builds a dense undirected adjacency matrix from a directed/undirected edge list.

    Args:
        edge_list: Graph edge list.
        node_count: Number of graph nodes.

    Returns:
        np.ndarray: Dense adjacency matrix `[node_count, node_count]`.
    """
    adjacency = np.zeros((node_count, node_count), dtype=np.float64)
    for src, dst in edge_list:
        s = int(src)
        d = int(dst)
        adjacency[s, d] = 1.0
        adjacency[d, s] = 1.0
    return adjacency


def _compute_eigen_rotation_matrix(rotation_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Computes eigenvalues/eigenvectors and returns eigenvectors as row vectors.

    Args:
        rotation_matrix: Square matrix.

    Returns:
        tuple[np.ndarray, np.ndarray]: Eigenvalues and row-wise eigenvectors.
    """
    eigenvalues, eigenvectors = np.linalg.eig(rotation_matrix)
    eigenvalues = np.real_if_close(eigenvalues, tol=1000)
    eigenvectors = np.real_if_close(eigenvectors, tol=1000)
    sorted_indices = np.argsort(eigenvalues)
    eigenvalues = np.asarray(eigenvalues[sorted_indices], dtype=np.float64)
    eigenvectors = np.asarray(eigenvectors[:, sorted_indices], dtype=np.float64)
    return eigenvalues, eigenvectors.T


def _compute_eigen_adjacency_matrix(
    adjacency_matrix: np.ndarray,
    degree_matrix: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Computes graph-spectrum eigenpairs from adjacency and optional degree normalization.

    Args:
        adjacency_matrix: Dense adjacency matrix.
        degree_matrix: Optional row-wise degree matrix.

    Returns:
        tuple[np.ndarray, np.ndarray]: Eigenvalues and row-wise eigenvectors.
    """
    if degree_matrix is None:
        normalized = adjacency_matrix.copy().astype(np.float64)
    else:
        normalized = np.divide(adjacency_matrix, degree_matrix)
    normalized += normalized.T
    return _compute_eigen_rotation_matrix(normalized)


def build_graph_spectral_state(
    edge_list: Sequence[tuple[int, int]],
    node_count: int,
) -> dict[str, np.ndarray]:
    """Builds graph spectral artifacts for Laplacian/spectral-bias analysis.

    Args:
        edge_list: Graph edge list.
        node_count: Number of graph nodes.

    Returns:
        dict[str, np.ndarray]: Spectral state dictionary.
    """
    adjacency = _build_dense_undirected_adjacency(edge_list=edge_list, node_count=node_count)
    degree_vector = np.sum(adjacency, axis=1)
    degree_matrix = np.tile(degree_vector, (node_count, 1)).T.astype(np.float64)
    degree_matrix[degree_matrix == 0] = 1e-6

    init_rotation = adjacency - np.multiply(np.eye(node_count, dtype=np.float64), degree_matrix)
    init_rotation = np.divide(init_rotation, degree_matrix)
    init_rotation += init_rotation.T

    eigenvalues_init, eigenvectors_rows_init = _compute_eigen_rotation_matrix(init_rotation)
    graph_eigenvalues, graph_eigenvectors_rows = _compute_eigen_adjacency_matrix(
        adjacency_matrix=adjacency,
        degree_matrix=degree_matrix,
    )

    return {
        'adjacency_matrix': adjacency,
        'degree_matrix': degree_matrix,
        'init_rotation_matrix': init_rotation,
        'init_eigenvalues': eigenvalues_init,
        'init_eigenvectors_rows': eigenvectors_rows_init,
        'graph_eigenvalues': graph_eigenvalues,
        'graph_eigenvectors_rows': graph_eigenvectors_rows,
    }


def compute_laplacian_coordinates(
    spectral_state: Mapping[str, np.ndarray],
    axis_indices: tuple[int, int, int] = (-2, -3, -4),
) -> np.ndarray:
    """Builds 3D Laplacian coordinates from spectral state.

    Args:
        spectral_state: Spectral state from `build_graph_spectral_state`.
        axis_indices: Eigenvector row indices used as x/y/z.

    Returns:
        np.ndarray: Coordinates `[num_nodes, 3]`.
    """
    eig_rows = np.asarray(spectral_state['init_eigenvectors_rows'])
    x_idx, y_idx, z_idx = axis_indices
    return np.stack([eig_rows[x_idx, :], eig_rows[y_idx, :], eig_rows[z_idx, :]], axis=1)


def compute_spectral_bias_from_state(
    embeddings: np.ndarray,
    spectral_state: Mapping[str, np.ndarray],
    *,
    drop_top_eigenvector: bool = True,
    eigenvalue_tie_tol: float = 1e-5,
    reorder_prefix: Sequence[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Computes normalized graph-spectrum and embedding-projection curves.

    Args:
        embeddings: Node embeddings `[num_nodes, dim]`.
        spectral_state: Spectral state from `build_graph_spectral_state`.
        drop_top_eigenvector: Whether to drop leading trivial eigenvector.
        eigenvalue_tie_tol: Tolerance for treating eigenvalues as tied when
            applying the projection-based secondary sort.
        reorder_prefix: Optional index permutation for the first k entries.

    Returns:
        tuple[np.ndarray, np.ndarray]: Normalized eigenvalue and projection arrays.
    """
    graph_eigenvalues = np.asarray(spectral_state['graph_eigenvalues'], dtype=np.float64)
    eig_rows = np.asarray(spectral_state['init_eigenvectors_rows'], dtype=np.float64)

    projection_matrix = np.matmul(np.asarray(embeddings, dtype=np.float64).T, eig_rows.T)
    projection_norms = np.linalg.norm(projection_matrix, axis=0)

    sorted_indices = np.argsort(graph_eigenvalues)[::-1]
    graph_eigenvalues = graph_eigenvalues[sorted_indices]
    projection_norms = projection_norms[sorted_indices]

    start_idx = 1 if drop_top_eigenvector else 0
    filtered_eigenvalues = graph_eigenvalues[start_idx:]
    filtered_projections = projection_norms[start_idx:]

    eig_den = max(np.max(filtered_eigenvalues), 1e-12)
    proj_den = max(np.max(filtered_projections), 1e-12)
    norm_eigenvalues = filtered_eigenvalues / eig_den
    norm_projections = filtered_projections / proj_den

    if len(norm_eigenvalues) > 1:
        tie_tolerance = max(eigenvalue_tie_tol, 1e-12)
        tie_buckets = np.round(norm_eigenvalues / tie_tolerance).astype(np.int64)
        projection_tiebreak = np.where(
            norm_eigenvalues < -tie_tolerance,
            norm_projections,
            -norm_projections,
        )
        sorted_indices = np.lexsort((projection_tiebreak, -tie_buckets))
        norm_eigenvalues = norm_eigenvalues[sorted_indices]
        norm_projections = norm_projections[sorted_indices]

    if reorder_prefix is not None and len(reorder_prefix) > 0:
        idx = np.asarray(reorder_prefix, dtype=int)
        if np.max(idx) < len(norm_eigenvalues):
            reordered_eigenvalues = norm_eigenvalues.copy()
            reordered_projections = norm_projections.copy()
            reordered_eigenvalues[: len(idx)] = norm_eigenvalues[idx]
            reordered_projections[: len(idx)] = norm_projections[idx]
            norm_eigenvalues = reordered_eigenvalues
            norm_projections = reordered_projections

    return norm_eigenvalues, norm_projections


def plot_spectral_bias(
    norm_eigenvalues: Sequence[float],
    norm_projections: Sequence[float],
    *,
    title: str = '',
    save: bool = False,
    filename: str | None = None,
    save_context: object | None = None,
    save_model_type: str | None = None,
    cutoff: int | None = None,
    figsize: tuple[float, float] = (15.0, 5.0),
    ylabel_fontsize: int = 28,
    xlabel_fontsize: int = 28,
    tick_fontsize: int = 20,
    legend_fontsize: int = 20,
    legend_anchor: tuple[float, float] = (0.0, 1.3),
):
    """Plots skewed low-rank spectral-bias bars.

    Args:
        norm_eigenvalues: Normalized graph-spectrum values.
        norm_projections: Normalized embedding-projection values.
        title: Optional title.
        save: Whether to save the plot into the experiment logs folder.
        filename: Optional filename stem to use when saving.
        save_context: Optional notebook section context for deterministic saves.
        save_model_type: Optional override for the model-type filename token.
        cutoff: Optional x-axis truncation.
        figsize: Figure size.
        ylabel_fontsize: Y-axis label font size.
        xlabel_fontsize: X-axis label font size.
        tick_fontsize: Tick label font size.
        legend_fontsize: Legend font size.
        legend_anchor: Legend anchor point.

    Returns:
        tuple[object, object]: Matplotlib `(fig, ax)` handles.
    """
    import matplotlib as mpl

    mpl.rcParams.update(mpl.rcParamsDefault)
    mpl.rcParams['font.family'] = 'monospace'

    eig_vals = np.asarray(norm_eigenvalues, dtype=np.float64)
    proj_vals = np.asarray(norm_projections, dtype=np.float64)

    if cutoff is not None:
        limit = max(1, min(int(cutoff), len(eig_vals), len(proj_vals)))
        eig_vals = eig_vals[:limit]
        proj_vals = proj_vals[:limit]

    fig, ax = plt.subplots(figsize=figsize)
    indices = np.arange(len(eig_vals))
    width = 0.3
    line_width = 2

    ax.bar(
        indices - width / 2,
        eig_vals,
        width,
        edgecolor='black',
        linewidth=line_width,
        label=r'Graph Spectrum ($\lambda_i/\lambda_1$)',
        color='#009E73',
        zorder=9,
    )
    ax.bar(
        indices + width / 2,
        proj_vals,
        width,
        edgecolor='black',
        linewidth=line_width,
        label=r'Normalized Embedding Projection ($||\mathbf{V}^T \mathbf{e}_i||$ $/$ $||\mathbf{V}^T \mathbf{e}_1||$)',
        color='#D55E00',
        zorder=10,
    )

    ax.set_ylabel('Normalized Magnitude', fontsize=ylabel_fontsize)
    ax.set_xlabel('Eigenvector Index', fontsize=xlabel_fontsize)
    if title:
        ax.set_title(title, fontsize=16, pad=20, y=1.1, loc='right')
    ax.axhline(0, color='black', linewidth=0.8)
    ax.grid(axis='y', which='major', linestyle='-', linewidth=0.8, color='grey', alpha=0.7, zorder=1)
    ax.grid(axis='y', which='minor', linestyle=':', linewidth=0.6, color='grey', alpha=0.5, zorder=1)
    ax.legend(loc='upper left', bbox_to_anchor=legend_anchor, ncol=1, frameon=False, fontsize=legend_fontsize)

    tick_labels = [rf'${{{i + 1}}}$' for i in range(len(indices))]
    ax.set_xticks(indices)
    ax.set_xticklabels(tick_labels, fontsize=tick_fontsize)
    ax.tick_params(axis='y', labelsize=tick_fontsize)

    fig.tight_layout()
    if save_context is not None:
        save_figure_for_context(
            save_context,
            'spectral_bias_plots',
            fig=fig,
            model_type=save_model_type,
        )
    elif save:
        # plot_dir = Path.cwd().resolve() / 'experiment_logs' / 'spectral_bias_plots'
        # plot_dir.mkdir(parents=True, exist_ok=True)
        # filename_stem = _slugify_plot_filename(filename or title or 'spectral_bias')
        # for extension in ('pdf', 'png'):
        #     fig.savefig(
        #         plot_dir / f'{filename_stem}.{extension}',
        #         dpi=300,
        #         bbox_inches='tight',
        #     )
        fig.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    return fig, ax


# ============================================================================
# Edge Margin and Accuracy Computation
# ============================================================================

import torch


@torch.no_grad()
def compute_skewed_low_rank_logits(model: torch.nn.Module) -> torch.Tensor:
    """Compute skewed low-rank logits from model embeddings and weights.
    
    Args:
        model: PyTorch model with embedding and middle layer.
        
    Returns:
        torch.Tensor: Skewed logits matrix.
    """
    # Support both naming styles:
    # 1) model.embedding / model.middle_layer
    # 2) model.embed_tokens / model.layers[0].mlp.fc1
    embedding_module = getattr(model, "embedding", None)
    if embedding_module is None:
        embedding_module = getattr(model, "embed_tokens")

    middle_layer = getattr(model, "middle_layer", None)
    if middle_layer is None:
        middle_layer = model.layers[0].mlp.fc1

    E = embedding_module.weight.detach()   # (N, D)
    W = middle_layer.weight.detach()       # (D, D)

    # SVD(E) = U S Vh
    _, _, Vh = torch.linalg.svd(E, full_matrices=False)  # Vh: (K, D)

    # lambdas = diag(Vh @ W @ Vh.T)
    Vh_W_Vht = Vh @ W @ Vh.T
    lambdas = torch.diag(Vh_W_Vht)

    # W' = Vh.T @ diag(lambdas) @ Vh
    W_prime = Vh.T @ torch.diag(lambdas) @ Vh

    # logits = E @ W' @ E.T
    logits = E @ W_prime @ E.T
    return logits


@torch.no_grad()
def _build_edge_mask(node_count: int, edge_list, device: torch.device) -> torch.Tensor:
    """Build boolean edge mask from edge list."""
    edge_mask = torch.zeros((node_count, node_count), dtype=torch.bool, device=device)
    for edge in edge_list:
        if len(edge) < 2:
            continue
        src, dst = int(edge[0]), int(edge[1])
        if 0 <= src < node_count and 0 <= dst < node_count:
            edge_mask[src, dst] = True
    return edge_mask


@torch.no_grad()
def _edge_margin(logits: torch.Tensor, edge_mask: torch.Tensor) -> float:
    """Compute edge margin: max(true_edge_logits) - max(non_edge_logits).
    
    Excludes self-edges from both calculations.
    """
    node_count = logits.size(0)
    no_self_edge = ~torch.eye(node_count, dtype=torch.bool, device=logits.device)

    # Mask: true edges excluding self-edges
    pos_mask = edge_mask & no_self_edge
    # Mask: non-edges excluding self-edges
    neg_mask = (~edge_mask) & no_self_edge

    pos_exists = pos_mask.any(dim=1)
    neg_exists = neg_mask.any(dim=1)
    valid_rows = pos_exists & neg_exists

    pos_scores = logits.masked_fill(~pos_mask, float("-inf")).max(dim=1).values
    neg_scores = logits.masked_fill(~neg_mask, float("-inf")).max(dim=1).values

    if not valid_rows.any():
        return float("nan")

    return (pos_scores[valid_rows] - neg_scores[valid_rows]).mean().item()


@torch.no_grad()
def _top1_edge_accuracy(logits: torch.Tensor, edge_mask: torch.Tensor) -> float:
    """Compute top-1 edge prediction accuracy, excluding self-edge predictions."""
    node_count = logits.size(0)
    no_self_edge = ~torch.eye(node_count, dtype=torch.bool, device=logits.device)

    # Only consider predictions for non-self positions
    edge_mask_no_self = edge_mask & no_self_edge
    valid_rows = edge_mask_no_self.any(dim=1)

    if not valid_rows.any():
        return float("nan")

    preds = torch.argmax(logits, dim=-1)
    # Check if predictions are actual edges (excluding self-edge predictions)
    hits = edge_mask_no_self[torch.arange(logits.size(0), device=logits.device), preds]
    return hits[valid_rows].float().mean().item() * 100.0


@torch.no_grad()
def compute_margin_and_accuracy(
    model: torch.nn.Module,
    edge_list,
    label: str = "",
) -> tuple[float, float, float]:
    """Compute edge margin and accuracy for original and skewed logits.
    
    Args:
        model: PyTorch model.
        edge_list: Graph edge list.
        label: Optional label for printing.
        
    Returns:
        tuple[float, float, float]: (margin_original, margin_skewed, accuracy_skewed)
    """
    logits_skewed = compute_skewed_low_rank_logits(model)

    embedding_module = getattr(model, "embedding", None)
    if embedding_module is None:
        embedding_module = getattr(model, "embed_tokens")

    middle_layer = getattr(model, "middle_layer", None)
    if middle_layer is None:
        middle_layer = model.layers[0].mlp.fc1

    E = embedding_module.weight.detach()
    W = middle_layer.weight.detach()

    # Original logits use the learned middle layer: E @ W @ E.T
    logits_original = E @ W @ E.T

    edge_mask = _build_edge_mask(logits_skewed.size(0), edge_list, logits_skewed.device)

    margin_original = _edge_margin(logits_original, edge_mask)
    margin_skewed = _edge_margin(logits_skewed, edge_mask)
    accuracy_skewed = _top1_edge_accuracy(logits_skewed, edge_mask)

    prefix = f"{label} | " if label else ""
    print(f"{prefix}edge_margin_original: {margin_original:.6f}")
    print(f"{prefix}edge_margin_skewed: {margin_skewed:.6f}")
    print(f"{prefix}edge_top1_accuracy_skewed: {accuracy_skewed:.2f}%")
    return margin_original, margin_skewed, accuracy_skewed
