"""Associative/geometric curve analysis helpers for tiny-graph notebooks."""

from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

def compute_associative_geometric_curves(
    embedding_history: Mapping[int, np.ndarray],
    topk_recovery_history: Mapping[int, float],
    edge_list: Sequence[tuple[int, int]],
):
    """Computes aligned associative/geometric curves over training steps.

    Args:
        embedding_history: Mapping `step -> embeddings`.
        topk_recovery_history: Mapping `step -> top-k recovery %`.
        edge_list: Directed edge list.

    Returns:
        tuple[list[int], list[float], list[float]]: Steps, associative %, geometric score.
    """
    common_steps = sorted(set(embedding_history.keys()) & set(topk_recovery_history.keys()))
    if not common_steps:
        common_steps = sorted(embedding_history.keys())

    edge_array = np.asarray(edge_list, dtype=int)
    src_idx = edge_array[:, 0]
    dst_idx = edge_array[:, 1]

    associative_scores = []
    geometric_scores = []
    for step in common_steps:
        emb = embedding_history[int(step)]
        src_vec = emb[src_idx]
        dst_vec = emb[dst_idx]
        dot = np.sum(src_vec * dst_vec, axis=1)
        norm = np.linalg.norm(src_vec, axis=1) * np.linalg.norm(dst_vec, axis=1) + 1e-8
        geometric_scores.append(float(np.mean(dot / norm)))
        associative_scores.append(float(topk_recovery_history.get(int(step), 0.0)))

    return common_steps, associative_scores, geometric_scores

def plot_associative_vs_geometric_curves(
    steps: Sequence[int],
    associative_scores: Sequence[float],
    geometric_scores: Sequence[float],
    *,
    title: str,
    save_path: str | None = None,
):
    """Plots associative-vs-geometric curves with a combined legend.

    Args:
        steps: X-axis steps.
        associative_scores: Associative memorization percentages.
        geometric_scores: Geometric level scores.
        title: Figure title.
        save_path: Optional path to save the figure as PDF.

    Returns:
        tuple[object, object, object]: `(fig, ax_left, ax_right)`.
    """
    fig, ax_left = plt.subplots(figsize=(7.0, 5.0))

    line_assoc, = ax_left.plot(
        steps,
        associative_scores,
        color='#E69F00',
        linewidth=4,
        label='Memorization (top-k) accuracy',
    )
    ax_left.set_xlabel('Training Steps', fontsize=20)
    ax_left.set_ylabel('Memorization [%]', color='#D55E00', fontsize=18)
    ax_left.tick_params(axis='x', labelsize=14)
    ax_left.tick_params(axis='y', labelsize=14, labelcolor='#D55E00')
    ax_left.set_ylim(-1, 105)

    ax_right = ax_left.twinx()
    line_geo, = ax_right.plot(
        steps,
        geometric_scores,
        color='#0072B2',
        linewidth=4,
        label='Geometric level',
    )
    ax_right.set_ylabel('Geometric Level', color='#0072B2', fontsize=18)
    ax_right.tick_params(axis='y', labelsize=14, labelcolor='#0072B2')
    gmin = float(np.min(geometric_scores))
    gmax = float(np.max(geometric_scores))
    gpad = max(0.03, 0.1 * (gmax - gmin + 1e-8))
    ax_right.set_ylim(gmin - gpad, gmax + gpad)

    ax_left.grid(alpha=0.25)
    lines = [line_assoc, line_geo]
    labels = [line.get_label() for line in lines]
    ax_left.legend(
        lines,
        labels,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.15),
        frameon=False,
        fontsize=12,
        ncol=1,
    )

    ax_left.set_title(title, fontsize=16)
    fig.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    return fig, ax_left, ax_right

