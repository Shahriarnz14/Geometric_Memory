"""Shared tiny-graph embedding metrics used by notebook helpers."""

from __future__ import annotations

import numpy as np


def topk_predictions_from_embeddings(embeddings: np.ndarray, k: int) -> np.ndarray:
    """Computes top-k node predictions for each source node by dot products.

    Args:
        embeddings: Node embedding matrix with shape `[num_nodes, dim]`.
        k: Number of neighbors to retrieve per node.

    Returns:
        np.ndarray: Top-k node indices with shape `[num_nodes, k]`.
    """
    emb = np.asarray(embeddings, dtype=np.float64)
    num_nodes = emb.shape[0]
    top_k = max(1, min(int(k), max(1, num_nodes - 1)))

    scores = emb @ emb.T
    np.fill_diagonal(scores, -np.inf)
    candidate_idx = np.argpartition(scores, kth=-top_k, axis=1)[:, -top_k:]
    row_idx = np.arange(num_nodes)[:, None]
    ordering = np.argsort(scores[row_idx, candidate_idx], axis=1)[:, ::-1]
    return candidate_idx[row_idx, ordering]


def topk_predictions_from_embeddings_allowing_self(
    embeddings: np.ndarray,
    k: int,
    *,
    allow_self_predictions: bool = False,
) -> np.ndarray:
    """Computes top-k predictions, optionally allowing self-node predictions."""
    emb = np.asarray(embeddings, dtype=np.float64)
    num_nodes = emb.shape[0]
    top_k = max(1, min(int(k), max(1, num_nodes - (0 if allow_self_predictions else 1))))

    scores = emb @ emb.T
    if not allow_self_predictions:
        np.fill_diagonal(scores, -np.inf)
    candidate_idx = np.argpartition(scores, kth=-top_k, axis=1)[:, -top_k:]
    row_idx = np.arange(num_nodes)[:, None]
    ordering = np.argsort(scores[row_idx, candidate_idx], axis=1)[:, ::-1]
    return candidate_idx[row_idx, ordering]


def topk_recovery_percent(
    topk_predictions: np.ndarray,
    directed_edges: list[tuple[int, int]],
) -> float:
    """Computes directed-edge recovery percentage from top-k predictions.

    Args:
        topk_predictions: Integer prediction matrix `[num_nodes, k]`.
        directed_edges: Directed edge list `(src, dst)`.

    Returns:
        float: Recovery percentage in range `[0, 100]`.
    """
    if not directed_edges:
        return 0.0
    hits = 0
    for src, dst in directed_edges:
        if int(dst) in set(np.asarray(topk_predictions[int(src)]).tolist()):
            hits += 1
    return float(100.0 * hits / len(directed_edges))
