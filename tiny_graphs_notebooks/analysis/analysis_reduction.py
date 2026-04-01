"""Dimensionality-reduction and snapshot-selection helpers for tiny-graph analysis."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
from sklearn.decomposition import PCA

def reduce_embeddings_for_plot(
    embeddings: np.ndarray,
    *,
    use_umap: bool,
    reduction_dim: int,
    seed: int,
    umap_n_neighbors: int,
    umap_min_dist: float,
):
    """Reduces embeddings with UMAP or PCA and returns first 3 dimensions.

    Args:
        embeddings: Node embeddings `[num_nodes, dim]`.
        use_umap: Whether to use UMAP; otherwise PCA.
        reduction_dim: Number of reduced dimensions to compute.
        seed: Random seed for reducer.
        umap_n_neighbors: UMAP neighborhood size.
        umap_min_dist: UMAP min-dist parameter.

    Returns:
        tuple[np.ndarray, np.ndarray]: `(reduced_full, reduced_xyz)`.
    """
    output_dim = max(int(reduction_dim), 3)
    if use_umap:
        import umap

        reducer = umap.UMAP(
            n_components=output_dim,
            init='pca',
            n_neighbors=umap_n_neighbors,
            min_dist=umap_min_dist,
            random_state=seed,
        )
    else:
        reducer = PCA(n_components=output_dim, random_state=seed)

    reduced_full = reducer.fit_transform(embeddings)
    reduced_xyz = reduced_full[:, :3]
    return reduced_full, reduced_xyz

def _nearest_available_step(available_steps: Sequence[int], requested_step: int) -> int:
    """Returns closest available step to a requested step.

    Args:
        available_steps: Sorted available step values.
        requested_step: Desired step.

    Returns:
        int: Closest available step.
    """
    return min(available_steps, key=lambda step: abs(step - int(requested_step)))

def resolve_evolution_steps(
    embedding_history: Mapping[int, np.ndarray],
    requested_steps: Sequence[int],
):
    """Resolves requested evolution steps to existing history keys.

    Args:
        embedding_history: Mapping `step -> embeddings`.
        requested_steps: Preferred evolution steps from notebook constants.

    Returns:
        list[int]: Resolved existing step keys.
    """
    available_steps = sorted(int(step) for step in embedding_history.keys())
    if not available_steps:
        return []
    resolved = [_nearest_available_step(available_steps, s) for s in requested_steps]
    unique_resolved = []
    for step in resolved:
        if step not in unique_resolved:
            unique_resolved.append(step)
    return unique_resolved

def build_reduced_evolution_snapshots(
    embedding_history: Mapping[int, np.ndarray],
    selected_steps: Sequence[int],
    *,
    use_umap: bool,
    reduction_dim: int,
    seed: int,
    umap_n_neighbors: int,
    umap_min_dist: float,
):
    """Reduces selected embedding snapshots into a shared low-dimensional space.

    Args:
        embedding_history: Mapping `step -> embeddings`.
        selected_steps: Requested or resolved steps to visualize.
        use_umap: Whether to use UMAP (`False` uses PCA).
        reduction_dim: Joint reduced dimensionality before truncating to 3.
        seed: Reducer random seed.
        umap_n_neighbors: UMAP neighborhood size.
        umap_min_dist: UMAP min-dist.

    Returns:
        tuple[list[int], dict[int, np.ndarray]]: Resolved steps and reduced 3D coords by step.
    """
    resolved_steps = resolve_evolution_steps(embedding_history, selected_steps)
    if not resolved_steps:
        return [], {}

    stacked = np.vstack([embedding_history[int(step)] for step in resolved_steps])
    _, reduced_xyz = reduce_embeddings_for_plot(
        stacked,
        use_umap=use_umap,
        reduction_dim=max(reduction_dim, 3),
        seed=seed,
        umap_n_neighbors=umap_n_neighbors,
        umap_min_dist=umap_min_dist,
    )

    reduced_by_step: dict[int, np.ndarray] = {}
    cursor = 0
    for step in resolved_steps:
        n = embedding_history[int(step)].shape[0]
        reduced_by_step[int(step)] = reduced_xyz[cursor : cursor + n]
        cursor += n
    return resolved_steps, reduced_by_step

def resolve_embedding_step(
    embedding_history: Mapping[int, np.ndarray],
    requested_step: int,
) -> int:
    """Resolves an embedding-history step, supporting `-1` for latest step.

    Args:
        embedding_history: Mapping `step -> embeddings`.
        requested_step: Requested step (`-1` means latest available).

    Returns:
        int: Resolved step key from `embedding_history`.
    """
    if not embedding_history:
        raise ValueError('embedding_history is empty; cannot resolve a requested step.')

    available_steps = sorted(int(step) for step in embedding_history.keys())
    if int(requested_step) == -1:
        return available_steps[-1]
    if int(requested_step) in embedding_history:
        return int(requested_step)
    return _nearest_available_step(available_steps, int(requested_step))

def select_embedding_snapshot(
    embedding_history: Mapping[int, np.ndarray],
    fallback_embeddings: np.ndarray,
    requested_step: int,
) -> tuple[np.ndarray, int | None]:
    """Selects embeddings for plotting from history or fallback embeddings.

    Args:
        embedding_history: Mapping `step -> embeddings`.
        fallback_embeddings: Embeddings used when no history is available.
        requested_step: Requested history step (`-1` means latest).

    Returns:
        tuple[np.ndarray, int | None]: Selected embeddings and resolved step.
    """
    if not embedding_history:
        return np.asarray(fallback_embeddings), None

    resolved_step = resolve_embedding_step(
        embedding_history=embedding_history,
        requested_step=requested_step,
    )
    return np.asarray(embedding_history[int(resolved_step)]), int(resolved_step)

