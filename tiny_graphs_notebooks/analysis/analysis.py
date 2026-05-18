"""Shared embedding-analysis and plotting helpers for tiny-graph notebooks.

This file is intentionally lightweight and re-exports helpers from focused
analysis modules so notebooks can keep stable imports while code stays modular.
"""

from tiny_graphs_notebooks.analysis.analysis_curves import (
    compute_associative_geometric_curves,
    plot_associative_vs_geometric_curves,
)
from tiny_graphs_notebooks.analysis.analysis_plots import (
    compute_node_ordering,
    get_star_branch_layout,
    plot_ordered_heatmaps,
    plot_stylized_embedding_graph,
    plot_three_snapshot_evolution,
)
from tiny_graphs_notebooks.analysis.analysis_reduction import (
    build_reduced_evolution_snapshots,
    reduce_embeddings_for_plot,
    resolve_embedding_step,
    resolve_evolution_steps,
    select_embedding_snapshot,
)
from tiny_graphs_notebooks.analysis.analysis_spectral import (
    build_graph_spectral_state,
    compute_laplacian_coordinates,
    compute_spectral_bias_from_state,
    plot_spectral_bias,
)
from tiny_graphs_notebooks.notebook_utils.figure_saving import (
    build_notebook_figure_path,
    save_figure_for_context,
)

__all__ = [
    'build_graph_spectral_state',
    'build_notebook_figure_path',
    'build_reduced_evolution_snapshots',
    'compute_associative_geometric_curves',
    'compute_laplacian_coordinates',
    'compute_node_ordering',
    'compute_spectral_bias_from_state',
    'get_star_branch_layout',
    'plot_associative_vs_geometric_curves',
    'plot_ordered_heatmaps',
    'plot_spectral_bias',
    'plot_stylized_embedding_graph',
    'plot_three_snapshot_evolution',
    'reduce_embeddings_for_plot',
    'resolve_embedding_step',
    'resolve_evolution_steps',
    'select_embedding_snapshot',
    'save_figure_for_context',
]
