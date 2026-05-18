"""Deterministic figure paths and save helpers for tiny-graph notebooks."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any

import matplotlib.pyplot as plt


FIGURE_SUBFOLDERS = {
    "embeddings": "embeddings",
    "geometric_memorization_evolution": "geometric_memorization_evolution",
    "node_embedding_similarity_heatmap": "node_embedding_similarity_heatmap",
    "spectral_bias_plots": "spectral_bias_plots",
}

FIGURE_SUFFIXES = {
    "embeddings": "embedding_graph",
    "geometric_memorization_evolution": "memorization_evolution",
    "node_embedding_similarity_heatmap": "embedding_heatmap",
    "spectral_bias_plots": "eigen_projection",
}


def format_scientific_token(value: float) -> str:
    """Formats a number as compact scientific notation for filename tokens."""
    numeric = float(value)
    if numeric == 0.0:
        return "0e0"
    mantissa, exponent = f"{numeric:.15e}".split("e")
    mantissa = mantissa.rstrip("0").rstrip(".")
    exponent_int = int(exponent)
    return f"{mantissa}e{exponent_int}"


def _token(value: object) -> str:
    """Normalizes one filename token while preserving underscores."""
    normalized = re.sub(r"[^A-Za-z0-9_-]+", "_", str(value).strip().lower())
    normalized = re.sub(r"_+", "_", normalized).strip("_-")
    return normalized or "unknown"


def _context_args(context: object) -> object:
    return getattr(context, "args", context)


def _context_config(context: object) -> dict[str, Any]:
    config = getattr(context, "config", None)
    return dict(config) if isinstance(config, dict) else {}


def _get_context_value(context: object, name: str, default: object = None) -> object:
    args = _context_args(context)
    if hasattr(args, name):
        return getattr(args, name)
    config = _context_config(context)
    if name in config:
        return config[name]
    if hasattr(context, name):
        return getattr(context, name)
    return default


def _model_type_token(context: object, override: str | None = None) -> str:
    if override:
        return override

    model_family = str(_get_context_value(context, "model_family", "") or "").strip().lower()
    architecture = str(
        _get_context_value(context, "model_architecture_label", "") or ""
    ).strip().lower()
    config = _context_config(context)

    if model_family:
        if model_family == "gpt":
            if architecture.startswith("feedforward"):
                return "neuralnet"
            if architecture == "associative":
                return "associative"
        return model_family

    if "embedding_dim" in config or context.__class__.__name__.lower().startswith("node2vec"):
        return "node2vec"

    return architecture or "model"


def _is_forward_only(context: object) -> bool:
    return bool(_get_context_value(context, "add_forward_edges", False)) and not bool(
        _get_context_value(context, "add_backward_edges", False)
    )


def build_notebook_figure_path(
    context: object,
    plot_type: str,
    *,
    alt_view: bool = False,
    model_type: str | None = None,
    notebook_dir: str | Path | None = None,
) -> Path:
    """Builds a deterministic notebook-local PDF path for a tiny-graph figure.

    Args:
        context: Notebook section context. Supports transformer-style contexts
            with `.args` and Node2Vec-style contexts with `.config` fields.
        plot_type: One of `FIGURE_SUBFOLDERS`.
        alt_view: Whether to use the alternate embedding-view suffix.
        model_type: Optional override for the model-type filename token.
        notebook_dir: Optional notebook directory. Defaults to current working
            directory, matching notebook-local artifact behavior.

    Returns:
        Path: `saved_artifacts/figures/<subfolder>/<tokens>__<suffix>.pdf`.
    """
    if plot_type not in FIGURE_SUBFOLDERS:
        valid = ", ".join(sorted(FIGURE_SUBFOLDERS))
        raise ValueError(f"Unknown plot_type {plot_type!r}. Expected one of: {valid}")

    graph_type = _get_context_value(context, "graph_type", None)
    if graph_type is None:
        raise ValueError("Figure context must provide graph_type.")

    tokens = [_token(graph_type), _token(_model_type_token(context, override=model_type))]

    if _is_forward_only(context):
        tokens.append("forward_only")

    if str(_get_context_value(context, "weight_init_mode", "") or "") == "non_geometric":
        tokens.append("init_non_geometric")

    lr = _get_context_value(context, "edge_memorization_learning_rate", None)
    if lr is None and _context_config(context):
        lr = _get_context_value(context, "learning_rate", None)
    if lr is not None:
        tokens.append(f"lr_{format_scientific_token(float(lr))}")

    dropout = float(_get_context_value(context, "dropout_rate", 0.0) or 0.0)
    if dropout != 0.0:
        tokens.append(f"dropout_{format_scientific_token(dropout)}")

    weight_decay = float(_get_context_value(context, "optimizer_weight_decay", 0.0) or 0.0)
    if weight_decay != 0.0:
        tokens.append(f"weight_decay_{format_scientific_token(weight_decay)}")

    if bool(_get_context_value(context, "add_self_edges", False)):
        tokens.append("self_edges")

    suffix = FIGURE_SUFFIXES[plot_type]
    if plot_type == "embeddings" and alt_view:
        suffix = "embedding_graph_alt_view"

    root = Path(notebook_dir).resolve() if notebook_dir is not None else Path.cwd().resolve()
    figure_dir = root / "saved_artifacts" / "figures" / FIGURE_SUBFOLDERS[plot_type]
    figure_dir.mkdir(parents=True, exist_ok=True)
    return figure_dir / f"{'__'.join(tokens)}__{suffix}.pdf"


def save_figure_for_context(
    context: object,
    plot_type: str,
    *,
    fig=None,
    alt_view: bool = False,
    model_type: str | None = None,
    notebook_dir: str | Path | None = None,
) -> Path:
    """Tight-layouts and saves a figure with the deterministic notebook path."""
    path = build_notebook_figure_path(
        context=context,
        plot_type=plot_type,
        alt_view=alt_view,
        model_type=model_type,
        notebook_dir=notebook_dir,
    )
    if fig is not None:
        fig.tight_layout()
        fig.savefig(path, dpi=300, bbox_inches="tight")
    else:
        plt.tight_layout()
        plt.savefig(path, dpi=300, bbox_inches="tight")
    return path
