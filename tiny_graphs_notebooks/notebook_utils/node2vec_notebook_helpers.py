"""Helper utilities for tiny-graph Node2Vec (tied) notebooks.

This module provides a clean, reproducible notebook-facing API for:
1. Building tiny graph sections (star/grid/cycle/irregular).
2. Training or loading tied Node2Vec embeddings.
3. Evaluating top-k edge memorization.
4. Loading/saving embedding and top-k histories for downstream plots.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Mapping

os.environ.setdefault("TORCH_DISABLE_DYNAMO", "1")

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

from tiny_graphs_notebooks.analysis.graphs import (
    FIXED_IRREGULAR_EDGE_COUNT,
    FIXED_IRREGULAR_NODE_COUNT,
    build_bidirectional_edge_list,
    build_tiny_graph_from_config,
)
from tiny_graphs_notebooks.analysis.metrics import (
    topk_predictions_from_embeddings_allowing_self,
    topk_recovery_percent,
)
from tiny_graphs_notebooks.analysis.reproducibility import set_all_seeds
from tiny_graphs_notebooks.analysis.storage import (
    load_json,
    read_pickle,
    save_json,
    write_pickle,
)
from tiny_graphs_notebooks.notebook_utils.paths import get_tiny_graphs_root

@dataclass
class Node2VecSectionContext:
    """Notebook section context for one tiny-graph Node2Vec experiment.

    Attributes:
        config: Section configuration dictionary.
        seed: Deterministic random seed.
        graph: Undirected graph used for training.
        graph_type: Graph family string (`star`, `grid`, `cycle`, `irregular`).
        node_count: Number of graph nodes.
        root_node_index: Root node for star graphs, otherwise `None`.
        directed_edge_list: Directed edge list (both directions for each undirected edge).
        run_name: Generated run identifier used for artifact names.
        model: Tied embedding model (`nn.Embedding`).
        checkpoint_path: Saved/loaded checkpoint path.
        embedding_history_path: Saved/loaded embedding-history pickle path.
        topk_history_path: Saved/loaded top-k recovery-history pickle path.
        final_embeddings_path: Saved/loaded final-embedding pickle path.
        manifest_path: Saved/loaded manifest path.
    """

    config: dict[str, object]
    seed: int
    graph: nx.Graph
    graph_type: str
    node_count: int
    root_node_index: int | None
    directed_edge_list: list[tuple[int, int]]
    run_name: str
    model: nn.Embedding
    checkpoint_path: str = ""
    embedding_history_path: str = ""
    topk_history_path: str = ""
    final_embeddings_path: str = ""
    manifest_path: str = ""


def _node2vec_defaults(config: Mapping[str, object]) -> dict[str, object]:
    """Applies Node2Vec defaults and returns a normalized config dict.

    Args:
        config: Partial config mapping supplied by notebook cell.

    Returns:
        dict[str, object]: Normalized config with defaults filled.
    """

    normalized = dict(config)
    normalized.setdefault("graph_type", "star")
    normalized.setdefault("star_degree", 4)
    normalized.setdefault("path_length", 5)
    normalized.setdefault("grid_rows", 4)
    normalized.setdefault("grid_cols", 4)
    normalized.setdefault("total_nodes", 15)
    normalized.setdefault("irregular_edge_count", FIXED_IRREGULAR_EDGE_COUNT)

    if str(normalized.get("graph_type")) == "irregular":
        if "total_nodes" not in config:
            normalized["total_nodes"] = FIXED_IRREGULAR_NODE_COUNT
        if "irregular_edge_count" not in config:
            normalized["irregular_edge_count"] = FIXED_IRREGULAR_EDGE_COUNT

    normalized.setdefault("embedding_dim", 100)
    normalized.setdefault("add_self_edges", False)
    normalized.setdefault("learning_rate", 0.01)
    normalized.setdefault("num_epochs", 10_000)
    normalized.setdefault("neg_samples_per_pos", 3)
    normalized.setdefault("use_full_softmax_objective", True)
    normalized.setdefault("embedding_checkpoint_interval", 25)
    normalized.setdefault("top_k", 5)
    return normalized


def _build_run_name(config: Mapping[str, object], seed: int) -> str:
    """Builds a compact node2vec run name for artifact files.

    Args:
        config: Normalized config mapping.
        seed: Deterministic seed.

    Returns:
        str: Run-name slug.
    """

    graph_type = str(config["graph_type"])
    emb_dim = int(config["embedding_dim"])
    lr = float(config["learning_rate"])
    epochs = int(config["num_epochs"])
    add_self_edges = int(bool(config.get("add_self_edges", False)))
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    lr_tag = str(lr).replace(".", "p")
    return (
        f"node2vec_{graph_type}_selfedge{add_self_edges}_"
        f"tied_d{emb_dim}_lr{lr_tag}_e{epochs}_s{seed}_{timestamp}"
    )


def _artifact_dirs(context: Node2VecSectionContext) -> dict[str, Path]:
    """Resolves and creates node2vec artifact directories.

    Args:
        context: Section context.

    Returns:
        dict[str, Path]: Named artifact directories.
    """

    root = get_tiny_graphs_root() / "saved_artifacts"
    graph_scope = str(context.graph_type)
    paths = {
        "root": root,
        "checkpoints": root / "checkpoints" / graph_scope,
        "pickles": root / "pickles" / graph_scope,
        "manifests": root / "manifests" / graph_scope,
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def _is_node2vec_manifest(payload: Mapping[str, object]) -> bool:
    """Checks whether a manifest payload belongs to node2vec.

    Args:
        payload: Manifest JSON payload.

    Returns:
        bool: True for node2vec manifests.
    """
    payload_family = str(payload.get("model_family", "")).strip()
    if payload_family:
        return payload_family == 'node2vec'

    # Backfill for older manifests without `model_family` metadata.
    run_name = str(payload.get("run_name", ""))
    checkpoint_name = Path(str(payload.get("checkpoint_path", ""))).name
    return run_name.startswith('node2vec_') or checkpoint_name.startswith('node2vec_')


def _manifest_matches_node2vec_context(
    payload: Mapping[str, object],
    context: Node2VecSectionContext,
) -> bool:
    """Checks whether a node2vec manifest matches the current self-edge variant."""
    if not _is_node2vec_manifest(payload):
        return False

    expected_self_edges = bool(context.config.get("add_self_edges", False))
    if "add_self_edges" in payload:
        return bool(payload.get("add_self_edges", False)) == expected_self_edges

    run_name = str(payload.get("run_name", ""))
    expected_token = f"_selfedge{int(expected_self_edges)}_"
    if expected_self_edges:
        return expected_token in run_name
    return expected_token in run_name or "_selfedge1_" not in run_name


def _find_latest_node2vec_manifest(
    manifest_dir: Path,
    context: Node2VecSectionContext,
) -> Path | None:
    """Finds the newest node2vec manifest in a graph-scoped manifest directory.

    Args:
        manifest_dir: Directory containing graph-scoped manifests.

    Returns:
        Path | None: Most recent compatible manifest or None.
    """
    matched: list[Path] = []
    for manifest_path in manifest_dir.glob('*.json'):
        payload = load_json(manifest_path)
        if _manifest_matches_node2vec_context(payload, context):
            matched.append(manifest_path)
    if not matched:
        return None
    matched.sort(key=lambda p: p.stat().st_mtime)
    return matched[-1]


def _find_node2vec_manifest_by_checkpoint(
    manifest_dir: Path,
    checkpoint_path: str,
    context: Node2VecSectionContext,
) -> Path | None:
    """Finds a node2vec manifest whose checkpoint path matches the input path.

    Args:
        manifest_dir: Directory containing graph-scoped manifests.
        checkpoint_path: Checkpoint path to resolve.

    Returns:
        Path | None: Matching manifest path or None.
    """
    checkpoint_resolved = str(Path(checkpoint_path).resolve())
    for manifest_path in manifest_dir.glob('*.json'):
        payload = load_json(manifest_path)
        if not _manifest_matches_node2vec_context(payload, context):
            continue
        payload_checkpoint = str(Path(str(payload.get('checkpoint_path', ''))).resolve())
        if payload_checkpoint == checkpoint_resolved:
            return manifest_path
    return None


def _allow_self_predictions(context: Node2VecSectionContext) -> bool:
    """Returns whether node2vec top-k metrics should allow predicting self."""
    return bool(context.config.get("add_self_edges", False))


def build_node2vec_section_context(config: Mapping[str, object], seed: int) -> Node2VecSectionContext:
    """Builds a reproducible Node2Vec notebook section context.

    Args:
        config: Section config dictionary.
        seed: Deterministic random seed.

    Returns:
        Node2VecSectionContext: Prepared section context.
    """

    normalized = _node2vec_defaults(config)
    set_all_seeds(seed)
    graph, root_node_index = build_tiny_graph_from_config(normalized)
    node_count = graph.number_of_nodes()
    model = nn.Embedding(node_count, int(normalized["embedding_dim"]))
    run_name = _build_run_name(normalized, seed)
    return Node2VecSectionContext(
        config=normalized,
        seed=int(seed),
        graph=graph,
        graph_type=str(normalized["graph_type"]),
        node_count=node_count,
        root_node_index=root_node_index,
        directed_edge_list=build_bidirectional_edge_list(graph),
        run_name=run_name,
        model=model,
    )


def _train_node2vec_tied(
    context: Node2VecSectionContext,
) -> tuple[dict[int, np.ndarray], dict[int, float], list[float]]:
    """Trains tied Node2Vec embeddings and records histories.

    Args:
        context: Prepared section context.

    Returns:
        tuple[dict[int, np.ndarray], dict[int, float], list[float]]:
            Embedding history, top-k recovery history, and epoch losses.
    """

    config = context.config
    graph = context.graph
    node_count = context.node_count
    model = context.model

    lr = float(config["learning_rate"])
    num_epochs = int(config["num_epochs"])
    use_full_softmax = bool(config["use_full_softmax_objective"])
    neg_samples_per_pos = int(config["neg_samples_per_pos"])
    checkpoint_interval = max(1, int(config["embedding_checkpoint_interval"]))
    top_k = int(config["top_k"])

    embedding_history: dict[int, np.ndarray] = {}
    topk_history: dict[int, float] = {}
    epoch_losses: list[float] = []

    node_indices = list(range(node_count))
    for epoch in tqdm(range(1, num_epochs + 1), desc=f"Node2Vec ({context.graph_type})"):
        node_losses: list[torch.Tensor] = []
        for src in graph.nodes():
            src_idx = int(src)
            neighbors = [int(n) for n in graph.neighbors(src_idx)]
            if not neighbors:
                continue

            src_emb = model(torch.tensor(src_idx, dtype=torch.long))
            all_emb = model.weight
            logits = torch.matmul(src_emb, all_emb.T)
            neighbor_tensor = torch.tensor(neighbors, dtype=torch.long)

            if use_full_softmax:
                loss = -F.log_softmax(logits, dim=0)[neighbor_tensor].mean()
            else:
                candidate_neg = [i for i in node_indices if i != src_idx and i not in neighbors]
                if not candidate_neg:
                    continue
                neg_count = max(1, min(len(candidate_neg), neg_samples_per_pos * len(neighbors)))
                replace = len(candidate_neg) < neg_count
                neg_indices = np.random.choice(candidate_neg, size=neg_count, replace=replace).tolist()
                pos_emb = model(neighbor_tensor)
                neg_emb = model(torch.tensor(neg_indices, dtype=torch.long))
                pos_loss = -torch.log(torch.sigmoid(torch.sum(src_emb * pos_emb, dim=1)) + 1e-8).mean()
                neg_loss = -torch.log(torch.sigmoid(-torch.sum(src_emb * neg_emb, dim=1)) + 1e-8).mean()
                loss = pos_loss + neg_loss

            node_losses.append(loss)

        if not node_losses:
            epoch_losses.append(0.0)
            continue

        total_loss = torch.stack(node_losses).sum()
        model.zero_grad(set_to_none=True)
        total_loss.backward()
        with torch.no_grad():
            model.weight -= lr * model.weight.grad
        epoch_losses.append(float(total_loss.item() / len(node_losses)))

        should_checkpoint = (epoch % checkpoint_interval == 0) or (epoch == num_epochs)
        if should_checkpoint:
            emb_np = model.weight.detach().cpu().numpy().copy()
            embedding_history[int(epoch)] = emb_np
            topk_predictions = topk_predictions_from_embeddings_allowing_self(
                emb_np,
                k=top_k,
                allow_self_predictions=_allow_self_predictions(context),
            )
            topk_history[int(epoch)] = topk_recovery_percent(
                topk_predictions=topk_predictions,
                directed_edges=context.directed_edge_list,
            )

    return embedding_history, topk_history, epoch_losses


def _persist_artifacts(
    context: Node2VecSectionContext,
    embedding_history: dict[int, np.ndarray],
    topk_history: dict[int, float],
    epoch_losses: list[float],
) -> str:
    """Persists checkpoint, histories, and manifest for a section run.

    Args:
        context: Section context.
        embedding_history: Step-indexed embedding history.
        topk_history: Step-indexed top-k recovery history.
        epoch_losses: Epoch loss list.

    Returns:
        str: Resolved checkpoint path.
    """

    dirs = _artifact_dirs(context)
    checkpoint_path = dirs["checkpoints"] / f"{context.run_name}_final.pt"
    embedding_history_path = dirs["pickles"] / f"{context.run_name}_embedding_history.pkl"
    topk_history_path = dirs["pickles"] / f"{context.run_name}_topk_recovery.pkl"
    final_embeddings_path = dirs["pickles"] / f"{context.run_name}_final_embeddings.pkl"
    manifest_path = dirs["manifests"] / f"{context.run_name}.json"

    torch.save(context.model.state_dict(), checkpoint_path)
    write_pickle(embedding_history_path, embedding_history)
    write_pickle(topk_history_path, topk_history)
    write_pickle(final_embeddings_path, context.model.weight.detach().cpu().numpy().copy())

    manifest_payload = {
        "run_name": context.run_name,
        "graph_type": context.graph_type,
        "model_family": "node2vec",
        "model_variant": "tied",
        "add_self_edges": bool(context.config.get("add_self_edges", False)),
        "seed": int(context.seed),
        "config": context.config,
        "node_count": int(context.node_count),
        "checkpoint_path": str(checkpoint_path),
        "embedding_history_path": str(embedding_history_path),
        "topk_history_path": str(topk_history_path),
        "final_embeddings_path": str(final_embeddings_path),
        "epoch_losses": epoch_losses,
    }
    save_json(manifest_path, manifest_payload)

    context.checkpoint_path = str(checkpoint_path)
    context.embedding_history_path = str(embedding_history_path)
    context.topk_history_path = str(topk_history_path)
    context.final_embeddings_path = str(final_embeddings_path)
    context.manifest_path = str(manifest_path)
    return str(checkpoint_path)


def train_or_load_node2vec_model(
    context: Node2VecSectionContext,
    *,
    train_from_scratch: bool,
    checkpoint_path: str = "",
) -> str:
    """Trains a tied Node2Vec model or loads it from a checkpoint.

    Args:
        context: Prepared section context.
        train_from_scratch: Whether to run training.
        checkpoint_path: Optional explicit checkpoint path for loading.

    Returns:
        str: Resolved checkpoint path.
    """

    dirs = _artifact_dirs(context)
    if train_from_scratch:
        embedding_history, topk_history, epoch_losses = _train_node2vec_tied(context)
        return _persist_artifacts(context, embedding_history, topk_history, epoch_losses)

    resolved_checkpoint = checkpoint_path.strip()
    manifest_payload: dict[str, object] | None = None
    if not resolved_checkpoint:
        latest_manifest = _find_latest_node2vec_manifest(dirs["manifests"], context)
        if latest_manifest is None:
            raise ValueError(
                "No prior node2vec manifest found. Set train_from_scratch=True "
                "or provide checkpoint_path explicitly."
            )
        manifest_payload = load_json(latest_manifest)
        resolved_checkpoint = str(manifest_payload.get("checkpoint_path", "")).strip()
        context.manifest_path = str(latest_manifest)

    if not resolved_checkpoint:
        raise ValueError("Unable to resolve checkpoint_path for node2vec loading.")

    context.model.load_state_dict(torch.load(resolved_checkpoint, map_location="cpu"))
    context.model.eval()
    context.checkpoint_path = str(Path(resolved_checkpoint))

    if manifest_payload is None:
        matched_manifest = _find_node2vec_manifest_by_checkpoint(
            dirs["manifests"],
            resolved_checkpoint,
            context,
        )
        if matched_manifest is not None:
            manifest_payload = load_json(matched_manifest)
            context.manifest_path = str(matched_manifest)

    if manifest_payload is not None:
        context.embedding_history_path = str(manifest_payload.get("embedding_history_path", ""))
        context.topk_history_path = str(manifest_payload.get("topk_history_path", ""))
        context.final_embeddings_path = str(manifest_payload.get("final_embeddings_path", ""))

    return context.checkpoint_path


def evaluate_node2vec_edges(
    context: Node2VecSectionContext,
) -> dict[str, float]:
    """Evaluates top-k and top-1 directed-edge recovery for tied node2vec.

    Args:
        context: Prepared section context.

    Returns:
        dict[str, float]: Evaluation metrics dictionary.
    """

    emb_np = context.model.weight.detach().cpu().numpy().copy()
    top_k = int(context.config.get("top_k", 5))
    topk_predictions = topk_predictions_from_embeddings_allowing_self(
        emb_np,
        k=top_k,
        allow_self_predictions=_allow_self_predictions(context),
    )
    top1_predictions = topk_predictions_from_embeddings_allowing_self(
        emb_np,
        k=1,
        allow_self_predictions=_allow_self_predictions(context),
    )

    topk_percent = topk_recovery_percent(topk_predictions, context.directed_edge_list)
    top1_percent = topk_recovery_percent(top1_predictions, context.directed_edge_list)
    return {
        "topk_edge_recovery_percent": float(topk_percent),
        "top1_edge_recovery_percent": float(top1_percent),
        "path_train_accuracy_percent": float("nan"),
        "path_test_accuracy_percent": float("nan"),
    }


def print_node2vec_evaluation_report(section_name: str, metrics: Mapping[str, float]) -> None:
    """Prints a clean node2vec evaluation summary.

    Args:
        section_name: Human-readable section title.
        metrics: Metrics from `evaluate_node2vec_edges`.

    Returns:
        None: Summary is printed.
    """

    print("=" * 72)
    print(f"{section_name} | Node2Vec (Tied) Evaluation Summary")
    print("-" * 72)
    print(f"Top-k edge recovery : {metrics['topk_edge_recovery_percent']:.2f}%")
    print(f"Top-1 edge recovery : {metrics['top1_edge_recovery_percent']:.2f}%")
    print("Path train accuracy : N/A (not trained in node2vec mode)")
    print("Path test accuracy  : N/A (not trained in node2vec mode)")
    print("=" * 72)


def collect_node2vec_embeddings_and_edges(
    context: Node2VecSectionContext,
) -> tuple[np.ndarray, list[tuple[int, int]], int | None]:
    """Collects current embeddings and edge metadata for plotting.

    Args:
        context: Section context.

    Returns:
        tuple[np.ndarray, list[tuple[int, int]], int | None]:
            Embedding matrix, directed edge list, and optional root node index.
    """

    emb_np = context.model.weight.detach().cpu().numpy().copy()
    return emb_np, list(context.directed_edge_list), context.root_node_index


def load_node2vec_analysis_histories(
    context: Node2VecSectionContext,
    *,
    embedding_history_path: str = "",
    topk_history_path: str = "",
) -> tuple[dict[int, np.ndarray], dict[int, float]]:
    """Loads persisted node2vec histories for analysis plots.

    Args:
        context: Section context.
        embedding_history_path: Optional explicit embedding-history path.
        topk_history_path: Optional explicit top-k history path.

    Returns:
        tuple[dict[int, np.ndarray], dict[int, float]]:
            Histories keyed by integer training step.
    """

    emb_path = embedding_history_path or context.embedding_history_path
    topk_path = topk_history_path or context.topk_history_path

    embedding_history: dict[int, np.ndarray] = {}
    topk_history: dict[int, float] = {}

    if emb_path and Path(emb_path).exists():
        raw = read_pickle(emb_path)
        embedding_history = {int(step): np.asarray(val) for step, val in raw.items()}

    if topk_path and Path(topk_path).exists():
        raw = read_pickle(topk_path)
        topk_history = {int(step): float(val) for step, val in raw.items()}

    return embedding_history, topk_history
