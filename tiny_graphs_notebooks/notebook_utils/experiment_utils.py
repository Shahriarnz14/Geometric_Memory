"""Shared tiny-graph notebook helpers for edge-memorization experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence, Tuple
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from geometric_memory.data.dataset_naming import resolve_graph_dataset_directory
from geometric_memory.data.build_in_weights_datasets import main as build_datasets_main
from geometric_memory.in_weights.config import get_args
from geometric_memory.in_weights.data_loader import (
    EdgeMemorizationDataset,
    build_dataset_split_path,
)
from geometric_memory.in_weights.evaluation import evaluate_edge_memorization
from geometric_memory.in_weights.trainer import run_edge_memorization_training
from geometric_memory.in_weights.utils import (
    build_batched_prefix_tensor,
    build_edge_prefix_strings_for_all_nodes,
    get_node_embeddings,
    get_top_k_predictions_for_all_nodes,
)
from geometric_memory.models import get_model
from geometric_memory.tokenizing import get_tokenizer
from geometric_memory.utils.run_management import build_run_name, prepare_run_directories
from geometric_memory.utils.device import resolve_default_device
from tiny_graphs_notebooks.notebook_utils.paths import resolve_from_tiny_graphs_root


DEFAULT_NOTEBOOK_ARGS = [
    "--experiment_log_root",
    "./experiment_logs",
    "--training_recipe",
    "staged_full_path",
    "--dataset_root",
    "data/datasets/in_weights_graphs",
    "--dataset_name",
    "in_weights",
    "--train_split_ratio",
    "0.75",
    "--add_forward_edges",
    "--add_backward_edges",
    "--edge_memorization_batch_size",
    "2048",
    "--path_finetuning_batch_size",
    "2048",
    "--edge_memorization_epochs",
    "500",
    "--path_finetuning_epochs",
    "0",
    "--edge_memorization_learning_rate",
    "0.01",
    "--path_finetuning_learning_rate",
    "0.001",
    "--optimizer_weight_decay",
    "0.01",
    "--edge_memorization_eval_interval_epochs",
    "50",
    "--path_finetuning_eval_interval_epochs",
    "1",
    "--checkpoint_interval_epochs",
    "2000",
    "--edge_memorization_warmup_steps",
    "0",
    "--path_finetuning_warmup_steps",
    "0",
    "--disable_edge_memorization_lr_decay",
    "--disable_path_finetuning_lr_decay",
]


def _normalize_runtime_paths(args):
    """Recomputes derived filesystem paths after notebook-root normalization."""
    args.dataset_root = resolve_from_tiny_graphs_root(args.dataset_root)
    args.experiment_log_root = resolve_from_tiny_graphs_root(args.experiment_log_root)
    args.dataset_directory = resolve_graph_dataset_directory(
        dataset_root=Path(args.dataset_root),
        graph_type=args.graph_type,
    )

    if getattr(args, "track_embedding_evolution", False):
        args.embedding_evolution_dir = Path(args.experiment_log_root) / "embedding_evolution"
        args.embedding_evolution_dir.mkdir(parents=True, exist_ok=True)

    if getattr(args, "track_top_k_predictions", False):
        args.top_k_prediction_dir = Path(args.experiment_log_root) / "top_k_predictions"
        args.top_k_prediction_dir.mkdir(parents=True, exist_ok=True)

    return args


@dataclass
class TinyGraphRunArtifacts:
    """Outputs produced by a tiny-graph edge-memorization run.

    Args:
        section_name: Notebook section identifier.
        args: Parsed CLI/config namespace used for the run.
        model: Trained or loaded model instance.
        tokenizer: Tokenizer used by the model/datasets.
        device: Runtime device string (`"cpu"`, `"mps"`, or `"cuda"`).
        pretrain_path: Absolute or relative path to edge pretrain text file.
        train_path: Path-finetuning train split path (created for consistency).
        test_path: Path-finetuning test split path (created for consistency).
        checkpoint_path: Final model checkpoint used by the section.
        edges: Directed edge list parsed from the pretrain file.
        root_node_index: Root node id for star/path-star graphs, when detectable.
        final_node_embeddings: Final node embedding matrix `[num_nodes, dim]`.
        final_topk_recovery_percent: Mean per-node top-k edge recovery in percent.
        embedding_history: Optional mapping of step -> node embeddings.
        topk_recovery_history: Optional mapping of step -> top-k recovery percent.
    """

    section_name: str
    args: object
    model: torch.nn.Module
    tokenizer: object
    device: str
    pretrain_path: str
    train_path: str
    test_path: str
    checkpoint_path: str
    edges: list[Tuple[int, int]]
    root_node_index: int | None
    final_node_embeddings: np.ndarray
    final_topk_recovery_percent: float
    embedding_history: Dict[int, np.ndarray]
    topk_recovery_history: Dict[int, float]


def set_global_seed(seed: int) -> None:
    """Sets all common PRNG seeds for reproducibility.

    Args:
        seed: Integer random seed.

    Returns:
        None: Sets random-state side effects globally.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _prepare_runtime(args) -> tuple[object, str]:
    """Initializes tokenizer and derived runtime fields on parsed args.

    Args:
        args: Namespace returned by `get_args`.

    Returns:
        tuple[object, str]: `(tokenizer, device)` for subsequent training/eval.
    """
    device = resolve_default_device()
    tokenizer = get_tokenizer(args)
    args.device = device
    args.vocab_size = tokenizer.vocab_size
    args.block_size = max(64, args.path_length * 3)
    args.teacherless_token = tokenizer.encode("$")[0] if args.use_teacherless_inputs else None
    args.use_flash = False
    return tokenizer, device


def _build_args(
    section_args: Sequence[str],
    *,
    track_embedding_evolution: bool,
    track_top_k_predictions: bool,
):
    """Builds parsed args for a notebook section from default + custom CLI args.

    Args:
        section_args: Section-specific CLI args to append to defaults.
        track_embedding_evolution: Whether to save embedding snapshots.
        track_top_k_predictions: Whether to save top-k prediction snapshots.

    Returns:
        argparse.Namespace: Fully parsed and post-processed args.
    """
    args_list = list(DEFAULT_NOTEBOOK_ARGS)
    if track_embedding_evolution:
        args_list.append("--track_embedding_evolution")
    if track_top_k_predictions:
        args_list.append("--track_top_k_predictions")
    args_list.extend(section_args)
    return _normalize_runtime_paths(get_args(args_list))


def _build_dataset_generation_args(args, random_seed: int, overwrite: bool) -> list[str]:
    """Converts parsed training args into dataset-generation CLI args.

    Args:
        args: Parsed training args.
        random_seed: Random seed used for deterministic dataset generation.
        overwrite: Whether existing dataset files should be overwritten.

    Returns:
        list[str]: CLI args for `data/build_in_weights_datasets.py`.
    """
    dataset_args = [
        "--graph_type",
        args.graph_type,
        "--path_length",
        str(args.path_length),
        "--total_nodes",
        str(args.total_nodes),
        "--train_split_ratio",
        str(args.train_split_ratio),
        "--random_seed",
        str(random_seed),
        "--dataset_root",
        str(args.dataset_root),
        "--star_degree",
        str(args.star_degree),
        "--star_subtree_degree",
        str(args.star_subtree_degree),
        "--grid_rows",
        str(args.grid_rows),
        "--grid_cols",
        str(args.grid_cols),
        "--irregular_edge_count",
        str(args.irregular_edge_count),
    ]
    if args.add_forward_edges:
        dataset_args.append("--add_forward_edges")
    if args.add_backward_edges:
        dataset_args.append("--add_backward_edges")
    if args.include_start_node_in_path_finetuning:
        dataset_args.append("--include_start_node_in_path_finetuning")
    if args.use_directional_edge_pretraining:
        dataset_args.append("--use_directional_edge_pretraining")
    if args.split_subtree_holdout:
        dataset_args.append("--split_subtree_holdout")
    if overwrite:
        dataset_args.append("--overwrite")
    return dataset_args


def ensure_dataset_files(args, random_seed: int, overwrite: bool = False) -> tuple[str, str, str]:
    """Creates tiny-graph dataset files and returns split paths.

    Args:
        args: Parsed training args.
        random_seed: Seed used for deterministic dataset generation.
        overwrite: Whether to overwrite existing pretrain/train/test files.

    Returns:
        tuple[str, str, str]: `(pretrain_path, train_path, test_path)`.
    """
    dataset_cli_args = _build_dataset_generation_args(
        args,
        random_seed=random_seed,
        overwrite=overwrite,
    )
    try:
        build_datasets_main(dataset_cli_args)
    except FileExistsError as exc:
        print(
            "Dataset files already exist for this configuration. "
            "Reusing existing files.\n"
            f"Details: {exc}"
        )

    pretrain_path = build_dataset_split_path(args, "pretrain")
    train_path = build_dataset_split_path(args, f"train_{args.expected_train_path_count}")
    test_path = build_dataset_split_path(args, f"test_{args.expected_test_path_count}")
    return pretrain_path, train_path, test_path


def _prepare_run_layout(args, section_name: str) -> None:
    """Sets run-name and output directories required by trainer utilities.

    Args:
        args: Parsed training args namespace.
        section_name: Short section token to make notebook run names readable.

    Returns:
        None: Mutates `args` with run directory and artifact fields.
    """
    run_name = build_run_name(args, recipe_name=f"tiny_edge_{section_name}")
    run_dirs = prepare_run_directories(
        experiment_log_root=args.experiment_log_root,
        dataset=args.dataset_name,
        run_name=run_name,
    )
    args.run_name = run_name
    args.run_dir = str(run_dirs.run_dir)
    args.output_checkpoint_dir = str(run_dirs.checkpoints_dir)
    args.output_artifact_dir = str(run_dirs.artifacts_dir)


def _parse_edge_list(pretrain_path: str) -> list[Tuple[int, int]]:
    """Parses `u=v` edges from a pretrain text file.

    Args:
        pretrain_path: Dataset pretrain file path.

    Returns:
        list[tuple[int, int]]: Directed edges in file order.
    """
    edges: list[Tuple[int, int]] = []
    with open(pretrain_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if "=" not in line:
                continue
            src, dst = line.split("=", maxsplit=1)
            edges.append((int(src), int(dst)))
    return edges


def _infer_root_node(edges: Iterable[Tuple[int, int]]) -> int | None:
    """Returns the max out-degree node as a star-root proxy.

    Args:
        edges: Directed edge list.

    Returns:
        int | None: Detected root node id, or `None` when unavailable.
    """
    out_degree: Dict[int, int] = {}
    for src, _ in edges:
        out_degree[src] = out_degree.get(src, 0) + 1
    if not out_degree:
        return None
    return max(out_degree.items(), key=lambda item: item[1])[0]


def _build_edge_loader(args, tokenizer, device: str, pretrain_path: str) -> tuple[EdgeMemorizationDataset, DataLoader]:
    """Builds edge dataset and dataloader for training or evaluation.

    Args:
        args: Parsed training args.
        tokenizer: Tokenizer matching the model.
        device: Runtime device string.
        pretrain_path: Edge pretrain file path.

    Returns:
        tuple[EdgeMemorizationDataset, DataLoader]: Prepared dataset and loader.
    """
    edge_dataset = EdgeMemorizationDataset(
        tokenizer=tokenizer,
        data_path=pretrain_path,
        device=device,
        teacherless_token_id=args.teacherless_token,
        drop_pause_token=args.edge_memorization_drop_pause_token,
        include_task_token_in_prefix=args.include_task_token_in_prefix,
    )
    edge_loader = DataLoader(
        edge_dataset,
        batch_size=args.edge_memorization_batch_size,
        shuffle=True,
    )
    return edge_dataset, edge_loader


def _build_adjacency(edges: Iterable[Tuple[int, int]]) -> Dict[int, set[int]]:
    """Builds adjacency dictionary from directed edges.

    Args:
        edges: Directed edge list.

    Returns:
        dict[int, set[int]]: `adj[src] = {dst_1, dst_2, ...}`.
    """
    adjacency: Dict[int, set[int]] = {}
    for src, dst in edges:
        if src not in adjacency:
            adjacency[src] = set()
        adjacency[src].add(dst)
    return adjacency


def compute_topk_recovery_percent(topk_predictions: np.ndarray, edges: Iterable[Tuple[int, int]]) -> float:
    """Computes mean per-node fraction of true neighbors recovered by top-k.

    Args:
        topk_predictions: Integer matrix `[num_nodes, k]` of top-k predictions.
        edges: Directed edge list used as ground truth.

    Returns:
        float: Mean recovery fraction in percent over nodes with out-degree > 0.
    """
    adjacency = _build_adjacency(edges)
    per_node_scores = []
    for node_id, neighbors in adjacency.items():
        degree = len(neighbors)
        if degree == 0 or node_id >= topk_predictions.shape[0]:
            continue
        node_topk = set(topk_predictions[node_id, : min(degree, topk_predictions.shape[1])].tolist())
        recovered = sum(int(v in node_topk) for v in neighbors)
        per_node_scores.append(recovered / degree)
    if not per_node_scores:
        return 0.0
    return float(np.mean(per_node_scores) * 100.0)


def _build_topk_history_percent(
    topk_by_step: Mapping[int, np.ndarray],
    edges: Iterable[Tuple[int, int]],
) -> Dict[int, float]:
    """Converts saved top-k predictions-by-step into recovery-percent curve.

    Args:
        topk_by_step: Mapping `step -> [num_nodes, k]` top-k prediction arrays.
        edges: Directed edge list used as ground truth.

    Returns:
        dict[int, float]: Mapping `step -> recovery_percent`.
    """
    return {
        int(step): compute_topk_recovery_percent(topk_predictions=topk_matrix, edges=edges)
        for step, topk_matrix in topk_by_step.items()
    }


def _load_pickle_if_exists(path: Path | str | None):
    """Loads a pickle file when present.

    Args:
        path: Optional filesystem path.

    Returns:
        object: Unpickled object, or `None` when missing.
    """
    if not path:
        return None
    candidate = Path(path)
    if not candidate.exists():
        return None
    import pickle

    with candidate.open("rb") as f:
        return pickle.load(f)


def run_tiny_graph_edge_experiment(
    *,
    section_name: str,
    section_args: Sequence[str],
    random_seed: int = 0,
    train_from_scratch: bool = True,
    checkpoint_path: str = "",
    dataset_overwrite: bool = False,
    track_embedding_evolution: bool = True,
    track_top_k_predictions: bool = True,
    embedding_history_path: str = "",
    topk_history_path: str = "",
) -> TinyGraphRunArtifacts:
    """Runs one tiny-graph edge-memorization section end-to-end.

    Args:
        section_name: Short name for this notebook section.
        section_args: Section-specific CLI args (graph/model/training choices).
        random_seed: Seed for both dataset generation and model training.
        train_from_scratch: Whether to train model now or only load checkpoint.
        checkpoint_path: Existing checkpoint path used when not training.
        dataset_overwrite: Whether to force overwrite existing dataset files.
        track_embedding_evolution: Whether to save embedding history while training.
        track_top_k_predictions: Whether to save top-k history while training.
        embedding_history_path: Optional override path to saved embedding history.
        topk_history_path: Optional override path to saved top-k history.

    Returns:
        TinyGraphRunArtifacts: Aggregated runtime outputs for analysis and plotting.
    """
    set_global_seed(random_seed)

    args = _build_args(
        section_args=section_args,
        track_embedding_evolution=track_embedding_evolution,
        track_top_k_predictions=track_top_k_predictions,
    )
    tokenizer, device = _prepare_runtime(args)

    pretrain_path, train_path, test_path = ensure_dataset_files(
        args=args,
        random_seed=random_seed,
        overwrite=dataset_overwrite,
    )
    _prepare_run_layout(args, section_name=section_name)

    model = get_model(args).to(device)
    edge_dataset, edge_loader = _build_edge_loader(
        args=args,
        tokenizer=tokenizer,
        device=device,
        pretrain_path=pretrain_path,
    )

    resolved_checkpoint_path = checkpoint_path
    if train_from_scratch:
        model = run_edge_memorization_training(
            model=model,
            edge_train_loader=edge_loader,
            args=args,
            device=device,
            tokenizer=tokenizer,
            logging_enabled=False,
            experiment_logger=None,
        )
        resolved_checkpoint_path = str(
            Path(args.output_checkpoint_dir) / f"{args.run_name}_edge_memorization_final.pt"
        )
        torch.save(model.state_dict(), resolved_checkpoint_path)
    else:
        if not checkpoint_path:
            raise ValueError(
                f"Section '{section_name}' requested load-only mode but checkpoint_path is empty."
            )
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.to(device)

    eval_loader = DataLoader(edge_dataset, batch_size=min(1024, args.edge_memorization_batch_size), shuffle=False)
    evaluate_edge_memorization(
        model=model,
        loader=eval_loader,
        device=device,
        logging_enabled=False,
        experiment_logger=None,
        during_edge_memorization=False,
    )

    edges = _parse_edge_list(pretrain_path)
    prefixes = build_edge_prefix_strings_for_all_nodes(args)
    x_topk = build_batched_prefix_tensor(tokenizer, device, prefixes)
    topk_predictions = get_top_k_predictions_for_all_nodes(model, x_topk=x_topk, k=5)
    final_topk_recovery_percent = compute_topk_recovery_percent(topk_predictions, edges=edges)

    final_node_embeddings = get_node_embeddings(model, num_nodes=args.total_nodes)
    root_node_index = _infer_root_node(edges) if args.graph_type == "star" else None

    embedding_history = {}
    topk_recovery_history = {}

    embedded_hist_data = _load_pickle_if_exists(embedding_history_path)
    if embedded_hist_data is None and track_embedding_evolution:
        embedded_hist_data = _load_pickle_if_exists(
            Path(args.embedding_evolution_dir) / args.embedding_evolution_filename
        )
    if embedded_hist_data:
        embedding_history = {int(step): np.asarray(value) for step, value in embedded_hist_data.items()}

    topk_hist_data = _load_pickle_if_exists(topk_history_path)
    if topk_hist_data is None and track_top_k_predictions:
        topk_hist_data = _load_pickle_if_exists(
            Path(args.top_k_prediction_dir) / args.top_k_prediction_filename
        )
    if topk_hist_data:
        topk_recovery_history = _build_topk_history_percent(topk_hist_data, edges=edges)

    if not topk_recovery_history:
        topk_recovery_history = {0: final_topk_recovery_percent}
    if not embedding_history:
        embedding_history = {0: final_node_embeddings}

    return TinyGraphRunArtifacts(
        section_name=section_name,
        args=args,
        model=model,
        tokenizer=tokenizer,
        device=device,
        pretrain_path=pretrain_path,
        train_path=train_path,
        test_path=test_path,
        checkpoint_path=resolved_checkpoint_path,
        edges=edges,
        root_node_index=root_node_index,
        final_node_embeddings=final_node_embeddings,
        final_topk_recovery_percent=final_topk_recovery_percent,
        embedding_history=embedding_history,
        topk_recovery_history=topk_recovery_history,
    )
