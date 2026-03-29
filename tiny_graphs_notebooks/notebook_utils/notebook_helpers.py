"""Shared helper utilities for self-contained tiny-graph notebook sections."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
import shutil
from typing import Mapping, Sequence
from datetime import datetime, timezone

import numpy as np
import torch
from torch.utils.data import DataLoader

from geometric_memory.data.dataset_naming import resolve_graph_dataset_directory
from geometric_memory.data.build_in_weights_datasets import main as build_datasets_main
from geometric_memory.in_weights.config import get_args
from geometric_memory.in_weights.data_loader import (
    EdgeMemorizationDataset,
    PathFinetuningDataset,
    build_dataset_split_path,
)
from geometric_memory.in_weights.evaluation import evaluate, evaluate_edge_memorization
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
from geometric_memory.utils.device import is_cuda_device, resolve_default_device
from tiny_graphs_notebooks.analysis.analysis import (
    build_reduced_evolution_snapshots,
    compute_associative_geometric_curves,
    get_star_branch_layout,
    plot_associative_vs_geometric_curves,
    plot_ordered_heatmaps,
    plot_stylized_embedding_graph,
    plot_three_snapshot_evolution,
    reduce_embeddings_for_plot,
    resolve_evolution_steps,
    resolve_embedding_step,
    select_embedding_snapshot,
)
from tiny_graphs_notebooks.analysis.reproducibility import set_all_seeds
from tiny_graphs_notebooks.analysis.storage import (
    load_json,
    read_pickle,
    save_json,
    write_pickle,
)
from tiny_graphs_notebooks.notebook_utils.experiment_utils import (
    compute_topk_recovery_percent,
)
from tiny_graphs_notebooks.notebook_utils.paths import (
    get_tiny_graphs_root,
    resolve_from_tiny_graphs_root,
)


@dataclass
class TransformerSectionContext:
    """Container for one notebook section runtime state.

    Args:
        args: Parsed CLI argument namespace for this section.
        device: Runtime device string (`"cpu"`, `"mps"`, or `"cuda"`).
        tokenizer: Tokenizer object used for model/data.
        model: Initialized sequence model.
        pretrain_path: Edge memorization file path.
        train_path: Path task train file path.
        test_path: Path task test file path.
        edge_dataset: Edge memorization dataset object.
        edge_train_loader: Edge training dataloader.
        edge_eval_loader: Edge evaluation dataloader.
        path_train_loader: Path train dataloader.
        path_test_loader: Path test dataloader.
        checkpoint_path: Current resolved checkpoint path.
        embedding_history_path: Optional saved embedding-history pickle path.
        topk_history_path: Optional saved top-k prediction pickle path.
        final_embeddings_path: Optional saved final-embeddings pickle path.
        manifest_path: Optional saved run-manifest JSON path.
    """

    args: object
    device: str
    tokenizer: object
    model: torch.nn.Module
    pretrain_path: str
    train_path: str
    test_path: str
    edge_dataset: EdgeMemorizationDataset
    edge_train_loader: DataLoader
    edge_eval_loader: DataLoader
    path_train_loader: DataLoader
    path_test_loader: DataLoader
    checkpoint_path: str
    embedding_history_path: str = ""
    topk_history_path: str = ""
    final_embeddings_path: str = ""
    manifest_path: str = ""


def args_dict_to_cli_list(args_dict: Mapping[str, object]) -> list[str]:
    """Converts an ordered CLI-config dictionary into a flat argument list.

    Args:
        args_dict: Dictionary whose keys are CLI switches (e.g., `--flag`).

    Returns:
        list[str]: Flat list consumable by `get_args`.
    """
    cli_args: list[str] = []
    for key, value in args_dict.items():
        if isinstance(value, bool):
            if value:
                cli_args.append(key)
            continue
        if value is None:
            continue
        cli_args.extend([key, str(value)])
    return cli_args


def _build_dataset_generation_args(args, seed: int, overwrite: bool) -> list[str]:
    """Builds dataset generation CLI args from a parsed training config.

    Args:
        args: Parsed args namespace.
        seed: Deterministic dataset seed.
        overwrite: Whether to overwrite existing files.

    Returns:
        list[str]: CLI args for dataset builder.
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
        str(seed),
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
    if getattr(args, "add_self_edges", False):
        dataset_args.append("--add_self_edges")
    if args.include_start_node_in_path_finetuning:
        dataset_args.append("--include_start_node_in_path_finetuning")
    if args.use_directional_edge_pretraining:
        dataset_args.append("--use_directional_edge_pretraining")
    if args.split_subtree_holdout:
        dataset_args.append("--split_subtree_holdout")
    if overwrite:
        dataset_args.append("--overwrite")
    return dataset_args


def _normalize_runtime_paths(args):
    """Re-resolves notebook paths against `tiny_graphs_notebooks` root.

    `get_args` derives dataset and analysis-output directories immediately, so
    when notebooks run from a deeper working directory we need to recompute the
    derived paths after normalizing the roots.
    """
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


def build_transformer_section_context(
    cli_config: Mapping[str, object],
    seed: int,
    *,
    dataset_overwrite: bool = False,
):
    """Builds full section context: args, files, datasets, model, and loaders.

    Args:
        cli_config: Section config dict (`--arg`: value) for `get_args`.
        seed: Random seed used for runtime and dataset generation.
        dataset_overwrite: Whether to overwrite dataset files.

    Returns:
        TransformerSectionContext: Prepared section context object.
    """
    set_all_seeds(seed)

    args = _normalize_runtime_paths(get_args(args_dict_to_cli_list(cli_config)))
    device = resolve_default_device()
    tokenizer = get_tokenizer(args)
    args.device = device
    args.vocab_size = tokenizer.vocab_size
    args.block_size = max(64, args.path_length * 3)
    args.teacherless_token = tokenizer.encode("$")[0] if args.use_teacherless_inputs else None
    args.use_flash = False

    architecture_label = str(getattr(args, "model_architecture_label", "transformer"))
    run_name = build_run_name(
        args,
        recipe_name=f"tiny_{args.graph_type}_{architecture_label}_edge",
    )
    run_dirs = prepare_run_directories(
        experiment_log_root=args.experiment_log_root,
        dataset=args.dataset_name,
        run_name=run_name,
    )
    args.run_name = run_name
    args.run_dir = str(run_dirs.run_dir)
    args.output_checkpoint_dir = str(run_dirs.checkpoints_dir)
    args.output_artifact_dir = str(run_dirs.artifacts_dir)

    dataset_generation_args = _build_dataset_generation_args(
        args,
        seed=seed,
        overwrite=dataset_overwrite,
    )
    try:
        build_datasets_main(dataset_generation_args)
    except FileExistsError as exc:
        print("Dataset files already exist. Reusing files for this section.")
        print(exc)

    pretrain_path = build_dataset_split_path(args, "pretrain")
    train_path = build_dataset_split_path(args, f"train_{args.expected_train_path_count}")
    test_path = build_dataset_split_path(args, f"test_{args.expected_test_path_count}")

    edge_dataset = EdgeMemorizationDataset(
        tokenizer=tokenizer,
        data_path=pretrain_path,
        device=device,
        teacherless_token_id=args.teacherless_token,
        drop_pause_token=args.edge_memorization_drop_pause_token,
        include_task_token_in_prefix=args.include_task_token_in_prefix,
    )
    edge_train_loader = DataLoader(
        edge_dataset,
        batch_size=args.edge_memorization_batch_size,
        shuffle=True,
    )
    edge_eval_loader = DataLoader(
        edge_dataset,
        batch_size=min(1024, args.edge_memorization_batch_size),
        shuffle=False,
    )

    path_train_dataset = PathFinetuningDataset(
        tokenizer=tokenizer,
        data_path=train_path,
        device=device,
        teacherless_token_id=args.teacherless_token,
        reverse_path_targets=args.reverse_path_targets,
        path_prefix_pause_token_count=args.path_prefix_pause_token_count,
        predict_full_path=True,
        include_task_token_in_prefix=args.include_task_token_in_prefix,
    )
    path_test_dataset = PathFinetuningDataset(
        tokenizer=tokenizer,
        data_path=test_path,
        device=device,
        teacherless_token_id=args.teacherless_token,
        reverse_path_targets=args.reverse_path_targets,
        path_prefix_pause_token_count=args.path_prefix_pause_token_count,
        predict_full_path=True,
        include_task_token_in_prefix=args.include_task_token_in_prefix,
    )
    path_train_loader = DataLoader(
        path_train_dataset,
        batch_size=args.path_finetuning_batch_size,
        shuffle=False,
    )
    path_test_loader = DataLoader(
        path_test_dataset,
        batch_size=args.path_finetuning_batch_size,
        shuffle=False,
    )

    model = get_model(args).to(device)

    return TransformerSectionContext(
        args=args,
        device=device,
        tokenizer=tokenizer,
        model=model,
        pretrain_path=pretrain_path,
        train_path=train_path,
        test_path=test_path,
        edge_dataset=edge_dataset,
        edge_train_loader=edge_train_loader,
        edge_eval_loader=edge_eval_loader,
        path_train_loader=path_train_loader,
        path_test_loader=path_test_loader,
        checkpoint_path="",
    )


# Shared aliases for future non-transformer tiny-graph notebooks.
NotebookSectionContext = TransformerSectionContext


def build_section_context(
    cli_config: Mapping[str, object],
    seed: int,
    *,
    dataset_overwrite: bool = False,
) -> TransformerSectionContext:
    """Builds a generic tiny-graph notebook section context.

    Args:
        cli_config: Section config dict (`--arg`: value) for `get_args`.
        seed: Random seed used for runtime and dataset generation.
        dataset_overwrite: Whether to overwrite dataset files.

    Returns:
        TransformerSectionContext: Prepared section context object.
    """
    return build_transformer_section_context(
        cli_config=cli_config,
        seed=seed,
        dataset_overwrite=dataset_overwrite,
    )


def _tiny_notebook_artifact_root() -> Path:
    """Returns the base folder for notebook checkpoints and plot pickles."""
    return get_tiny_graphs_root() / "saved_artifacts"


def _tiny_notebook_artifact_dirs(context: TransformerSectionContext) -> dict[str, Path]:
    """Builds and creates tiny-notebook artifact directories for this graph scope."""
    graph_scope = str(context.args.graph_type)
    root = _tiny_notebook_artifact_root()
    checkpoint_dir = root / "checkpoints" / graph_scope
    pickle_dir = root / "pickles" / graph_scope
    manifest_dir = root / "manifests" / graph_scope
    for path in (checkpoint_dir, pickle_dir, manifest_dir):
        path.mkdir(parents=True, exist_ok=True)
    return {
        "root": root,
        "checkpoint_dir": checkpoint_dir,
        "pickle_dir": pickle_dir,
        "manifest_dir": manifest_dir,
    }


def _embedding_history_source_path(context: TransformerSectionContext) -> Path | None:
    """Returns the trainer-generated embedding-history pickle path if available."""
    if not context.args.track_embedding_evolution:
        return None
    candidate = Path(context.args.embedding_evolution_dir) / context.args.embedding_evolution_filename
    return candidate if candidate.exists() else None


def _topk_prediction_source_path(context: TransformerSectionContext) -> Path | None:
    """Returns the trainer-generated top-k prediction pickle path if available."""
    if not context.args.track_top_k_predictions:
        return None
    candidate = Path(context.args.top_k_prediction_dir) / context.args.top_k_prediction_filename
    return candidate if candidate.exists() else None


def _build_topk_predictions_snapshot(context: TransformerSectionContext) -> dict[int, object]:
    """Builds a fallback top-k snapshot when trainer top-k history is unavailable."""
    prefix_strings = build_edge_prefix_strings_for_all_nodes(context.args)
    x_topk = build_batched_prefix_tensor(context.tokenizer, context.device, prefix_strings)
    topk_predictions = get_top_k_predictions_for_all_nodes(context.model, x_topk, k=5)
    return {-1: topk_predictions}


def _persist_notebook_run_artifacts(
    context: TransformerSectionContext,
    checkpoint_source_path: str,
) -> dict[str, str]:
    """Persists checkpoint and plotting pickles inside `tiny_graphs_notebooks/saved_artifacts`.

    Args:
        context: Prepared notebook section context.
        checkpoint_source_path: Existing checkpoint file path to archive.

    Returns:
        dict[str, str]: Resolved artifact paths (`checkpoint`, `embedding_history`,
            `topk_predictions`, `final_embeddings`, `manifest`).
    """
    dirs = _tiny_notebook_artifact_dirs(context)
    run_name = str(context.args.run_name)

    checkpoint_source = Path(checkpoint_source_path).resolve()
    checkpoint_target = dirs["checkpoint_dir"] / f"{run_name}_edge_memorization_final.pt"
    if checkpoint_source != checkpoint_target.resolve():
        shutil.copy2(checkpoint_source, checkpoint_target)
    else:
        checkpoint_target = checkpoint_source

    final_embeddings = get_node_embeddings(context.model, num_nodes=context.args.total_nodes)
    final_embeddings_target = dirs["pickle_dir"] / f"{run_name}_final_embeddings.pkl"
    write_pickle(final_embeddings_target, final_embeddings)

    embedding_history_target = dirs["pickle_dir"] / f"{run_name}_embedding_history.pkl"
    embedding_history_source = (
        Path(context.embedding_history_path)
        if context.embedding_history_path and Path(context.embedding_history_path).exists()
        else _embedding_history_source_path(context)
    )
    if embedding_history_source is not None:
        shutil.copy2(embedding_history_source, embedding_history_target)
    else:
        write_pickle(embedding_history_target, {-1: final_embeddings})

    topk_predictions_target = dirs["pickle_dir"] / f"{run_name}_topk_predictions.pkl"
    topk_source = (
        Path(context.topk_history_path)
        if context.topk_history_path and Path(context.topk_history_path).exists()
        else _topk_prediction_source_path(context)
    )
    if topk_source is not None:
        shutil.copy2(topk_source, topk_predictions_target)
    else:
        write_pickle(topk_predictions_target, _build_topk_predictions_snapshot(context))

    manifest_target = dirs["manifest_dir"] / f"{run_name}.json"
    manifest_payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_name": run_name,
        "graph_type": str(context.args.graph_type),
        "model_family": str(context.args.model_family),
        "add_self_edges": bool(getattr(context.args, "add_self_edges", False)),
        "checkpoint_path": str(checkpoint_target),
        "embedding_history_path": str(embedding_history_target),
        "topk_predictions_path": str(topk_predictions_target),
        "final_embeddings_path": str(final_embeddings_target),
    }
    save_json(manifest_target, manifest_payload)

    context.checkpoint_path = str(checkpoint_target)
    context.embedding_history_path = str(embedding_history_target)
    context.topk_history_path = str(topk_predictions_target)
    context.final_embeddings_path = str(final_embeddings_target)
    context.manifest_path = str(manifest_target)

    return {
        "checkpoint": str(checkpoint_target),
        "embedding_history": str(embedding_history_target),
        "topk_predictions": str(topk_predictions_target),
        "final_embeddings": str(final_embeddings_target),
        "manifest": str(manifest_target),
    }


def _manifest_matches_transformer_context(
    payload: Mapping[str, object],
    context: TransformerSectionContext,
) -> bool:
    """Checks whether a manifest belongs to the current transformer model family.

    Args:
        payload: Manifest JSON payload.
        context: Transformer section context.

    Returns:
        bool: True when the manifest is compatible with this context.
    """
    expected_family = str(context.args.model_family)
    expected_self_edges = bool(getattr(context.args, "add_self_edges", False))
    payload_family = str(payload.get("model_family", "")).strip()
    payload_self_edges = bool(payload.get("add_self_edges", False))
    if payload_family:
        return payload_family == expected_family and payload_self_edges == expected_self_edges

    # Backfill for older manifests that may miss `model_family`.
    run_name = str(payload.get("run_name", ""))
    family_matches = f"_{expected_family}-" in run_name or run_name.startswith(f"{expected_family}_")
    self_edge_token = f"-selfedge{int(expected_self_edges)}"
    if expected_self_edges:
        return family_matches and self_edge_token in run_name
    return family_matches and (self_edge_token in run_name or "-selfedge1" not in run_name)


def _find_latest_manifest_path(context: TransformerSectionContext) -> Path | None:
    """Finds the newest compatible manifest for the current graph/model scope."""
    manifest_dir = _tiny_notebook_artifact_dirs(context)["manifest_dir"]
    matched: list[Path] = []
    for manifest_path in manifest_dir.glob('*.json'):
        payload = load_json(manifest_path)
        if _manifest_matches_transformer_context(payload, context):
            matched.append(manifest_path)
    if not matched:
        return None
    matched.sort(key=lambda p: p.stat().st_mtime)
    return matched[-1]


def _find_manifest_by_checkpoint(
    context: TransformerSectionContext,
    checkpoint_path: str,
) -> Path | None:
    """Finds a compatible manifest that references a given checkpoint path."""
    manifest_dir = _tiny_notebook_artifact_dirs(context)["manifest_dir"]
    checkpoint_resolved = str(Path(checkpoint_path).resolve())
    for manifest_path in manifest_dir.glob('*.json'):
        payload = load_json(manifest_path)
        if not _manifest_matches_transformer_context(payload, context):
            continue
        payload_checkpoint = str(Path(str(payload.get('checkpoint_path', ''))).resolve())
        if payload_checkpoint == checkpoint_resolved:
            return manifest_path
    return None


def _iter_transformer_manifest_candidates(context: TransformerSectionContext) -> list[Path]:
    """Returns compatible transformer manifests sorted newest-first.

    Args:
        context: Transformer section context.

    Returns:
        list[Path]: Candidate manifest paths ordered by recency.
    """
    manifest_dir = _tiny_notebook_artifact_dirs(context)["manifest_dir"]
    matched: list[Path] = []
    for manifest_path in manifest_dir.glob('*.json'):
        payload = load_json(manifest_path)
        if _manifest_matches_transformer_context(payload, context):
            matched.append(manifest_path)
    matched.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return matched


def _load_checkpoint_state_dict(
    context: TransformerSectionContext,
    checkpoint_path: str,
) -> None:
    """Loads model weights from a checkpoint path into the current context model.

    Args:
        context: Transformer section context.
        checkpoint_path: Path to checkpoint file.

    Returns:
        None: Model is updated in-place.
    """
    context.model.load_state_dict(torch.load(checkpoint_path, map_location=context.device))
    context.model.to(context.device)


def _load_latest_compatible_transformer_checkpoint(
    context: TransformerSectionContext,
) -> tuple[str, dict[str, object]]:
    """Loads the newest manifest checkpoint that is state-dict compatible.

    Args:
        context: Transformer section context.

    Returns:
        tuple[str, dict[str, object]]: Loaded checkpoint path and manifest payload.

    Raises:
        RuntimeError: If no compatible checkpoint can be loaded.
    """
    candidates = _iter_transformer_manifest_candidates(context)
    if not candidates:
        raise RuntimeError(
            "No transformer manifests found for this graph scope/model family. "
            "Train once with train_from_scratch=True, or set checkpoint_path explicitly."
        )

    errors: list[str] = []
    for manifest_path in candidates:
        payload = load_json(manifest_path)
        checkpoint_candidate = str(Path(str(payload.get("checkpoint_path", ""))).resolve())
        if not checkpoint_candidate or not Path(checkpoint_candidate).exists():
            errors.append(f"{manifest_path.name}: missing checkpoint file")
            continue

        try:
            _load_checkpoint_state_dict(context, checkpoint_candidate)
        except RuntimeError as exc:
            errors.append(f"{manifest_path.name}: {exc}")
            continue

        context.checkpoint_path = checkpoint_candidate
        context.manifest_path = str(manifest_path)
        return checkpoint_candidate, payload

    summary = "\n".join(errors[:5])
    raise RuntimeError(
        "No compatible transformer checkpoint found in manifests for this graph scope. "
        "Consider retraining or pass checkpoint_path explicitly.\n"
        f"Sample load errors:\n{summary}"
    )


def _resolve_checkpoint_for_loading(
    context: TransformerSectionContext,
    checkpoint_path: str,
) -> tuple[str, dict[str, object] | None]:
    """Resolves a checkpoint path for load mode.

    If `checkpoint_path` is empty, uses the latest saved-run manifest for this
    graph scope.
    """
    if checkpoint_path:
        resolved_checkpoint = str(Path(checkpoint_path).resolve())
        if not Path(resolved_checkpoint).exists():
            raise FileNotFoundError(f"Checkpoint file not found: {resolved_checkpoint}")
        manifest_path = _find_manifest_by_checkpoint(context, resolved_checkpoint)
        return (
            resolved_checkpoint,
            load_json(manifest_path) if manifest_path is not None else None,
        )

    latest_manifest_path = _find_latest_manifest_path(context)
    if latest_manifest_path is None:
        raise ValueError(
            "No saved checkpoint manifest found for this graph scope. "
            "Train once with train_from_scratch=True, or set checkpoint_path explicitly."
        )

    manifest_payload = load_json(latest_manifest_path)
    resolved_checkpoint = str(Path(str(manifest_payload["checkpoint_path"])).resolve())
    if not Path(resolved_checkpoint).exists():
        raise FileNotFoundError(
            f"Saved manifest points to a missing checkpoint: {resolved_checkpoint}"
        )
    return resolved_checkpoint, manifest_payload


def train_or_load_edge_model(
    context: TransformerSectionContext,
    *,
    train_from_scratch: bool,
    checkpoint_path: str = "",
) -> str:
    """Trains edge memorization model or loads a saved checkpoint.

    Args:
        context: Prepared section context.
        train_from_scratch: Whether to train in this run.
        checkpoint_path: Optional path to load when `train_from_scratch` is False.
            If empty, the latest saved run for the same graph scope is used.

    Returns:
        str: Resolved archived checkpoint path inside `saved_artifacts/checkpoints`.
    """
    if train_from_scratch:
        context.model = run_edge_memorization_training(
            model=context.model,
            edge_train_loader=context.edge_train_loader,
            args=context.args,
            device=context.device,
            tokenizer=context.tokenizer,
            logging_enabled=False,
            experiment_logger=None,
        )
        resolved_checkpoint = str(
            Path(context.args.output_checkpoint_dir)
            / f"{context.args.run_name}_edge_memorization_final.pt"
        )
        torch.save(context.model.state_dict(), resolved_checkpoint)
    else:
        if checkpoint_path:
            resolved_checkpoint, manifest_payload = _resolve_checkpoint_for_loading(
                context=context,
                checkpoint_path=checkpoint_path,
            )
            _load_checkpoint_state_dict(context, resolved_checkpoint)
        else:
            resolved_checkpoint, manifest_payload = _load_latest_compatible_transformer_checkpoint(
                context=context,
            )

        if manifest_payload is not None:
            context.embedding_history_path = str(manifest_payload.get("embedding_history_path", ""))
            context.topk_history_path = str(manifest_payload.get("topk_predictions_path", ""))
            context.final_embeddings_path = str(manifest_payload.get("final_embeddings_path", ""))
            if not context.manifest_path:
                context.manifest_path = str(
                    _find_manifest_by_checkpoint(context, resolved_checkpoint) or ""
                )

    persisted_paths = _persist_notebook_run_artifacts(
        context=context,
        checkpoint_source_path=resolved_checkpoint,
    )
    return persisted_paths["checkpoint"]


def evaluate_edge_and_path(context: TransformerSectionContext) -> dict[str, float]:
    """Runs edge evaluation, top-k recovery, and path train/test evaluation.

    Args:
        context: Prepared section context with loaded/trained model.

    Returns:
        dict[str, float]: Aggregated evaluation metrics.
    """
    evaluate_edge_memorization(
        model=context.model,
        loader=context.edge_eval_loader,
        device=context.device,
        logging_enabled=False,
        experiment_logger=None,
        during_edge_memorization=False,
    )

    edge_list = parse_edge_list_from_pretrain(context.pretrain_path)
    prefix_strings = build_edge_prefix_strings_for_all_nodes(context.args)
    x_topk = build_batched_prefix_tensor(context.tokenizer, context.device, prefix_strings)
    topk_predictions = get_top_k_predictions_for_all_nodes(context.model, x_topk, k=5)
    topk_recovery_percent = compute_topk_recovery_percent(topk_predictions, edge_list)

    if is_cuda_device(context.device) and torch.cuda.is_available():
        precision_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        eval_context = torch.amp.autocast(device_type="cuda", dtype=precision_dtype)
    else:
        eval_context = nullcontext()

    path_train_results = evaluate(
        model=context.model,
        loader=context.path_train_loader,
        temperature=0.01,
        ctx=eval_context,
        top_k=1,
        results={},
        mode="train",
    )
    path_test_results = evaluate(
        model=context.model,
        loader=context.path_test_loader,
        temperature=0.01,
        ctx=eval_context,
        top_k=1,
        results={},
        mode="test",
    )

    return {
        "topk_edge_recovery_percent": float(topk_recovery_percent),
        "path_train_accuracy_percent": float(path_train_results.get("train/accuracy", 0.0)),
        "path_test_accuracy_percent": float(path_test_results.get("test/accuracy", 0.0)),
    }


def print_evaluation_report(section_name: str, metrics: Mapping[str, float]) -> None:
    """Prints a clean, section-labeled evaluation summary.

    Args:
        section_name: Human-readable section label (e.g., `"Tiny Path-Star"`).
        metrics: Evaluation output from `evaluate_edge_and_path`.

    Returns:
        None: Report is printed to stdout.
    """
    print("=" * 72)
    print(f"{section_name} | Evaluation Summary")
    print("-" * 72)
    print(f"Top-k edge recovery : {metrics['topk_edge_recovery_percent']:.2f}%")
    print(f"Path train accuracy : {metrics['path_train_accuracy_percent']:.2f}%")
    print(f"Path test accuracy  : {metrics['path_test_accuracy_percent']:.2f}%")
    print("=" * 72)


def parse_edge_list_from_pretrain(pretrain_path: str) -> list[tuple[int, int]]:
    """Parses edge list from `u=v` pretrain dataset file.

    Args:
        pretrain_path: Edge memorization dataset path.

    Returns:
        list[tuple[int, int]]: Directed `(u, v)` edge list.
    """
    edge_list: list[tuple[int, int]] = []
    with open(pretrain_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if "=" not in line:
                continue
            src, dst = line.split("=", maxsplit=1)
            edge_list.append((int(src), int(dst)))
    return edge_list


def infer_root_node_index(edge_list: Sequence[tuple[int, int]]) -> int | None:
    """Infers likely root node from max out-degree.

    Args:
        edge_list: Directed graph edges.

    Returns:
        int | None: Root-like node index or `None` if unavailable.
    """
    out_degree: dict[int, int] = {}
    for src, _ in edge_list:
        out_degree[src] = out_degree.get(src, 0) + 1
    if not out_degree:
        return None
    return max(out_degree.items(), key=lambda item: item[1])[0]


def collect_embeddings_and_edges(context: TransformerSectionContext):
    """Collects model node embeddings, edge list, and root-node guess.

    Args:
        context: Prepared section context.

    Returns:
        tuple[np.ndarray, list[tuple[int, int]], int | None]: Embeddings, edges, root.
    """
    node_embeddings = get_node_embeddings(context.model, num_nodes=context.args.total_nodes)
    edge_list = parse_edge_list_from_pretrain(context.pretrain_path)
    root_node = infer_root_node_index(edge_list) if context.args.graph_type == "star" else None
    return node_embeddings, edge_list, root_node


def load_analysis_histories(
    context: TransformerSectionContext,
    edge_list: Sequence[tuple[int, int]],
    *,
    embedding_history_path: str = "",
    topk_history_path: str = "",
):
    """Loads embedding and top-k history dictionaries for analysis plots.

    Args:
        context: Prepared section context.
        edge_list: Edge list for top-k recovery conversion.
        embedding_history_path: Optional explicit embedding history path.
        topk_history_path: Optional explicit top-k history path.

    Returns:
        tuple[dict[int, np.ndarray], dict[int, float]]: Embedding and recovery histories.
    """
    embedding_history: dict[int, np.ndarray] = {}
    topk_recovery_history: dict[int, float] = {}

    embedding_candidate = None
    if embedding_history_path:
        embedding_candidate = Path(embedding_history_path)
    elif context.embedding_history_path:
        embedding_candidate = Path(context.embedding_history_path)
    elif context.args.track_embedding_evolution:
        embedding_candidate = (
            Path(context.args.embedding_evolution_dir)
            / context.args.embedding_evolution_filename
        )
    if embedding_candidate is not None and embedding_candidate.exists():
        raw_history = read_pickle(embedding_candidate)
        embedding_history = {int(step): np.asarray(emb) for step, emb in raw_history.items()}

    topk_candidate = None
    if topk_history_path:
        topk_candidate = Path(topk_history_path)
    elif context.topk_history_path:
        topk_candidate = Path(context.topk_history_path)
    elif context.args.track_top_k_predictions:
        topk_candidate = (
            Path(context.args.top_k_prediction_dir)
            / context.args.top_k_prediction_filename
        )
    if topk_candidate is not None and topk_candidate.exists():
        raw_topk = read_pickle(topk_candidate)
        topk_recovery_history = {
            int(step): compute_topk_recovery_percent(topk_predictions=topk, edges=edge_list)
            for step, topk in raw_topk.items()
        }

    return embedding_history, topk_recovery_history
