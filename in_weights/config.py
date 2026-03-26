"""Configuration and argument parsing for in-weights training."""

from __future__ import annotations

import argparse
from pathlib import Path

from geometric_memory.data.dataset_naming import (
    derive_expected_split_sizes,
    derive_total_nodes,
    resolve_graph_dataset_directory,
)
from geometric_memory.in_weights.experiment_modes import (
    get_training_recipe_help_text,
)


def _derive_model_architecture_label(args):
    """ derive model architecture label.
    
    Args:
        args: Input parameter.
    
    Returns:
        object: Function return value.
    """
    if args.model_family.startswith("gpt"):
        label = "transformer"
        if args.freeze_token_embeddings:
            label = "associative"
        elif not args.use_attention:
            label = "feedforward_residual" if args.use_residual_connections else "feedforward_plain"
        if not args.tie_input_output_embeddings:
            label += "_untied"
        return label

    if args.model_family == "mamba":
        return "mamba"
    if args.model_family.startswith("pythia"):
        return "pythia"
    return args.model_family


def get_args(args_list=None):
    """Defines and parses command-line arguments for training.

    Args:
        args_list: Input parameter.

    Returns:
        object: Function return value.
    """
    parser = argparse.ArgumentParser(
        description="In-weights geometric-memory training configuration"
    )

    # Recipe selection.
    parser.add_argument(
        "--training_recipe",
        type=str,
        default="mixed_full_path",
        choices=[
            "staged_full_path",
            "mixed_full_path",
            "staged_hardest_token",
            "mixed_hardest_token",
        ],
        help=f"Training recipe: {get_training_recipe_help_text()}",
    )

    # Model architecture.
    parser.add_argument(
        "--model_family",
        type=str,
        default="gpt",
        choices=["gpt", "gpt2", "pythia", "mamba"],
        help="Model family.",
    )
    parser.add_argument(
        "--transformer_layer_count",
        type=int,
        default=12,
        help="Number of transformer (or mamba) layers.",
    )
    parser.add_argument(
        "--embedding_dimension",
        type=int,
        default=384,
        help="Hidden embedding dimension.",
    )
    parser.add_argument(
        "--attention_head_count",
        type=int,
        default=8,
        help="Number of attention heads (transformer only).",
    )
    parser.add_argument(
        "--dropout_rate",
        type=float,
        default=0.0,
        help="Dropout probability for trainable transformer blocks.",
    )

    parser.add_argument(
        "--use_attention",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable self-attention (transformer only).",
    )
    parser.add_argument(
        "--use_residual_connections",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable residual connections.",
    )
    parser.add_argument(
        "--use_layer_norm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable layer normalization.",
    )
    parser.add_argument(
        "--use_positional_encoding",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable positional encoding.",
    )
    parser.add_argument(
        "--tie_input_output_embeddings",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Tie token embedding and output projection weights.",
    )
    parser.add_argument(
        "--freeze_token_embeddings",
        action="store_true",
        help="Freeze token embeddings.",
    )
    parser.add_argument(
        "--use_mlp_only_blocks",
        action="store_true",
        help="Use single MLP blocks instead of transformer-style MLP.",
    )
    parser.add_argument(
        "--track_embedding_evolution",
        action="store_true",
        help="Record node embedding evolution during edge memorization.",
    )
    parser.add_argument(
        "--track_top_k_predictions",
        action="store_true",
        help="Record top-k predicted nodes during edge memorization.",
    )

    # Mamba-specific architecture.
    parser.add_argument(
        "--mamba_state_dimension",
        type=int,
        default=16,
        help="Mamba state dimension.",
    )
    parser.add_argument(
        "--mamba_convolution_kernel",
        type=int,
        default=4,
        help="Mamba convolution kernel size.",
    )
    parser.add_argument(
        "--mamba_expand_factor",
        type=int,
        default=2,
        help="Mamba expansion factor.",
    )

    # Graph shape.
    parser.add_argument(
        "--graph_type",
        type=str,
        default="star",
        choices=["star", "grid", "cycle", "irregular"],
        help="Graph family.",
    )
    parser.add_argument(
        "--star_degree",
        type=int,
        default=10000,
        help="Root degree for star/path-star graphs.",
    )
    parser.add_argument(
        "--star_subtree_degree",
        type=int,
        default=1,
        help="Subtree branching factor for star-tree graphs.",
    )
    parser.add_argument(
        "--path_length",
        type=int,
        default=6,
        help="Target path length (number of nodes in a path).",
    )
    parser.add_argument(
        "--total_nodes",
        type=int,
        default=-1,
        help="Total number of nodes. Auto-derived for star/grid/irregular if -1.",
    )
    parser.add_argument(
        "--grid_rows",
        type=int,
        default=50,
        help="Grid rows (when graph_type=grid).",
    )
    parser.add_argument(
        "--grid_cols",
        type=int,
        default=50,
        help="Grid cols (when graph_type=grid).",
    )
    parser.add_argument(
        "--irregular_edge_count",
        type=int,
        default=20,
        help="Undirected edge count for the fixed irregular graph (must be 20).",
    )

    # Edge directionality.
    parser.add_argument(
        "--add_forward_edges",
        action="store_true",
        help="Include forward edges.",
    )
    parser.add_argument(
        "--add_backward_edges",
        action="store_true",
        help="Include backward edges.",
    )
    parser.add_argument(
        "--use_directional_edge_pretraining",
        action="store_true",
        help="Directional edge-pretraining setting (sd token second digit).",
    )
    parser.add_argument(
        "--include_start_node_in_path_finetuning",
        action="store_true",
        help="Include start node in path-finetuning prefix (sd first digit).",
    )

    # Dataset location and split.
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="data/datasets/in_weights_graphs",
        help="Root folder containing generated in-weights datasets.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="in_weights",
        help="Dataset identifier used for run metadata.",
    )
    parser.add_argument(
        "--train_split_ratio",
        type=float,
        default=0.75,
        help="Train split ratio for expected path counts.",
    )
    parser.add_argument(
        "--split_subtree_holdout",
        action="store_true",
        help=("For star_subtree_degree>1, split train/test by subtree instead of" " by leaves."),
    )

    # Training controls.
    parser.add_argument(
        "--edge_memorization_batch_size",
        type=int,
        default=16384,
        help="Batch size for edge-memorization training.",
    )
    parser.add_argument(
        "--path_finetuning_batch_size",
        type=int,
        default=2048,
        help="Batch size for path-finetuning training.",
    )
    parser.add_argument(
        "--edge_memorization_learning_rate",
        type=float,
        default=1e-2,
        help="Learning rate for edge memorization.",
    )
    parser.add_argument(
        "--path_finetuning_learning_rate",
        type=float,
        default=5e-4,
        help="Learning rate for path finetuning.",
    )
    parser.add_argument(
        "--optimizer_weight_decay",
        type=float,
        default=1e-2,
        help="AdamW weight decay.",
    )
    parser.add_argument(
        "--edge_memorization_warmup_steps",
        type=int,
        default=500,
        help="Warmup steps for edge memorization.",
    )
    parser.add_argument(
        "--path_finetuning_warmup_steps",
        type=int,
        default=1000,
        help="Warmup steps for path finetuning.",
    )
    parser.add_argument(
        "--disable_edge_memorization_lr_decay",
        action="store_true",
        help="Disable LR decay in edge memorization.",
    )
    parser.add_argument(
        "--disable_path_finetuning_lr_decay",
        action="store_true",
        help="Disable LR decay in path finetuning.",
    )
    parser.add_argument(
        "--edge_memorization_epochs",
        type=int,
        default=2500,
        help="Epochs for edge memorization.",
    )
    parser.add_argument(
        "--path_finetuning_epochs",
        type=int,
        default=10000,
        help="Epochs for path finetuning.",
    )
    parser.add_argument(
        "--enable_edge_memorization_early_stopping",
        action="store_true",
        help="Enable early stopping in edge memorization.",
    )
    parser.add_argument(
        "--skip_edge_memorization",
        action="store_true",
        help="Skip edge memorization and load an existing checkpoint.",
    )
    parser.add_argument(
        "--edge_memorization_include_pause_token",
        action="store_true",
        help=(
            "Include [PAUSE] in edge-memorization prefixes. "
            "By default, pause is dropped."
        ),
    )
    parser.add_argument(
        "--path_prefix_pause_token_count",
        type=int,
        default=1,
        help="Number of pause tokens in path-finetuning prefix.",
    )
    parser.add_argument(
        "--exclude_task_token_in_prefix",
        action="store_true",
        help='Disable task tag in prefixes (e.g. "[EDGE]", "[PATH]").',
    )
    parser.add_argument(
        "--edge_memorization_checkpoint_path",
        type=str,
        default="",
        help="Checkpoint path used when --skip_edge_memorization is enabled.",
    )
    parser.add_argument(
        "--checkpoint_interval_epochs",
        type=int,
        default=500,
        help="Save checkpoint every N epochs.",
    )
    parser.add_argument(
        "--edge_memorization_eval_interval_epochs",
        type=int,
        default=50,
        help="Edge-memorization eval interval (epochs).",
    )
    parser.add_argument(
        "--path_finetuning_eval_interval_epochs",
        type=int,
        default=1,
        help="Path-finetuning eval interval (epochs).",
    )
    parser.add_argument(
        "--prediction_logging_multiplier",
        type=int,
        default=20,
        help=(
            "Log prediction samples every (eval_interval * multiplier) epochs "
            "during path finetuning."
        ),
    )
    parser.add_argument(
        "--use_teacherless_inputs",
        action="store_true",
        help="Replace target tokens in input with teacherless token.",
    )
    parser.add_argument(
        "--reverse_path_targets",
        action="store_true",
        help="Reverse target path order for path finetuning.",
    )

    # Logging.
    parser.add_argument(
        "--experiment_log_root",
        type=str,
        default="./experiment_logs",
        help="Root directory for run artifacts and logs.",
    )
    parser.add_argument(
        "--enable_wandb",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable Weights & Biases logging.",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="geometric-memory",
        help="W&B project.",
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default="",
        help="W&B entity/team.",
    )
    parser.add_argument(
        "--wandb_group",
        type=str,
        default="",
        help="W&B group name.",
    )
    parser.add_argument(
        "--wandb_tags",
        type=str,
        default="",
        help="Comma-separated W&B tags.",
    )
    parser.add_argument(
        "--wandb_mode",
        type=str,
        default="offline",
        choices=["online", "offline", "disabled"],
        help="W&B mode.",
    )

    args = parser.parse_args(args_list)
    raw_total_nodes = args.total_nodes
    args.include_task_token_in_prefix = not args.exclude_task_token_in_prefix
    args.edge_memorization_drop_pause_token = (
        not args.edge_memorization_include_pause_token
    )

    args.total_nodes = derive_total_nodes(args)
    if args.graph_type == "irregular" and raw_total_nodes == -1:
        print("Irregular graph node count not provided. Defaulting to total_nodes=16.")

    if args.graph_type == "irregular" and args.irregular_edge_count != 20:
        raise ValueError(
            "Irregular graph is fixed and requires --irregular_edge_count 20. "
            f"Received {args.irregular_edge_count}."
        )

    args.dataset_root = Path(args.dataset_root)
    args.dataset_directory = resolve_graph_dataset_directory(
        dataset_root=args.dataset_root,
        graph_type=args.graph_type,
    )

    if not args.add_forward_edges and not args.add_backward_edges:
        args.add_forward_edges = True
        args.add_backward_edges = True
        print(
            "Warning: neither --add_forward_edges nor --add_backward_edges was set. "
            "Defaulting to both."
        )

    if args.skip_edge_memorization and not args.edge_memorization_checkpoint_path:
        raise ValueError("--skip_edge_memorization requires --edge_memorization_checkpoint_path.")

    args.expected_train_path_count, args.expected_test_path_count = (
        derive_expected_split_sizes(args)
    )

    args.model_architecture_label = _derive_model_architecture_label(args)

    # Optional analysis artifact output setup.
    if args.track_embedding_evolution:
        args.embedding_evolution_dir = Path(args.experiment_log_root) / "embedding_evolution"
        args.embedding_evolution_dir.mkdir(parents=True, exist_ok=True)
        args.embedding_evolution_filename = (
            f"{args.graph_type}_{args.model_architecture_label}"
            f"_embedding_evolution_lr_{args.edge_memorization_learning_rate}.pkl"
        )

    if args.track_top_k_predictions:
        args.top_k_prediction_dir = Path(args.experiment_log_root) / "top_k_predictions"
        args.top_k_prediction_dir.mkdir(parents=True, exist_ok=True)
        args.top_k_prediction_filename = (
            f"{args.graph_type}_{args.model_architecture_label}"
            f"_top_k_predictions_lr_{args.edge_memorization_learning_rate}.pkl"
        )

    return args
