"""Run naming helpers for in-weights experiment runs."""

from datetime import datetime

from geometric_memory.utils.experiment_logging import (
    RunDirectories,
    prepare_run_directories,
)


def _format_float(value: float) -> str:
    """Formats a float for compact run-name tokens.

    Args:
        value: Floating-point value to serialize.

    Returns:
        str: Compact string form with `.` replaced by `p`.
    """
    return f"{value:g}".replace(".", "p")


def _graph_token(args) -> str:
    """Builds the graph-shape token used in run names.

    Args:
        args: Parsed CLI arguments namespace.

    Returns:
        str: Encoded graph token for run naming.
    """
    if args.graph_type == "star":
        return (
            f"star-d{args.star_degree}"
            f"-dt{args.star_subtree_degree}"
            f"-p{args.path_length}"
            f"-n{args.total_nodes}"
        )
    if args.graph_type == "grid":
        return f"grid-r{args.grid_rows}-c{args.grid_cols}-n{args.total_nodes}"
    if args.graph_type == "cycle":
        return f"cycle-n{args.total_nodes}"
    if args.graph_type == "irregular":
        return f"irregular-n{args.total_nodes}-e{args.irregular_edge_count}"
    return f"{args.graph_type}-n{args.total_nodes}"


def build_run_name(args, recipe_name: str) -> str:
    """Builds a concise run name with graph/model/training metadata.

    Args:
        args: Parsed CLI arguments namespace.
        recipe_name: Training recipe identifier.

    Returns:
        str: Fully formatted run name.
    """
    now = datetime.now().strftime("%Y%m%d-%H%M%S")
    sd_token = (
        f"{int(args.include_start_node_in_path_finetuning)}"
        f"{int(args.use_directional_edge_pretraining)}"
    )
    fb_token = f"{int(args.add_forward_edges)}{int(args.add_backward_edges)}"

    model_token = (
        f"{args.model_family}-L{args.transformer_layer_count}"
        f"-D{args.embedding_dimension}-H{args.attention_head_count}"
        if not args.model_family.startswith("mamba")
        else (
            f"{args.model_family}-L{args.transformer_layer_count}"
            f"-D{args.embedding_dimension}"
            f"-S{args.mamba_state_dimension}"
            f"-C{args.mamba_convolution_kernel}"
            f"-E{args.mamba_expand_factor}"
        )
    )
    training_token = (
        f"bs{args.edge_memorization_batch_size}x{args.path_finetuning_batch_size}"
        f"-lr{_format_float(args.edge_memorization_learning_rate)}"
        f"x{_format_float(args.path_finetuning_learning_rate)}"
    )
    behavior_token = (
        f"tl{int(args.use_teacherless_inputs)}"
        f"-rev{int(args.reverse_path_targets)}"
        f"-sd{sd_token}"
        f"-fb{fb_token}"
        f"-task{int(args.include_task_token_in_prefix)}"
        f"-split{int(args.split_subtree_holdout)}"
    )

    return "_".join(
        [
            args.dataset_name,
            _graph_token(args),
            model_token,
            recipe_name,
            training_token,
            behavior_token,
            now,
        ]
    )


__all__ = ["RunDirectories", "build_run_name", "prepare_run_directories"]
