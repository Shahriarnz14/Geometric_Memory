"""Shared naming and sizing helpers for in-weights datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple


def graph_folder_name(graph_type: str) -> str:
    """Graph folder name.
    
    Args:
        graph_type: Input parameter.
    
    Returns:
        object: Function return value.
    """
    return {
        "star": "star_graphs_randomized",
        "grid": "grid_graphs_randomized",
        "cycle": "cycle_graphs_randomized",
        "irregular": "irregular_graphs_randomized",
    }[graph_type]


def resolve_graph_dataset_directory(dataset_root: Path, graph_type: str) -> Path:
    """Resolve graph dataset directory.
    
    Args:
        dataset_root: Input parameter.
        graph_type: Input parameter.
    
    Returns:
        object: Function return value.
    """
    return dataset_root / graph_folder_name(graph_type)


def derive_total_nodes(args) -> int:
    """Derive total nodes.
    
    Args:
        args: Input parameter.
    
    Returns:
        object: Function return value.
    """
    if args.total_nodes != -1:
        return args.total_nodes

    if args.graph_type == "star":
        if args.star_subtree_degree == 1:
            return args.star_degree * (args.path_length - 1) + 1
        subtree_depth = args.path_length - 1
        nodes_per_subtree = (args.star_subtree_degree**subtree_depth - 1) // (
            args.star_subtree_degree - 1
        )
        return 1 + args.star_degree * nodes_per_subtree

    if args.graph_type == "grid":
        return args.grid_rows * args.grid_cols

    if args.graph_type == "cycle":
        raise ValueError("For cycle graphs, you must pass --total_nodes.")

    if args.graph_type == "irregular":
        return 16

    raise ValueError(f"Unsupported graph_type: {args.graph_type}")


def derive_expected_split_sizes(args) -> Tuple[int, int]:
    """Derive expected split sizes.
    
    Args:
        args: Input parameter.
    
    Returns:
        object: Function return value.
    """
    if args.graph_type == "star":
        if args.star_subtree_degree == 1:
            total_path_count = args.star_degree
        else:
            leaves_per_subtree = args.star_subtree_degree ** (args.path_length - 2)
            total_path_count = args.star_degree * leaves_per_subtree
            if args.split_subtree_holdout:
                train_subtree_count = int(args.train_split_ratio * args.star_degree)
                train_count = train_subtree_count * leaves_per_subtree
                return train_count, total_path_count - train_count

        train_count = int(total_path_count * args.train_split_ratio)
        return train_count, total_path_count - train_count

    if args.graph_type in {"grid", "cycle", "irregular"}:
        total_path_count = args.total_nodes
        train_count = int(total_path_count * args.train_split_ratio)
        return train_count, total_path_count - train_count

    raise ValueError(f"Unsupported graph_type: {args.graph_type}")


def build_graph_dataset_stem(args) -> str:
    """Build graph dataset stem.
    
    Args:
        args: Input parameter.
    
    Returns:
        object: Function return value.
    """
    if args.graph_type == "star":
        return (
            "star"
            f"_deg_{args.star_degree}"
            f"_deg_tree_{args.star_subtree_degree}"
            f"_path_{args.path_length}"
            f"_nodes_{args.total_nodes}"
        )
    if args.graph_type == "grid":
        return f"grid_rows_{args.grid_rows}_cols_{args.grid_cols}_nodes_{args.total_nodes}"
    if args.graph_type == "cycle":
        return f"cycle_N_{args.total_nodes}"
    if args.graph_type == "irregular":
        return f"irregular_graph_nodes_{args.total_nodes}_edges_{args.irregular_edge_count}"
    raise ValueError(f"Unsupported graph_type: {args.graph_type}")


def build_dataset_split_filename(args, split_suffix: str) -> str:
    """Build dataset split filename.
    
    Args:
        args: Input parameter.
        split_suffix: Input parameter.
    
    Returns:
        object: Function return value.
    """
    stem = build_graph_dataset_stem(args)
    sd_token = (
        f"{int(args.include_start_node_in_path_finetuning)}"
        f"{int(args.use_directional_edge_pretraining)}"
    )
    fb_token = f"{int(args.add_forward_edges)}{int(args.add_backward_edges)}"
    selfedge_token = f"{int(getattr(args, 'add_self_edges', False))}"
    return f"{stem}_sd_{sd_token}_fb_{fb_token}_selfedge_{selfedge_token}_{split_suffix}.txt"
