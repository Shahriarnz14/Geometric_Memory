"""CLI utility for generating in-weights datasets.

This script creates three files for a given graph setup:
1) `*_pretrain.txt` (edge memorization pairs: `u=v`)
2) `*_train_*.txt`   (path finetuning pairs: `leaf=path`)
3) `*_test_*.txt`    (path finetuning pairs: `leaf=path`)
"""

from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

# Allow running as `python data/build_in_weights_datasets.py ...` from repo root.
if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from geometric_memory.data.dataset_naming import (
    build_dataset_split_filename,
    derive_total_nodes,
    graph_folder_name,
)


GraphEdge = Tuple[int, int]
NodePath = List[int]


FIXED_IRREGULAR_NODE_COUNT = 16
FIXED_IRREGULAR_EDGE_COUNT = 20
FIXED_IRREGULAR_UNDIRECTED_EDGES: tuple[tuple[int, int], ...] = (
    (12, 15),
    (5, 8),
    (5, 6),
    (8, 9),
    (7, 8),
    (1, 3),
    (13, 14),
    (13, 15),
    (12, 14),
    (12, 13),
    (1, 2),
    (4, 5),
    (3, 4),
    (10, 11),
    (0, 1),
    (2, 3),
    (6, 7),
    (11, 0),
    (9, 10),
    (14, 15),
)


@dataclass(frozen=True)
class DatasetBuildConfig:
    """DatasetBuildConfig definition.
    
    Description:
        Encapsulates related behavior and state.
    """
    graph_type: str
    path_length: int
    total_nodes: int
    train_split_ratio: float
    split_subtree_holdout: bool
    add_forward_edges: bool
    add_backward_edges: bool
    include_start_node_in_path_finetuning: bool
    use_directional_edge_pretraining: bool
    random_seed: int
    dataset_root: Path
    overwrite: bool
    randomize_node_ids: bool
    star_degree: int
    star_subtree_degree: int
    grid_rows: int
    grid_cols: int
    irregular_edge_count: int


def _ordered_unique(items: Iterable[GraphEdge]) -> List[GraphEdge]:
    """ ordered unique.
    
    Args:
        items: Input parameter.
    
    Returns:
        object: Function return value.
    """
    return list(dict.fromkeys(items))


def _resolve_output_paths(
    config: DatasetBuildConfig, train_size: int, test_size: int
) -> Dict[str, Path]:
    """ resolve output paths.
    
    Args:
        config: Input parameter.
        train_size: Input parameter.
        test_size: Input parameter.
    
    Returns:
        object: Function return value.
    """
    dataset_dir = config.dataset_root / graph_folder_name(config.graph_type)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    return {
        "pretrain": dataset_dir / build_dataset_split_filename(config, "pretrain"),
        "train": dataset_dir / build_dataset_split_filename(config, f"train_{train_size}"),
        "test": dataset_dir / build_dataset_split_filename(config, f"test_{test_size}"),
    }


def _write_edge_memorization_file(path: Path, directed_edges: Sequence[GraphEdge]):
    """ write edge memorization file.
    
    Args:
        path: Input parameter.
        directed_edges: Input parameter.
    
    Returns:
        object: Function return value.
    """
    with path.open("w", encoding="utf-8") as f:
        for src, dst in directed_edges:
            f.write(f"{src}={dst}\n")


def _format_path_prefix(node_path: Sequence[int], include_start_node_in_prefix: bool) -> str:
    """ format path prefix.
    
    Args:
        node_path: Input parameter.
        include_start_node_in_prefix: Input parameter.
    
    Returns:
        object: Function return value.
    """
    if include_start_node_in_prefix:
        return f"{node_path[0]},{node_path[-1]}"
    return str(node_path[-1])


def _write_path_finetuning_file(
    path: Path,
    node_paths: Sequence[NodePath],
    include_start_node_in_prefix: bool,
):
    """ write path finetuning file.
    
    Args:
        path: Input parameter.
        node_paths: Input parameter.
        include_start_node_in_prefix: Input parameter.
    
    Returns:
        object: Function return value.
    """
    with path.open("w", encoding="utf-8") as f:
        for node_path in node_paths:
            prefix = _format_path_prefix(
                node_path=node_path,
                include_start_node_in_prefix=include_start_node_in_prefix,
            )
            target = ",".join(str(node_id) for node_id in node_path)
            f.write(f"{prefix}={target}\n")


def _split_paths(
    all_paths: Sequence[NodePath], train_ratio: float, rng: random.Random
) -> Tuple[List[NodePath], List[NodePath]]:
    """ split paths.
    
    Args:
        all_paths: Input parameter.
        train_ratio: Input parameter.
        rng: Input parameter.
    
    Returns:
        object: Function return value.
    """
    shuffled_paths = list(all_paths)
    rng.shuffle(shuffled_paths)
    train_size = int(len(shuffled_paths) * train_ratio)
    train_paths = shuffled_paths[:train_size]
    test_paths = shuffled_paths[train_size:]
    return train_paths, test_paths


def _split_star_paths_by_subtree(
    all_paths: Sequence[NodePath], train_ratio: float, rng: random.Random
) -> Tuple[List[NodePath], List[NodePath]]:
    """Splits star-tree paths by subtree root (path[1]) instead of by leaves.

    Args:
        all_paths: Input parameter.
        train_ratio: Input parameter.
        rng: Input parameter.

    Returns:
        object: Function return value.
    """
    subtree_roots = sorted({path[1] for path in all_paths if len(path) >= 2})
    shuffled_roots = list(subtree_roots)
    rng.shuffle(shuffled_roots)

    train_subtree_count = int(len(shuffled_roots) * train_ratio)
    train_roots = set(shuffled_roots[:train_subtree_count])

    train_paths: List[NodePath] = []
    test_paths: List[NodePath] = []
    for path in all_paths:
        if len(path) < 2:
            test_paths.append(path)
            continue
        if path[1] in train_roots:
            train_paths.append(path)
        else:
            test_paths.append(path)
    return train_paths, test_paths


def _build_star_graph_data(
    config: DatasetBuildConfig, rng: random.Random
) -> Tuple[List[GraphEdge], List[NodePath]]:
    """ build star graph data.
    
    Args:
        config: Input parameter.
        rng: Input parameter.
    
    Returns:
        object: Function return value.
    """
    if config.path_length < 2:
        raise ValueError("`path_length` must be >= 2 for star graphs.")
    if config.star_subtree_degree < 1:
        raise ValueError("`star_subtree_degree` must be >= 1.")

    minimum_required_nodes = 1 + config.star_degree * (config.path_length - 1)
    if config.star_subtree_degree > 1:
        depth = config.path_length - 1
        nodes_per_subtree = (config.star_subtree_degree**depth - 1) // (
            config.star_subtree_degree - 1
        )
        minimum_required_nodes = 1 + config.star_degree * nodes_per_subtree

    if config.total_nodes < minimum_required_nodes:
        raise ValueError(
            f"Not enough nodes for star graph. minimum_required_nodes="
            f"{minimum_required_nodes}, total_nodes={config.total_nodes}"
        )

    candidate_node_ids = list(range(config.total_nodes))
    if config.randomize_node_ids:
        rng.shuffle(candidate_node_ids)
    node_cursor = 0

    def next_node() -> int:
        nonlocal node_cursor
        node_id = candidate_node_ids[node_cursor]
        node_cursor += 1
        return node_id

    root_node = next_node()
    forward_edges: List[GraphEdge] = []
    all_paths: List[NodePath] = []

    if config.star_subtree_degree == 1:
        for _ in range(config.star_degree):
            current_path = [root_node]
            for _ in range(config.path_length - 1):
                current_path.append(next_node())
            forward_edges.extend((src, dst) for src, dst in zip(current_path, current_path[1:]))
            all_paths.append(current_path)
        return _ordered_unique(forward_edges), all_paths

    for _ in range(config.star_degree):
        first_child = next_node()
        forward_edges.append((root_node, first_child))
        current_paths = [[root_node, first_child]]

        for _depth in range(2, config.path_length):
            expanded_paths: List[NodePath] = []
            for partial_path in current_paths:
                parent_node = partial_path[-1]
                for _ in range(config.star_subtree_degree):
                    child_node = next_node()
                    forward_edges.append((parent_node, child_node))
                    expanded_paths.append(partial_path + [child_node])
            current_paths = expanded_paths
        all_paths.extend(current_paths)

    return _ordered_unique(forward_edges), all_paths


def _build_cycle_graph_data(
    config: DatasetBuildConfig,
) -> Tuple[List[GraphEdge], List[NodePath]]:
    """ build cycle graph data.
    
    Args:
        config: Input parameter.
    
    Returns:
        object: Function return value.
    """
    if config.path_length < 2:
        raise ValueError("`path_length` must be >= 2 for cycle graphs.")
    if config.total_nodes < 3:
        raise ValueError("`total_nodes` must be >= 3 for cycle graphs.")

    forward_edges = [
        (node_id, (node_id + 1) % config.total_nodes) for node_id in range(config.total_nodes)
    ]
    all_paths = [
        [((start + offset) % config.total_nodes) for offset in range(config.path_length)]
        for start in range(config.total_nodes)
    ]
    return forward_edges, all_paths


def _grid_index(row: int, col: int, cols: int) -> int:
    """ grid index.
    
    Args:
        row: Input parameter.
        col: Input parameter.
        cols: Input parameter.
    
    Returns:
        object: Function return value.
    """
    return row * cols + col


def _build_grid_adjacency(rows: int, cols: int) -> Dict[int, List[int]]:
    """ build grid adjacency.
    
    Args:
        rows: Input parameter.
        cols: Input parameter.
    
    Returns:
        object: Function return value.
    """
    adjacency: Dict[int, List[int]] = {}
    for row in range(rows):
        for col in range(cols):
            node = _grid_index(row, col, cols)
            neighbors = []
            if row > 0:
                neighbors.append(_grid_index(row - 1, col, cols))
            if row < rows - 1:
                neighbors.append(_grid_index(row + 1, col, cols))
            if col > 0:
                neighbors.append(_grid_index(row, col - 1, cols))
            if col < cols - 1:
                neighbors.append(_grid_index(row, col + 1, cols))
            adjacency[node] = neighbors
    return adjacency


def _random_walk_path(
    start_node: int,
    path_length: int,
    adjacency: Dict[int, List[int]],
    rng: random.Random,
) -> NodePath:
    """ random walk path.
    
    Args:
        start_node: Input parameter.
        path_length: Input parameter.
        adjacency: Input parameter.
        rng: Input parameter.
    
    Returns:
        object: Function return value.
    """
    path = [start_node]
    prev_node = None
    current_node = start_node
    while len(path) < path_length:
        candidates = adjacency[current_node]
        if prev_node is not None and len(candidates) > 1:
            non_backtracking = [n for n in candidates if n != prev_node]
            if non_backtracking:
                candidates = non_backtracking
        next_node = rng.choice(candidates)
        path.append(next_node)
        prev_node, current_node = current_node, next_node
    return path


def _build_grid_graph_data(
    config: DatasetBuildConfig, rng: random.Random
) -> Tuple[List[GraphEdge], List[NodePath]]:
    """ build grid graph data.
    
    Args:
        config: Input parameter.
        rng: Input parameter.
    
    Returns:
        object: Function return value.
    """
    if config.path_length < 2:
        raise ValueError("`path_length` must be >= 2 for grid graphs.")
    if config.grid_rows * config.grid_cols != config.total_nodes:
        raise ValueError("For grid graphs, `total_nodes` must equal grid_rows * grid_cols.")

    forward_edges: List[GraphEdge] = []
    for row in range(config.grid_rows):
        for col in range(config.grid_cols):
            current = _grid_index(row, col, config.grid_cols)
            if col < config.grid_cols - 1:
                forward_edges.append((current, _grid_index(row, col + 1, config.grid_cols)))
            if row < config.grid_rows - 1:
                forward_edges.append((current, _grid_index(row + 1, col, config.grid_cols)))

    path_candidates: Set[Tuple[int, ...]] = set()

    # Straight horizontal paths.
    for row in range(config.grid_rows):
        for col_start in range(0, config.grid_cols - config.path_length + 1):
            right_path = tuple(
                _grid_index(row, col_start + offset, config.grid_cols)
                for offset in range(config.path_length)
            )
            path_candidates.add(right_path)
            path_candidates.add(tuple(reversed(right_path)))

    # Straight vertical paths.
    for col in range(config.grid_cols):
        for row_start in range(0, config.grid_rows - config.path_length + 1):
            down_path = tuple(
                _grid_index(row_start + offset, col, config.grid_cols)
                for offset in range(config.path_length)
            )
            path_candidates.add(down_path)
            path_candidates.add(tuple(reversed(down_path)))

    adjacency = _build_grid_adjacency(config.grid_rows, config.grid_cols)
    max_attempts = 50000
    attempts = 0
    while len(path_candidates) < config.total_nodes and attempts < max_attempts:
        start_node = rng.randrange(config.total_nodes)
        candidate = tuple(
            _random_walk_path(
                start_node=start_node,
                path_length=config.path_length,
                adjacency=adjacency,
                rng=rng,
            )
        )
        path_candidates.add(candidate)
        attempts += 1

    if len(path_candidates) < config.total_nodes:
        raise RuntimeError(
            "Could not generate enough fixed-length grid paths. "
            f"generated={len(path_candidates)} required={config.total_nodes}"
        )

    all_paths = [list(path) for path in sorted(path_candidates)]
    rng.shuffle(all_paths)
    all_paths = all_paths[: config.total_nodes]
    return _ordered_unique(forward_edges), all_paths


def _build_irregular_undirected_edges(
    total_nodes: int, total_edges: int, rng: random.Random
) -> Set[Tuple[int, int]]:
    """Returns the fixed irregular-graph undirected edge set.

    Args:
        total_nodes: Expected node count (must equal 16).
        total_edges: Expected undirected edge count (must equal 20).
        rng: Unused for this fixed graph definition.

    Returns:
        Set[Tuple[int, int]]: Fixed undirected edge set.
    """
    _ = rng
    if total_nodes != FIXED_IRREGULAR_NODE_COUNT:
        raise ValueError(
            "Irregular graph uses a fixed node count of "
            f"{FIXED_IRREGULAR_NODE_COUNT}, got total_nodes={total_nodes}."
        )
    if total_edges != FIXED_IRREGULAR_EDGE_COUNT:
        raise ValueError(
            "Irregular graph uses a fixed undirected edge count of "
            f"{FIXED_IRREGULAR_EDGE_COUNT}, got irregular_edge_count={total_edges}."
        )

    return {
        (min(u, v), max(u, v))
        for (u, v) in FIXED_IRREGULAR_UNDIRECTED_EDGES
    }


def _build_irregular_graph_data(
    config: DatasetBuildConfig, rng: random.Random
) -> Tuple[List[GraphEdge], List[NodePath]]:
    """Builds the fixed irregular-graph edge/path data.

    Args:
        config: Dataset build configuration.
        rng: Random generator for path sampling.

    Returns:
        Tuple[List[GraphEdge], List[NodePath]]: Forward edges and node paths.
    """
    if config.path_length < 2:
        raise ValueError("`path_length` must be >= 2 for irregular graphs.")
    if config.total_nodes != FIXED_IRREGULAR_NODE_COUNT:
        raise ValueError(
            "Irregular graph uses a fixed node count of "
            f"{FIXED_IRREGULAR_NODE_COUNT}, got total_nodes={config.total_nodes}."
        )
    if config.irregular_edge_count != FIXED_IRREGULAR_EDGE_COUNT:
        raise ValueError(
            "Irregular graph uses a fixed undirected edge count of "
            f"{FIXED_IRREGULAR_EDGE_COUNT}, got irregular_edge_count={config.irregular_edge_count}."
        )

    undirected_edges = _build_irregular_undirected_edges(
        total_nodes=config.total_nodes,
        total_edges=config.irregular_edge_count,
        rng=rng,
    )
    forward_edges = sorted(undirected_edges)

    adjacency: Dict[int, List[int]] = {node: [] for node in range(config.total_nodes)}
    for u, v in undirected_edges:
        adjacency[u].append(v)
        adjacency[v].append(u)

    all_paths: List[NodePath] = []
    for start_node in range(config.total_nodes):
        all_paths.append(
            _random_walk_path(
                start_node=start_node,
                path_length=config.path_length,
                adjacency=adjacency,
                rng=rng,
            )
        )

    return forward_edges, all_paths


def _build_graph_data(
    config: DatasetBuildConfig, rng: random.Random
) -> Tuple[List[GraphEdge], List[NodePath]]:
    """ build graph data.
    
    Args:
        config: Input parameter.
        rng: Input parameter.
    
    Returns:
        object: Function return value.
    """
    if config.graph_type == "star":
        return _build_star_graph_data(config, rng)
    if config.graph_type == "cycle":
        return _build_cycle_graph_data(config)
    if config.graph_type == "grid":
        return _build_grid_graph_data(config, rng)
    if config.graph_type == "irregular":
        return _build_irregular_graph_data(config, rng)
    raise ValueError(f"Unsupported graph_type: {config.graph_type}")


def _directed_edges_for_pretraining(
    forward_edges: Sequence[GraphEdge],
    include_forward: bool,
    include_backward: bool,
) -> List[GraphEdge]:
    """ directed edges for pretraining.
    
    Args:
        forward_edges: Input parameter.
        include_forward: Input parameter.
        include_backward: Input parameter.
    
    Returns:
        object: Function return value.
    """
    directed_edges: List[GraphEdge] = []
    sorted_forward = sorted(forward_edges)
    if include_forward:
        directed_edges.extend(sorted_forward)
    if include_backward:
        directed_edges.extend((dst, src) for src, dst in sorted_forward)
    return _ordered_unique(directed_edges)


def _build_config_from_cli(args: argparse.Namespace) -> DatasetBuildConfig:
    """ build config from cli.
    
    Args:
        args: Input parameter.
    
    Returns:
        object: Function return value.
    """
    if not args.add_forward_edges and not args.add_backward_edges:
        print(
            "Warning: neither --add_forward_edges nor --add_backward_edges was "
            "provided. Defaulting to both."
        )
        args.add_forward_edges = True
        args.add_backward_edges = True

    total_nodes = derive_total_nodes(args)
    return DatasetBuildConfig(
        graph_type=args.graph_type,
        path_length=args.path_length,
        total_nodes=total_nodes,
        train_split_ratio=args.train_split_ratio,
        split_subtree_holdout=args.split_subtree_holdout,
        add_forward_edges=args.add_forward_edges,
        add_backward_edges=args.add_backward_edges,
        include_start_node_in_path_finetuning=args.include_start_node_in_path_finetuning,
        use_directional_edge_pretraining=args.use_directional_edge_pretraining,
        random_seed=args.random_seed,
        dataset_root=Path(args.dataset_root),
        overwrite=args.overwrite,
        randomize_node_ids=args.randomize_node_ids,
        star_degree=args.star_degree,
        star_subtree_degree=args.star_subtree_degree,
        grid_rows=args.grid_rows,
        grid_cols=args.grid_cols,
        irregular_edge_count=args.irregular_edge_count,
    )


def _validate_output_paths(output_paths: Dict[str, Path], overwrite: bool):
    """ validate output paths.
    
    Args:
        output_paths: Input parameter.
        overwrite: Input parameter.
    
    Returns:
        object: Function return value.
    """
    if overwrite:
        return
    existing_paths = [path for path in output_paths.values() if path.exists()]
    if existing_paths:
        path_list = "\n".join(str(path) for path in existing_paths)
        raise FileExistsError(
            "Refusing to overwrite existing dataset files.\n"
            "Use --overwrite to replace them.\n"
            f"Existing files:\n{path_list}"
        )


def _build_arg_parser() -> argparse.ArgumentParser:
    """ build arg parser.
    
    Args:
        None: This callable does not take external parameters.
    
    Returns:
        object: Function return value.
    """
    parser = argparse.ArgumentParser(
        description="Build in-weights pretrain/train/test dataset files."
    )
    parser.add_argument(
        "--graph_type",
        type=str,
        required=True,
        choices=["star", "cycle", "grid", "irregular"],
        help="Graph family to generate.",
    )
    parser.add_argument(
        "--path_length",
        type=int,
        default=5,
        help="Number of nodes in each target path sequence.",
    )
    parser.add_argument(
        "--total_nodes",
        type=int,
        default=-1,
        help=(
            "Total node count. If omitted, auto-derived for star/grid/irregular."
            " Required for cycle."
        ),
    )
    parser.add_argument(
        "--train_split_ratio",
        type=float,
        default=0.75,
        help="Fraction of path samples used for train split.",
    )
    parser.add_argument(
        "--split_subtree_holdout",
        action="store_true",
        help=(
            "When graph_type=star and star_subtree_degree>1, split train/test by "
            "subtree roots instead of by individual leaf paths."
        ),
    )
    parser.add_argument(
        "--add_forward_edges",
        action="store_true",
        help="Include forward edges in pretrain file.",
    )
    parser.add_argument(
        "--add_backward_edges",
        action="store_true",
        help="Include reversed edges in pretrain file.",
    )
    parser.add_argument(
        "--include_start_node_in_path_finetuning",
        action="store_true",
        help="Use `start,leaf` prefix instead of `leaf` in train/test files.",
    )
    parser.add_argument(
        "--use_directional_edge_pretraining",
        action="store_true",
        help="Sets the second `sd` filename bit. (Metadata toggle.)",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=0,
        help="Random seed for deterministic generation.",
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="data/datasets/in_weights_graphs",
        help="Base folder where graph-type subdirectories are created.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files if they already exist.",
    )
    parser.add_argument(
        "--randomize_node_ids",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Randomize node IDs before building graph structures.",
    )

    # Star-specific.
    parser.add_argument(
        "--star_degree",
        type=int,
        default=10000,
        help="Root degree for star/path-star generation.",
    )
    parser.add_argument(
        "--star_subtree_degree",
        type=int,
        default=1,
        help="Subtree branching factor for star-tree generation.",
    )

    # Grid-specific.
    parser.add_argument(
        "--grid_rows",
        type=int,
        default=4,
        help="Grid rows (used when graph_type=grid).",
    )
    parser.add_argument(
        "--grid_cols",
        type=int,
        default=4,
        help="Grid columns (used when graph_type=grid).",
    )

    # Irregular-specific.
    parser.add_argument(
        "--irregular_edge_count",
        type=int,
        default=20,
        help="Fixed undirected edge count for the predefined irregular graph (20).",
    )
    return parser


def main(argv: Sequence[str] | None = None):
    """Main.
    
    Args:
        argv: Input parameter.
    
    Returns:
        object: Function return value.
    """
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    config = _build_config_from_cli(args)
    rng = random.Random(config.random_seed)

    forward_edges, all_paths = _build_graph_data(config=config, rng=rng)
    if (
        config.graph_type == "star"
        and config.star_subtree_degree > 1
        and config.split_subtree_holdout
    ):
        train_paths, test_paths = _split_star_paths_by_subtree(
            all_paths=all_paths,
            train_ratio=config.train_split_ratio,
            rng=rng,
        )
    else:
        train_paths, test_paths = _split_paths(
            all_paths=all_paths,
            train_ratio=config.train_split_ratio,
            rng=rng,
        )
    directed_edges = _directed_edges_for_pretraining(
        forward_edges=forward_edges,
        include_forward=config.add_forward_edges,
        include_backward=config.add_backward_edges,
    )

    output_paths = _resolve_output_paths(
        config=config, train_size=len(train_paths), test_size=len(test_paths)
    )
    _validate_output_paths(output_paths, overwrite=config.overwrite)

    _write_edge_memorization_file(output_paths["pretrain"], directed_edges)
    _write_path_finetuning_file(
        output_paths["train"],
        train_paths,
        include_start_node_in_prefix=config.include_start_node_in_path_finetuning,
    )
    _write_path_finetuning_file(
        output_paths["test"],
        test_paths,
        include_start_node_in_prefix=config.include_start_node_in_path_finetuning,
    )

    print("Dataset generation complete:")
    print(f"- pretrain: {output_paths['pretrain']} ({len(directed_edges)} lines)")
    print(f"- train:    {output_paths['train']} ({len(train_paths)} lines)")
    print(f"- test:     {output_paths['test']} ({len(test_paths)} lines)")


if __name__ == "__main__":
    main()
