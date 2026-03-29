"""Shared training helpers used across training and evaluation code."""

from __future__ import annotations

import math

import torch


def get_lr(
    step_index: int,
    learning_rate: float,
    warmup_steps: int,
    lr_decay_steps: int,
    min_learning_rate: float,
) -> float:
    """Cosine LR schedule with linear warmup and minimum LR floor.

    Args:
        step_index: Input parameter.
        learning_rate: Input parameter.
        warmup_steps: Input parameter.
        lr_decay_steps: Input parameter.
        min_learning_rate: Input parameter.

    Returns:
        object: Function return value.
    """
    if warmup_steps > 0 and step_index < warmup_steps:
        return learning_rate * step_index / warmup_steps

    # Guard against invalid or degenerate decay windows.
    if lr_decay_steps <= warmup_steps:
        return min_learning_rate if step_index >= lr_decay_steps else learning_rate

    if step_index > lr_decay_steps:
        return min_learning_rate

    decay_ratio = (step_index - warmup_steps) / (lr_decay_steps - warmup_steps)
    cosine_coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_learning_rate + cosine_coeff * (learning_rate - min_learning_rate)


class AverageMeter:
    """Tracks a weighted running average."""

    def __init__(self):
        """  init  .
        
        Args:
            None: This callable does not take external parameters.
        
        Returns:
            object: Function return value.
        """
        self.total = 0.0
        self.count = 0

    def update(self, value=None, weight=None, **kwargs):
        """Updates meter state.

        Args:
            value: Input parameter.
            weight: Input parameter.

        Returns:
            object: Function return value.
        """
        if value is None and "val" in kwargs:
            value = kwargs.pop("val")
        if weight is None and "num" in kwargs:
            weight = kwargs.pop("num")
        if kwargs:
            unknown = ", ".join(sorted(kwargs))
            raise TypeError(f"Unknown AverageMeter.update kwargs: {unknown}")
        if value is None or weight is None:
            raise TypeError("AverageMeter.update requires value/weight or val/num arguments.")

        self.total += float(value) * int(weight)
        self.count += int(weight)

    def get(self, percentage: bool = False):
        """Get.
        
        Args:
            percentage: Input parameter.
        
        Returns:
            object: Function return value.
        """
        average = self.total / max(self.count, 1)
        return average * 100 if percentage else average


def accuracy(logits, targets):
    """Returns sequence-level and per-token accuracy over non-prefix targets.

    Args:
        logits: Input parameter.
        targets: Input parameter.

    Returns:
        object: Function return value.
    """
    # Prefix/pad positions are masked with -1 and excluded from scoring.
    valid_mask = targets.ne(-1)
    predicted = torch.argmax(logits, dim=-1)
    is_correct = predicted.eq(targets) & valid_mask

    valid_counts_per_example = valid_mask.sum(dim=1)
    sequence_correct = is_correct.sum(dim=1).eq(valid_counts_per_example).to(torch.float)
    sequence_accuracy = sequence_correct.mean()

    # Fast path for fixed-length targets (the common evaluation case).
    if torch.all(valid_mask.eq(valid_mask[0])):
        prefix_token_count = torch.sum(~valid_mask[0]).item()
        per_token_accuracy = is_correct[:, prefix_token_count:].to(torch.float).mean(dim=0)
        return sequence_accuracy, per_token_accuracy

    # Mixed variable-length case: compute per-position accuracy on valid entries only.
    valid_per_position = valid_mask.to(torch.float).sum(dim=0).clamp_min(1.0)
    per_token_accuracy = is_correct.to(torch.float).sum(dim=0) / valid_per_position
    return sequence_accuracy, per_token_accuracy


def _get_arg(args, *names: str, default=None):
    """ get arg.
    
    Args:
        args: Input parameter.
    
    Returns:
        object: Function return value.
    """
    for name in names:
        if hasattr(args, name):
            return getattr(args, name)
    return default


def _stringify_bool(flag_value) -> str:
    """ stringify bool.
    
    Args:
        flag_value: Input parameter.
    
    Returns:
        object: Function return value.
    """
    return str(int(bool(flag_value)))


def get_run_name(args):
    """Builds a descriptive run name for notebook and script workflows.

    Args:
        args: Input parameter.

    Returns:
        object: Function return value.
    """
    dataset_name = _get_arg(args, "dataset", "dataset_name")
    if dataset_name is None:
        raise ValueError("Could not infer dataset name from args.")

    model_name = _get_arg(args, "model_family", "model", default="model")
    degree = _get_arg(args, "star_degree", "deg")
    subtree_degree = _get_arg(args, "star_subtree_degree", "deg_tree")
    path_length = _get_arg(args, "path_length", "path_len")
    total_nodes = _get_arg(args, "total_nodes", "num_nodes")
    train_count = _get_arg(args, "n_train")
    is_teacherless = _stringify_bool(
        _get_arg(args, "use_teacherless_inputs", "teacherless", default=False)
    )
    is_reversed = _stringify_bool(
        _get_arg(args, "reverse_path_targets", "reverse", default=False)
    )

    tokens = [str(dataset_name)]

    if dataset_name == "graph":
        tokens.extend(
            [
                str(model_name),
                f"deg_{degree}",
                f"path_{path_length}",
                f"num_nodes_{total_nodes}",
                f"n_train_{train_count}",
                f"teacherless_{is_teacherless}",
                f"reverse_{is_reversed}",
            ]
        )
        return "_".join(tokens)

    if dataset_name == "graph_tree":
        tokens.extend(
            [
                str(model_name),
                f"deg_tree_{subtree_degree}",
                f"path_{path_length}",
                f"num_nodes_{total_nodes}",
                f"n_train_{train_count}",
                f"teacherless_{is_teacherless}",
                f"reverse_{is_reversed}",
            ]
        )
    elif dataset_name == "graph_bat":
        wing = _get_arg(args, "wing")
        tokens.extend(
            [
                str(model_name),
                f"deg_{degree}",
                f"wing_{wing}",
                f"path_{path_length}",
                f"num_nodes_{total_nodes}",
                f"n_train_{train_count}",
                f"teacherless_{is_teacherless}",
                f"reverse_{is_reversed}",
            ]
        )
    elif dataset_name == "in_weights":
        include_start = _stringify_bool(
            _get_arg(
                args,
                "include_start_node_in_path_finetuning",
                "include_start_finetuning",
                default=False,
            )
        )
        directional_edges = _stringify_bool(
            _get_arg(
                args,
                "use_directional_edge_pretraining",
                "directional_pretraining",
                default=False,
            )
        )
        include_forward = _stringify_bool(
            _get_arg(args, "add_forward_edges", "include_forward", default=False)
        )
        include_backward = _stringify_bool(
            _get_arg(args, "add_backward_edges", "include_backward", default=False)
        )
        include_self = _stringify_bool(
            _get_arg(args, "add_self_edges", "include_self", default=False)
        )
        task_prefix = _stringify_bool(
            _get_arg(
                args,
                "include_task_token_in_prefix",
                "include_task_in_prefix",
                default=True,
            )
        )
        split_subtrees = _stringify_bool(
            _get_arg(args, "split_subtree_holdout", "split_subtrees", default=False)
        )
        include_edge_pause = _stringify_bool(
            not _get_arg(
                args,
                "edge_memorization_drop_pause_token",
                "exclude_pause_token_phase1",
                default=True,
            )
        )
        path_pause_count = _get_arg(
            args, "path_prefix_pause_token_count", "num_pause_tokens_phase2", default=1
        )
        path_batch_size = _get_arg(
            args, "path_finetuning_batch_size", "batch_size_phase2", default="na"
        )
        path_learning_rate = _get_arg(
            args, "path_finetuning_learning_rate", "lr_phase2", default="na"
        )
        edge_learning_rate = _get_arg(
            args, "edge_memorization_learning_rate", "lr_phase1", default="na"
        )

        sd_token = f"{include_start}{directional_edges}"
        fb_token = f"{include_forward}{include_backward}"
        tokens.extend(
            [
                str(model_name),
                f"deg_{degree}",
                f"deg_tree_{subtree_degree}",
                f"path_{path_length}",
                f"num_nodes_{total_nodes}",
                f"teacherless_{is_teacherless}",
                f"reverse_{is_reversed}",
                f"sd_{sd_token}",
                f"fb_{fb_token}",
                f"selfedge_{include_self}",
                f"taskinprefix_{task_prefix}",
                f"splitsubtrees_{split_subtrees}",
                f"pause_edge_{include_edge_pause}",
                f"pause_path_{path_pause_count}",
                f"batch_size_{path_batch_size}",
                f"lr_{path_learning_rate}",
                f"lr_edge_{edge_learning_rate}",
            ]
        )
    else:
        raise ValueError(
            f"Dataset {dataset_name} does not currently support run name generation."
        )

    if "gpt2" not in str(model_name):
        layer_count = _get_arg(args, "transformer_layer_count", "n_layer")
        embedding_dim = _get_arg(args, "embedding_dimension", "n_embd")
        head_count = _get_arg(args, "attention_head_count", "n_head")
        if layer_count is not None:
            tokens.append(f"n_layer_{layer_count}")
        if embedding_dim is not None:
            tokens.append(f"n_embd_{embedding_dim}")
        if head_count is not None:
            tokens.append(f"n_head_{head_count}")

    tie_embeddings = _get_arg(
        args, "tie_input_output_embeddings", "use_weight_tying", default=True
    )
    freeze_embeddings = _get_arg(
        args, "freeze_token_embeddings", "freeze_embeddings", default=False
    )
    if not bool(tie_embeddings):
        tokens.append("untied")
    if bool(freeze_embeddings):
        tokens.append("associative")

    return "_".join(tokens)
