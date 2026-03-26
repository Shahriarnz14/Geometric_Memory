"""Data loading utilities for in-weights experiments."""

from __future__ import annotations

import os
import torch
from torch.utils.data import Dataset

from geometric_memory.data.dataset_naming import build_dataset_split_filename
from geometric_memory.tokenizing.numeral_tokenizer import NumeralTokenizer

PAUSE_TOKEN = NumeralTokenizer.PAUSE_TOKEN


def load_edge_memorization_pairs(
    filename,
    drop_pause_token=True,
    include_task_token_in_prefix=True,
):
    """Loads edge memorization samples from `u=v` rows.

    Args:
        filename: Input parameter.
        drop_pause_token: Input parameter.
        include_task_token_in_prefix: Input parameter.

    Returns:
        object: Function return value.
    """
    samples = []
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if "=" not in line:
                continue
            source_node, target_node = line.split("=", maxsplit=1)
            prefix = f"[EDGE]{source_node}" if include_task_token_in_prefix else source_node
            if not drop_pause_token:
                prefix += PAUSE_TOKEN
            samples.append((prefix, target_node))

    print(f"Loaded {len(samples)} edge memorization pairs from {filename}")
    return samples


def load_path_finetuning_pairs(
    filename,
    reverse_path_targets=False,
    path_prefix_pause_token_count=1,
    predict_full_path=True,
    include_task_token_in_prefix=True,
):
    """Loads path finetuning samples from `prefix=path` rows.

    Args:
        filename: Input parameter.
        reverse_path_targets: Input parameter.
        path_prefix_pause_token_count: Input parameter.
        predict_full_path: Input parameter.
        include_task_token_in_prefix: Input parameter.

    Returns:
        object: Function return value.
    """
    samples = []
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if "=" not in line:
                continue
            prefix, path_target = line.split("=", maxsplit=1)
            prefix += PAUSE_TOKEN * path_prefix_pause_token_count
            if include_task_token_in_prefix:
                prefix = f"[PATH]{prefix}"

            if not predict_full_path:
                # hardest/indecipherable token is the first token after root
                target_nodes = path_target.split(",")
                if len(target_nodes) < 2:
                    raise ValueError(
                        "Expected at least 2 nodes in path target for hardest-token "
                        f"mode, got: {path_target}"
                    )
                path_target = target_nodes[1]
            elif reverse_path_targets:
                path_target = ",".join(path_target.split(",")[::-1])

            samples.append((prefix, path_target))

    print(f"Loaded {len(samples)} path finetuning pairs from {filename}")
    return samples


def build_dataset_split_path(args, split_suffix):
    """Builds path to pretrain/train/test file for the current config.

    Args:
        args: Input parameter.
        split_suffix: Input parameter.

    Returns:
        object: Function return value.
    """
    filename = build_dataset_split_filename(args, split_suffix)
    return os.path.join(str(args.dataset_directory), filename)


class VariableLengthBatchCollator:
    """Pads variable-length `(x, y)` batches for mixed training."""

    def __init__(self, pad_token_id):
        """  init  .
        
        Args:
            pad_token_id: Input parameter.
        
        Returns:
            object: Function return value.
        """
        self.pad_token_id = pad_token_id

    def __call__(self, batch):
        """  call  .
        
        Args:
            batch: Input parameter.
        
        Returns:
            object: Function return value.
        """
        inputs, targets = zip(*batch)
        padded_inputs = torch.nn.utils.rnn.pad_sequence(
            inputs, batch_first=True, padding_value=self.pad_token_id
        )
        padded_targets = torch.nn.utils.rnn.pad_sequence(
            targets, batch_first=True, padding_value=-1
        )
        return padded_inputs, padded_targets


class EdgeMemorizationDataset(Dataset):
    """Dataset for edge memorization training."""

    def __init__(
        self,
        tokenizer,
        data_path,
        device,
        eval_mode=False,
        teacherless_token_id=None,
        drop_pause_token=True,
        include_task_token_in_prefix=True,
    ):
        """  init  .
        
        Args:
            tokenizer: Input parameter.
            data_path: Input parameter.
            device: Input parameter.
            eval_mode: Input parameter.
            teacherless_token_id: Input parameter.
            drop_pause_token: Input parameter.
            include_task_token_in_prefix: Input parameter.
        
        Returns:
            object: Function return value.
        """
        self.tokenizer = tokenizer
        self.device = device
        self.eval_mode = eval_mode
        self.teacherless_token_id = teacherless_token_id
        self.data_path = data_path
        self.drop_pause_token = drop_pause_token
        self.include_task_token_in_prefix = include_task_token_in_prefix

        self.pairs = load_edge_memorization_pairs(
            self.data_path,
            drop_pause_token=self.drop_pause_token,
            include_task_token_in_prefix=self.include_task_token_in_prefix,
        )
        (
            self.tokenized_sequences,
            self.prefix_token_count,
            self.target_token_count,
        ) = tokenizer.tokenize(self.pairs)
        # Maintained attribute names expected by evaluation utilities.
        self.num_prefix_tokens = self.prefix_token_count
        self.num_target_tokens = self.target_token_count

        self.sequence_token_count = self.prefix_token_count + self.target_token_count

        print(f"Edge memorization dataset size: {len(self.pairs)}")
        print(
            f"Edge dataset tokens -> prefix={self.prefix_token_count}, "
            f"target={self.target_token_count}"
        )

    def __len__(self):
        """  len  .
        
        Args:
            None: This callable does not take external parameters.
        
        Returns:
            object: Function return value.
        """
        return len(self.pairs)

    def __getitem__(self, index):
        """  getitem  .
        
        Args:
            index: Input parameter.
        
        Returns:
            object: Function return value.
        """
        if self.eval_mode:
            return self.tokenized_sequences[index].to(self.device)

        input_tokens = self.tokenized_sequences[index][:-1].clone()
        if self.teacherless_token_id is not None:
            input_tokens[self.prefix_token_count :] = self.teacherless_token_id

        target_tokens = torch.cat(
            [
                -torch.ones(self.prefix_token_count - 1),
                self.tokenized_sequences[index][self.prefix_token_count :].clone(),
            ]
        )

        return input_tokens.to(self.device), target_tokens.long().to(self.device)

    def eval(self):
        """Eval.
        
        Args:
            None: This callable does not take external parameters.
        
        Returns:
            object: Function return value.
        """
        self.eval_mode = True

    def train(self):
        """Train.
        
        Args:
            None: This callable does not take external parameters.
        
        Returns:
            object: Function return value.
        """
        self.eval_mode = False


class PathFinetuningDataset(Dataset):
    """Dataset for path finetuning training/evaluation."""

    def __init__(
        self,
        tokenizer,
        data_path,
        device,
        eval_mode=False,
        teacherless_token_id=None,
        reverse_path_targets=False,
        path_prefix_pause_token_count=1,
        predict_full_path=True,
        include_task_token_in_prefix=True,
    ):
        """  init  .
        
        Args:
            tokenizer: Input parameter.
            data_path: Input parameter.
            device: Input parameter.
            eval_mode: Input parameter.
            teacherless_token_id: Input parameter.
            reverse_path_targets: Input parameter.
            path_prefix_pause_token_count: Input parameter.
            predict_full_path: Input parameter.
            include_task_token_in_prefix: Input parameter.
        
        Returns:
            object: Function return value.
        """
        self.tokenizer = tokenizer
        self.device = device
        self.eval_mode = eval_mode
        self.teacherless_token_id = teacherless_token_id
        self.data_path = data_path
        self.reverse_path_targets = reverse_path_targets
        self.path_prefix_pause_token_count = path_prefix_pause_token_count
        self.predict_full_path = predict_full_path
        self.include_task_token_in_prefix = include_task_token_in_prefix

        self.pairs = load_path_finetuning_pairs(
            self.data_path,
            reverse_path_targets=self.reverse_path_targets,
            path_prefix_pause_token_count=self.path_prefix_pause_token_count,
            predict_full_path=self.predict_full_path,
            include_task_token_in_prefix=self.include_task_token_in_prefix,
        )
        (
            self.tokenized_sequences,
            self.prefix_token_count,
            self.target_token_count,
        ) = tokenizer.tokenize(self.pairs)
        # Maintained attribute names expected by evaluation utilities.
        self.num_prefix_tokens = self.prefix_token_count
        self.num_target_tokens = self.target_token_count

        self.sequence_token_count = self.prefix_token_count + self.target_token_count

        print(f"Path finetuning dataset size: {len(self.pairs)}")
        print(
            f"Path dataset tokens -> prefix={self.prefix_token_count}, "
            f"target={self.target_token_count}"
        )

    def __len__(self):
        """  len  .
        
        Args:
            None: This callable does not take external parameters.
        
        Returns:
            object: Function return value.
        """
        return len(self.pairs)

    def __getitem__(self, index):
        """  getitem  .
        
        Args:
            index: Input parameter.
        
        Returns:
            object: Function return value.
        """
        if self.eval_mode:
            return self.tokenized_sequences[index].to(self.device)

        input_tokens = self.tokenized_sequences[index][:-1].clone()
        if self.teacherless_token_id is not None:
            input_tokens[self.prefix_token_count :] = self.teacherless_token_id

        target_tokens = torch.cat(
            [
                -torch.ones(self.prefix_token_count - 1),
                self.tokenized_sequences[index][self.prefix_token_count :].clone(),
            ]
        )

        return input_tokens.to(self.device), target_tokens.long().to(self.device)

    def eval(self):
        """Eval.
        
        Args:
            None: This callable does not take external parameters.
        
        Returns:
            object: Function return value.
        """
        self.eval_mode = True

    def train(self):
        """Train.
        
        Args:
            None: This callable does not take external parameters.
        
        Returns:
            object: Function return value.
        """
        self.eval_mode = False
