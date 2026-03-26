"""High-level training recipe execution for in-weights experiments."""

import os

import torch
from torch.utils.data import ConcatDataset, DataLoader

from geometric_memory.in_weights.data_loader import (
    EdgeMemorizationDataset,
    PathFinetuningDataset,
    VariableLengthBatchCollator,
)
from geometric_memory.in_weights.trainer import (
    run_edge_memorization_training,
    run_joint_edge_and_path_training,
    run_path_finetuning_training,
)
from geometric_memory.in_weights.evaluation import evaluate_edge_memorization


def _build_edge_memorization_loader(args, tokenizer, device, data_path):
    """ build edge memorization loader.
    
    Args:
        args: Input parameter.
        tokenizer: Input parameter.
        device: Input parameter.
        data_path: Input parameter.
    
    Returns:
        object: Function return value.
    """
    dataset = EdgeMemorizationDataset(
        tokenizer=tokenizer,
        data_path=data_path,
        device=device,
        teacherless_token_id=args.teacherless_token,
        drop_pause_token=args.edge_memorization_drop_pause_token,
        include_task_token_in_prefix=args.include_task_token_in_prefix,
    )
    loader = DataLoader(dataset, batch_size=args.edge_memorization_batch_size, shuffle=True)
    return dataset, loader


def _build_path_finetuning_loaders(args, tokenizer, device, train_path, test_path, predict_full_path):
    """ build path finetuning loaders.
    
    Args:
        args: Input parameter.
        tokenizer: Input parameter.
        device: Input parameter.
        train_path: Input parameter.
        test_path: Input parameter.
        predict_full_path: Input parameter.
    
    Returns:
        object: Function return value.
    """
    train_dataset = PathFinetuningDataset(
        tokenizer=tokenizer,
        data_path=train_path,
        device=device,
        teacherless_token_id=args.teacherless_token,
        reverse_path_targets=args.reverse_path_targets,
        path_prefix_pause_token_count=args.path_prefix_pause_token_count,
        predict_full_path=predict_full_path,
        include_task_token_in_prefix=args.include_task_token_in_prefix,
    )
    test_dataset = PathFinetuningDataset(
        tokenizer=tokenizer,
        data_path=test_path,
        device=device,
        teacherless_token_id=args.teacherless_token,
        reverse_path_targets=args.reverse_path_targets,
        path_prefix_pause_token_count=args.path_prefix_pause_token_count,
        predict_full_path=predict_full_path,
        include_task_token_in_prefix=args.include_task_token_in_prefix,
    )
    train_loader = DataLoader(train_dataset, batch_size=args.path_finetuning_batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.path_finetuning_batch_size, shuffle=False)
    return train_dataset, test_dataset, train_loader, test_loader


def run_or_load_edge_memorization(args, model, edge_loader, tokenizer, device, experiment_logger):
    """Runs edge memorization or loads an existing edge checkpoint.

    Args:
        args: Input parameter.
        model: Input parameter.
        edge_loader: Input parameter.
        tokenizer: Input parameter.
        device: Input parameter.
        experiment_logger: Input parameter.

    Returns:
        object: Function return value.
    """
    if not args.skip_edge_memorization:
        print("\nStarting edge memorization stage...")
        model = run_edge_memorization_training(
            model=model,
            edge_train_loader=edge_loader,
            args=args,
            device=device,
            tokenizer=tokenizer,
            logging_enabled=args.enable_wandb,
            experiment_logger=experiment_logger,
        )
        final_edge_path = f"{args.output_checkpoint_dir}/{args.run_name}_edge_memorization_final.pt"
        torch.save(model.state_dict(), final_edge_path)
        print(f"Saved edge memorization final checkpoint: {final_edge_path}")
    else:
        print("Skipping edge memorization and loading checkpoint...")
        checkpoint_path = args.edge_memorization_checkpoint_path
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Loaded edge memorization checkpoint: {checkpoint_path}")

    print("Evaluating edge memorization capability...")
    evaluate_edge_memorization(
        model=model,
        loader=edge_loader,
        device=device,
        logging_enabled=args.enable_wandb,
        experiment_logger=experiment_logger,
        during_edge_memorization=True,
    )
    return model


def run_staged_training_recipe(
    args, model, pretrain_path, train_path, test_path, tokenizer, device, experiment_logger,
    predict_full_path=True
):
    """Runs staged recipe: edge memorization then path finetuning.

    Args:
        args: Input parameter.
        model: Input parameter.
        pretrain_path: Input parameter.
        train_path: Input parameter.
        test_path: Input parameter.
        tokenizer: Input parameter.
        device: Input parameter.
        experiment_logger: Input parameter.
        predict_full_path: Input parameter.

    Returns:
        object: Function return value.
    """
    print("\nPreparing edge memorization dataset...")
    if not os.path.exists(pretrain_path):
        raise FileNotFoundError(f"Pretrain file not found: {pretrain_path}")
    edge_dataset, edge_loader = _build_edge_memorization_loader(
        args=args,
        tokenizer=tokenizer,
        device=device,
        data_path=pretrain_path,
    )
    print(f"Edge dataset size: {len(edge_dataset)}")
    print(f"Edge loader size: {len(edge_loader)}")

    model = run_or_load_edge_memorization(
        args=args,
        model=model,
        edge_loader=edge_loader,
        tokenizer=tokenizer,
        device=device,
        experiment_logger=experiment_logger,
    )

    print("\nPreparing path finetuning datasets...")
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Train file not found: {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test file not found: {test_path}")

    (
        path_train_dataset,
        path_test_dataset,
        path_train_loader,
        path_test_loader,
    ) = _build_path_finetuning_loaders(
        args=args,
        tokenizer=tokenizer,
        device=device,
        train_path=train_path,
        test_path=test_path,
        predict_full_path=predict_full_path,
    )
    print(f"Path dataset sizes -> train={len(path_train_dataset)} test={len(path_test_dataset)}")

    model = run_path_finetuning_training(
        model=model,
        path_train_loader=path_train_loader,
        path_test_loader=path_test_loader,
        edge_eval_loader=edge_loader,
        args=args,
        device=device,
        tokenizer=tokenizer,
        logging_enabled=args.enable_wandb,
        experiment_logger=experiment_logger,
    )
    return model


def run_mixed_training_recipe(
    args, model, pretrain_path, train_path, test_path, tokenizer, device, experiment_logger,
    predict_full_path=True
):
    """Runs mixed recipe: joint edge+path training.

    Args:
        args: Input parameter.
        model: Input parameter.
        pretrain_path: Input parameter.
        train_path: Input parameter.
        test_path: Input parameter.
        tokenizer: Input parameter.
        device: Input parameter.
        experiment_logger: Input parameter.
        predict_full_path: Input parameter.

    Returns:
        object: Function return value.
    """
    print("\nPreparing mixed edge/path datasets...")
    if not os.path.exists(pretrain_path):
        raise FileNotFoundError(f"Pretrain file not found: {pretrain_path}")
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Train file not found: {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test file not found: {test_path}")

    edge_dataset, edge_loader = _build_edge_memorization_loader(
        args=args,
        tokenizer=tokenizer,
        device=device,
        data_path=pretrain_path,
    )
    (
        path_train_dataset,
        path_test_dataset,
        _,
        path_test_loader,
    ) = _build_path_finetuning_loaders(
        args=args,
        tokenizer=tokenizer,
        device=device,
        train_path=train_path,
        test_path=test_path,
        predict_full_path=predict_full_path,
    )

    collator = VariableLengthBatchCollator(pad_token_id=tokenizer.pad_token_id)
    mixed_train_dataset = ConcatDataset([edge_dataset, path_train_dataset])
    mixed_train_loader = DataLoader(
        mixed_train_dataset,
        batch_size=args.path_finetuning_batch_size,
        shuffle=True,
        collate_fn=collator,
    )

    print(
        f"Mixed dataset sizes -> edge={len(edge_dataset)} "
        f"path_train={len(path_train_dataset)} "
        f"path_test={len(path_test_dataset)} "
        f"mixed_train={len(mixed_train_dataset)}"
    )

    model = run_joint_edge_and_path_training(
        model=model,
        mixed_train_loader=mixed_train_loader,
        path_test_loader=path_test_loader,
        edge_eval_loader=edge_loader,
        args=args,
        device=device,
        tokenizer=tokenizer,
        logging_enabled=args.enable_wandb,
        experiment_logger=experiment_logger,
    )
    return model
