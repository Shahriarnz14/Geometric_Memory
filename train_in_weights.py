"""Entrypoint for in-weights training recipes."""

import logging
from pathlib import Path

import torch

from geometric_memory.in_weights.config import get_args
from geometric_memory.in_weights.data_loader import build_dataset_split_path
from geometric_memory.in_weights.experiment_modes import get_training_recipe
from geometric_memory.in_weights.experiments import (
    run_mixed_training_recipe,
    run_staged_training_recipe,
)
from geometric_memory.utils.run_management import (
    build_run_name,
    prepare_run_directories,
)
from geometric_memory.models import get_model
from geometric_memory.tokenizing import get_tokenizer
from geometric_memory.utils.experiment_logging import create_experiment_logger
from geometric_memory.utils.device import resolve_default_device


LOGGER = logging.getLogger("geometric_memory.train_in_weights")


def _setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _prepare_runtime_settings(args):
    args.device = resolve_default_device()
    tokenizer = get_tokenizer(args)
    args.vocab_size = tokenizer.vocab_size
    args.block_size = max(64, args.path_length * 3)
    args.teacherless_token = tokenizer.encode("$")[0] if args.use_teacherless_inputs else None
    args.use_flash = False
    return tokenizer


def _prepare_run_layout(args, recipe_name):
    run_name = build_run_name(args, recipe_name=recipe_name)
    run_dirs = prepare_run_directories(
        experiment_log_root=args.experiment_log_root,
        dataset=args.dataset_name,
        run_name=run_name,
    )
    args.run_name = run_name
    args.run_dir = str(run_dirs.run_dir)
    args.output_checkpoint_dir = str(run_dirs.checkpoints_dir)
    args.output_artifact_dir = str(run_dirs.artifacts_dir)


def main(args_list=None):
    _setup_logging()
    args = get_args(args_list)
    training_recipe = get_training_recipe(args.training_recipe)
    tokenizer = _prepare_runtime_settings(args)
    _prepare_run_layout(args, recipe_name=training_recipe.recipe_name)
    experiment_logger = create_experiment_logger(args, run_name=args.run_name)
    run_url = experiment_logger.get_run_url()
    if args.enable_wandb:
        LOGGER.info(
            "W&B requested: mode=%s project=%s entity=%s",
            args.wandb_mode,
            args.wandb_project,
            args.wandb_entity or "<default>",
        )
        if run_url:
            LOGGER.info("W&B run URL: %s", run_url)
        else:
            status_message = experiment_logger.get_status_message()
            if status_message:
                LOGGER.warning("W&B requested but inactive. %s", status_message)
            else:
                LOGGER.warning(
                    "W&B requested but run URL is unavailable. "
                    "Check WANDB login, WANDB_DISABLED, and network access."
                )

    LOGGER.info("Device: %s", args.device)
    LOGGER.info(
        "Graph setup: type=%s star_degree=%s star_subtree_degree=%s path_length=%s "
        "total_nodes=%s",
        args.graph_type,
        args.star_degree,
        args.star_subtree_degree,
        args.path_length,
        args.total_nodes,
    )
    LOGGER.info("Run directory: %s", args.run_dir)
    LOGGER.info("Recipe: %s", training_recipe.description)

    if experiment_logger.enabled:
        experiment_logger.log_text("config/hyperparameters", str(vars(args)))

    model = get_model(args).to(args.device)
    parameter_count = sum(p.numel() for p in model.parameters())
    LOGGER.info("Model parameters: %s", f"{parameter_count:,}")

    pretrain_path = build_dataset_split_path(args, "pretrain")
    train_path = build_dataset_split_path(args, f"train_{args.expected_train_path_count}")
    test_path = build_dataset_split_path(args, f"test_{args.expected_test_path_count}")

    LOGGER.info("Pretrain file: %s", pretrain_path)
    LOGGER.info("Train file: %s", train_path)
    LOGGER.info("Test file: %s", test_path)

    try:
        if training_recipe.use_mixed_edge_and_path_batches:
            model = run_mixed_training_recipe(
                args=args,
                model=model,
                pretrain_path=pretrain_path,
                train_path=train_path,
                test_path=test_path,
                tokenizer=tokenizer,
                device=args.device,
                experiment_logger=experiment_logger,
                predict_full_path=training_recipe.predict_full_path,
            )
        else:
            model = run_staged_training_recipe(
                args=args,
                model=model,
                pretrain_path=pretrain_path,
                train_path=train_path,
                test_path=test_path,
                tokenizer=tokenizer,
                device=args.device,
                experiment_logger=experiment_logger,
                predict_full_path=training_recipe.predict_full_path,
            )

        final_checkpoint_path = Path(args.output_checkpoint_dir) / f"{args.run_name}_final_model.pt"
        torch.save(model.state_dict(), final_checkpoint_path)
        LOGGER.info("Final checkpoint saved: %s", final_checkpoint_path)
    finally:
        experiment_logger.finish()

    LOGGER.info("Training finished.")


if __name__ == "__main__":
    main()
