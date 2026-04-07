"""Training loops for edge memorization and path finetuning."""

from contextlib import nullcontext
import pickle

import torch
from tqdm.auto import tqdm

from geometric_memory.in_weights.evaluation import (
    evaluate,
    evaluate_edge_memorization,
    evaluate_forced,
)
from geometric_memory.in_weights.utils import (
    build_batched_prefix_tensor,
    build_edge_prefix_strings_for_all_nodes,
    get_node_embeddings,
    get_top_k_predictions_for_all_nodes,
)
from geometric_memory.utils.experiment_logging import (
    make_prediction_rows,
    prediction_rows_to_markdown,
    write_all_scalars_once,
)
from geometric_memory.utils.training_utils import AverageMeter, get_lr
from geometric_memory.utils.device import is_cuda_device


def _create_autocast_context(device: str):
    """ create autocast context.
    
    Args:
        device: Input parameter.
    
    Returns:
        object: Function return value.
    """
    use_cuda = is_cuda_device(device) and torch.cuda.is_available()
    if use_cuda:
        precision_name = "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
    else:
        precision_name = "float32"
    precision_dtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[precision_name]
    gradient_scaler = torch.cuda.amp.GradScaler(enabled=(use_cuda and precision_name == "float16"))
    autocast_context = (
        nullcontext()
        if not use_cuda
        else torch.amp.autocast(device_type=device, dtype=precision_dtype)
    )
    return gradient_scaler, autocast_context


def _extract_path_eval_scalars(path_eval_results):
    """ extract path eval scalars.
    
    Args:
        path_eval_results: Input parameter.
    
    Returns:
        object: Function return value.
    """
    metrics = {}
    if "test/accuracy" in path_eval_results:
        metrics["test/accuracy"] = path_eval_results["test/accuracy"]
        for metric_key, metric_value in path_eval_results.items():
            if metric_key.startswith("test/token_"):
                token_id = metric_key.split("_")[-1]
                metrics[f"test/token_{token_id}"] = metric_value

    if "test/forced accuracy" in path_eval_results:
        metrics["test_forced/accuracy"] = path_eval_results["test/forced accuracy"]
    if "test/forced loss" in path_eval_results:
        metrics["test_forced/loss"] = path_eval_results["test/forced loss"]
        for metric_key, metric_value in path_eval_results.items():
            if metric_key.startswith("test/forced_token_"):
                token_id = metric_key.split("_")[-1]
                metrics[f"test_forced/token_{token_id}"] = metric_value
    return metrics


def _log_path_prediction_samples(
    experiment_logger,
    tokenizer,
    path_eval_results,
    tag,
    step=None,
):
    """ log path prediction samples.
    
    Args:
        experiment_logger: Input parameter.
        tokenizer: Input parameter.
        path_eval_results: Input parameter.
        tag: Input parameter.
        step: Input parameter.
    
    Returns:
        object: Function return value.
    """
    if experiment_logger is None or "prediction_txt" not in path_eval_results:
        return

    prediction_rows = make_prediction_rows(
        tokenizer, path_eval_results["prediction_txt"], max_rows=20
    )
    if not prediction_rows:
        return

    if hasattr(experiment_logger, "log_prediction_rows"):
        experiment_logger.log_prediction_rows(tag, prediction_rows, step=step)

    markdown_text = prediction_rows_to_markdown(prediction_rows)
    if hasattr(experiment_logger, "log_text"):
        experiment_logger.log_text(f"{tag}_markdown", markdown_text, step=step)


def _evaluate_path_generation(model, data_loader, autocast_context):
    """ evaluate path generation.
    
    Args:
        model: Input parameter.
        data_loader: Input parameter.
        autocast_context: Input parameter.
    
    Returns:
        object: Function return value.
    """
    evaluation_results = {}
    evaluation_results = evaluate(
        model,
        data_loader,
        temperature=0.01,
        ctx=autocast_context,
        top_k=1,
        results=evaluation_results,
        mode="test",
    )
    evaluation_results = evaluate_forced(
        model,
        data_loader,
        ctx=autocast_context,
        results=evaluation_results,
        mode="test",
    )
    return evaluation_results


def _should_log_prediction_samples(epoch_index, args):
    """ should log prediction samples.
    
    Args:
        epoch_index: Input parameter.
        args: Input parameter.
    
    Returns:
        object: Function return value.
    """
    interval = max(
        1,
        args.path_finetuning_eval_interval_epochs * args.prediction_logging_multiplier,
    )
    return (epoch_index % interval == 0) or (epoch_index == (args.path_finetuning_epochs - 1))


def run_edge_memorization_training(
    model,
    edge_train_loader,
    args,
    device,
    tokenizer,
    logging_enabled=False,
    experiment_logger=None,
):
    """Runs edge memorization training.

    Args:
        model: Input parameter.
        edge_train_loader: Input parameter.
        args: Input parameter.
        device: Input parameter.
        tokenizer: Input parameter.
        logging_enabled: Input parameter.
        experiment_logger: Input parameter.

    Returns:
        object: Function return value.
    """
    print("\n" + "=" * 60)
    print("EDGE MEMORIZATION TRAINING")
    print("=" * 60)

    gradient_scaler, autocast_context = _create_autocast_context(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.edge_memorization_learning_rate,
        weight_decay=args.optimizer_weight_decay,
    )
    logging_enabled = bool(logging_enabled and experiment_logger is not None)

    warmup_steps = args.edge_memorization_warmup_steps
    decay_total_steps = args.edge_memorization_epochs * len(edge_train_loader)
    min_learning_rate = args.edge_memorization_learning_rate / 10
    decay_learning_rate = not args.disable_edge_memorization_lr_decay

    model.train()
    optimizer_step_index = 0

    early_stop_patience = 250
    minimum_epoch_before_early_stop = min(1500, abs(args.edge_memorization_epochs - 2))
    epochs_without_improvement = 0
    best_edge_accuracy = 0.0
    best_checkpoint_path = f"{args.output_checkpoint_dir}/{args.run_name}_edge_memorization_best.pt"

    if args.track_embedding_evolution:
        embedding_history_by_step = {}
    if args.track_top_k_predictions:
        all_prefix_batch = build_batched_prefix_tensor(
            tokenizer=tokenizer,
            device=device,
            prefix_strings=build_edge_prefix_strings_for_all_nodes(args),
        )
        top_k_history_by_step = {}

    epoch_progress_bar = tqdm(
        range(args.edge_memorization_epochs),
        desc="Edge Training",
    )
    for epoch_index in epoch_progress_bar:
        epoch_loss_meter = AverageMeter()
        epoch_accuracy_meter = AverageMeter()

        for input_tokens, target_tokens in edge_train_loader:
            if args.track_embedding_evolution:
                embedding_history_by_step[optimizer_step_index] = get_node_embeddings(
                    model, args.total_nodes
                )
            if args.track_top_k_predictions:
                top_k_history_by_step[optimizer_step_index] = get_top_k_predictions_for_all_nodes(
                    model, all_prefix_batch, k=5
                )

            learning_rate = (
                get_lr(
                    optimizer_step_index,
                    args.edge_memorization_learning_rate,
                    warmup_steps,
                    decay_total_steps,
                    min_learning_rate,
                )
                if decay_learning_rate
                else args.edge_memorization_learning_rate
            )
            for parameter_group in optimizer.param_groups:
                parameter_group["lr"] = learning_rate

            with autocast_context:
                _, loss, accuracy_dict = model(input_tokens, target_tokens)

            epoch_loss_meter.update(loss.item(), input_tokens.shape[0])
            epoch_accuracy_meter.update(accuracy_dict["acc"], input_tokens.shape[0])

            gradient_scaler.scale(loss).backward()
            gradient_scaler.step(optimizer)
            gradient_scaler.update()
            optimizer.zero_grad(set_to_none=True)
            optimizer_step_index += 1

            epoch_progress_bar.set_description(
                f"Edge Epoch {epoch_index + 1}/{args.edge_memorization_epochs}"
            )
            epoch_progress_bar.set_postfix(
                loss=f"{epoch_loss_meter.get():.4f}",
                acc=f"{epoch_accuracy_meter.get(percentage=True):.2f}%",
            )

        if epoch_index >= minimum_epoch_before_early_stop:
            epoch_accuracy = epoch_accuracy_meter.get(percentage=True)
            if epoch_accuracy > best_edge_accuracy:
                best_edge_accuracy = epoch_accuracy
                epochs_without_improvement = 0
                torch.save(model.state_dict(), best_checkpoint_path)
                print(
                    f"\nEdge accuracy improved to {best_edge_accuracy:.2f}%."
                    f" Saved best checkpoint: {best_checkpoint_path}"
                )
            else:
                epochs_without_improvement += 1

            if (
                args.enable_edge_memorization_early_stopping
                and epochs_without_improvement >= early_stop_patience
            ):
                print(f"\nEarly stopping: no improvement for {early_stop_patience} epochs.")
                break

        if (
            (epoch_index + 1) % args.edge_memorization_eval_interval_epochs == 0
            or epoch_index == args.edge_memorization_epochs - 1
        ) and logging_enabled:
            write_all_scalars_once(
                experiment_logger,
                {
                    "edge_memorization/train_loss": epoch_loss_meter.get(),
                    "edge_memorization/train_accuracy": epoch_accuracy_meter.get(percentage=True),
                },
                epoch_index,
            )

    print(f"\nEdge memorization complete. Best edge accuracy: " f"{best_edge_accuracy:.2f}%")

    if args.track_embedding_evolution:
        output_path = args.embedding_evolution_dir / args.embedding_evolution_filename
        with output_path.open("wb") as f:
            pickle.dump(embedding_history_by_step, f, protocol=pickle.HIGHEST_PROTOCOL)

    if args.track_top_k_predictions:
        output_path = args.top_k_prediction_dir / args.top_k_prediction_filename
        with output_path.open("wb") as f:
            pickle.dump(top_k_history_by_step, f, protocol=pickle.HIGHEST_PROTOCOL)

    return model


def run_path_finetuning_training(
    model,
    path_train_loader,
    path_test_loader,
    edge_eval_loader,
    args,
    device,
    tokenizer,
    logging_enabled=False,
    experiment_logger=None,
):
    """Runs staged path finetuning after edge memorization.

    Args:
        model: Input parameter.
        path_train_loader: Input parameter.
        path_test_loader: Input parameter.
        edge_eval_loader: Input parameter.
        args: Input parameter.
        device: Input parameter.
        tokenizer: Input parameter.
        logging_enabled: Input parameter.
        experiment_logger: Input parameter.

    Returns:
        object: Function return value.
    """
    print("\n" + "=" * 60)
    print("PATH FINETUNING TRAINING")
    print("=" * 60)

    gradient_scaler, autocast_context = _create_autocast_context(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.path_finetuning_learning_rate,
        weight_decay=args.optimizer_weight_decay,
    )
    logging_enabled = bool(logging_enabled and experiment_logger is not None)

    warmup_steps = args.path_finetuning_warmup_steps
    decay_total_steps = args.path_finetuning_epochs * len(path_train_loader)
    min_learning_rate = args.path_finetuning_learning_rate / 10
    decay_learning_rate = not args.disable_path_finetuning_lr_decay

    model.train()
    optimizer_step_index = 0
    best_path_accuracy = -1.0

    if logging_enabled:
        initial_eval_results = _evaluate_path_generation(
            model,
            path_test_loader,
            autocast_context,
        )
        write_all_scalars_once(
            experiment_logger,
            _extract_path_eval_scalars(initial_eval_results),
            0,
        )
        _log_path_prediction_samples(
            experiment_logger,
            tokenizer,
            initial_eval_results,
            tag="path_finetuning/before_training_predictions",
            step=0,
        )

    epoch_progress_bar = tqdm(
        range(args.path_finetuning_epochs),
        desc="Path Training",
    )
    for epoch_index in epoch_progress_bar:
        epoch_loss_meter = AverageMeter()
        epoch_accuracy_meter = AverageMeter()

        for input_tokens, target_tokens in path_train_loader:
            learning_rate = (
                get_lr(
                    optimizer_step_index,
                    args.path_finetuning_learning_rate,
                    warmup_steps,
                    decay_total_steps,
                    min_learning_rate,
                )
                if decay_learning_rate
                else args.path_finetuning_learning_rate
            )
            for parameter_group in optimizer.param_groups:
                parameter_group["lr"] = learning_rate

            with autocast_context:
                _, loss, accuracy_dict = model(input_tokens, target_tokens)

            epoch_loss_meter.update(loss.item(), input_tokens.shape[0])
            epoch_accuracy_meter.update(accuracy_dict["acc"], input_tokens.shape[0])

            gradient_scaler.scale(loss).backward()
            gradient_scaler.step(optimizer)
            gradient_scaler.update()
            optimizer.zero_grad(set_to_none=True)
            optimizer_step_index += 1

            epoch_progress_bar.set_description(
                f"Path Epoch {epoch_index + 1}/{args.path_finetuning_epochs}"
            )
            epoch_progress_bar.set_postfix(
                loss=f"{epoch_loss_meter.get():.4f}",
                acc=f"{epoch_accuracy_meter.get(percentage=True):.2f}%",
            )

        should_evaluate = (epoch_index % args.path_finetuning_eval_interval_epochs == 0) or (
            epoch_index == args.path_finetuning_epochs - 1
        )
        if should_evaluate:
            eval_results = _evaluate_path_generation(
                model,
                path_test_loader,
                autocast_context,
            )
            test_accuracy = eval_results.get("test/accuracy", 0.0)
            forced_accuracy = eval_results.get("test/forced accuracy", 0.0)
            print(
                f"Epoch {epoch_index + 1} | Test Acc: {test_accuracy:.2f}%"
                f" | Forced Acc: {forced_accuracy:.2f}%"
            )

            is_best_checkpoint = test_accuracy > best_path_accuracy
            if is_best_checkpoint:
                best_path_accuracy = test_accuracy
                best_path_checkpoint = (
                    f"{args.output_checkpoint_dir}/{args.run_name}_path_finetuning_best.pt"
                )
                torch.save(model.state_dict(), best_path_checkpoint)
                print(f"Saved new best path checkpoint: {best_path_checkpoint}")

            if logging_enabled:
                scalar_metrics = {
                    "path_finetuning/train_loss": epoch_loss_meter.get(),
                    "path_finetuning/train_accuracy": epoch_accuracy_meter.get(percentage=True),
                }
                scalar_metrics.update(_extract_path_eval_scalars(eval_results))
                write_all_scalars_once(
                    experiment_logger,
                    scalar_metrics,
                    epoch_index + 1,
                )

                if _should_log_prediction_samples(epoch_index, args):
                    _log_path_prediction_samples(
                        experiment_logger,
                        tokenizer,
                        eval_results,
                        tag="path_finetuning/predictions",
                        step=epoch_index + 1,
                    )
                if is_best_checkpoint:
                    _log_path_prediction_samples(
                        experiment_logger,
                        tokenizer,
                        eval_results,
                        tag="path_finetuning/best_predictions",
                        step=epoch_index + 1,
                    )

            if edge_eval_loader is not None:
                evaluate_edge_memorization(
                    model=model,
                    loader=edge_eval_loader,
                    device=device,
                    logging_enabled=logging_enabled,
                    experiment_logger=experiment_logger,
                    step=epoch_index + 1,
                    during_edge_memorization=False,
                )

        if (epoch_index + 1) % args.checkpoint_interval_epochs == 0:
            checkpoint_path = (
                f"{args.output_checkpoint_dir}/{args.run_name}"
                f"_path_finetuning_epoch_{epoch_index + 1}.pt"
            )
            torch.save(model.state_dict(), checkpoint_path)

    print(f"Path finetuning complete. Best test accuracy: {best_path_accuracy:.2f}%")
    return model


def run_joint_edge_and_path_training(
    model,
    mixed_train_loader,
    path_test_loader,
    edge_eval_loader,
    args,
    device,
    tokenizer,
    logging_enabled=False,
    experiment_logger=None,
):
    """Runs mixed edge+path training in a single stage.

    Args:
        model: Input parameter.
        mixed_train_loader: Input parameter.
        path_test_loader: Input parameter.
        edge_eval_loader: Input parameter.
        args: Input parameter.
        device: Input parameter.
        tokenizer: Input parameter.
        logging_enabled: Input parameter.
        experiment_logger: Input parameter.

    Returns:
        object: Function return value.
    """
    print("\n" + "=" * 60)
    print("JOINT EDGE + PATH TRAINING")
    print("=" * 60)

    gradient_scaler, autocast_context = _create_autocast_context(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.path_finetuning_learning_rate,
        weight_decay=args.optimizer_weight_decay,
    )
    logging_enabled = bool(logging_enabled and experiment_logger is not None)

    warmup_steps = args.path_finetuning_warmup_steps
    decay_total_steps = args.path_finetuning_epochs * len(mixed_train_loader)
    min_learning_rate = args.path_finetuning_learning_rate / 10
    decay_learning_rate = not args.disable_path_finetuning_lr_decay

    model.train()
    optimizer_step_index = 0
    best_path_accuracy = -1.0

    epoch_progress_bar = tqdm(
        range(args.path_finetuning_epochs),
        desc="Joint Training",
    )
    for epoch_index in epoch_progress_bar:
        epoch_loss_meter = AverageMeter()
        epoch_accuracy_meter = AverageMeter()

        for input_tokens, target_tokens in mixed_train_loader:
            learning_rate = (
                get_lr(
                    optimizer_step_index,
                    args.path_finetuning_learning_rate,
                    warmup_steps,
                    decay_total_steps,
                    min_learning_rate,
                )
                if decay_learning_rate
                else args.path_finetuning_learning_rate
            )
            for parameter_group in optimizer.param_groups:
                parameter_group["lr"] = learning_rate

            with autocast_context:
                _, loss, accuracy_dict = model(
                    input_tokens,
                    target_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                )

            epoch_loss_meter.update(loss.item(), input_tokens.shape[0])
            epoch_accuracy_meter.update(accuracy_dict["acc"], input_tokens.shape[0])

            gradient_scaler.scale(loss).backward()
            gradient_scaler.step(optimizer)
            gradient_scaler.update()
            optimizer.zero_grad(set_to_none=True)
            optimizer_step_index += 1

            epoch_progress_bar.set_description(
                f"Joint Epoch {epoch_index + 1}/{args.path_finetuning_epochs}"
            )
            epoch_progress_bar.set_postfix(
                loss=f"{epoch_loss_meter.get():.4f}",
                acc=f"{epoch_accuracy_meter.get(percentage=True):.2f}%",
            )

        should_evaluate = (epoch_index % args.path_finetuning_eval_interval_epochs == 0) or (
            epoch_index == args.path_finetuning_epochs - 1
        )
        if should_evaluate:
            eval_results = _evaluate_path_generation(
                model,
                path_test_loader,
                autocast_context,
            )
            test_accuracy = eval_results.get("test/accuracy", 0.0)
            forced_accuracy = eval_results.get("test/forced accuracy", 0.0)
            print(
                f"Epoch {epoch_index + 1} | Test Acc: {test_accuracy:.2f}%"
                f" | Forced Acc: {forced_accuracy:.2f}%"
            )

            is_best_checkpoint = test_accuracy > best_path_accuracy
            if is_best_checkpoint:
                best_path_accuracy = test_accuracy
                best_checkpoint_path = (
                    f"{args.output_checkpoint_dir}/{args.run_name}_joint_training_best.pt"
                )
                torch.save(model.state_dict(), best_checkpoint_path)
                print(f"Saved new best joint checkpoint: {best_checkpoint_path}")

            if logging_enabled:
                scalar_metrics = {
                    "joint_training/train_loss": epoch_loss_meter.get(),
                    "joint_training/train_accuracy": epoch_accuracy_meter.get(percentage=True),
                }
                scalar_metrics.update(_extract_path_eval_scalars(eval_results))
                write_all_scalars_once(
                    experiment_logger,
                    scalar_metrics,
                    epoch_index + 1,
                )

                if _should_log_prediction_samples(epoch_index, args):
                    _log_path_prediction_samples(
                        experiment_logger,
                        tokenizer,
                        eval_results,
                        tag="joint_training/predictions",
                        step=epoch_index + 1,
                    )
                if is_best_checkpoint:
                    _log_path_prediction_samples(
                        experiment_logger,
                        tokenizer,
                        eval_results,
                        tag="joint_training/best_predictions",
                        step=epoch_index + 1,
                    )

            if edge_eval_loader is not None:
                evaluate_edge_memorization(
                    model=model,
                    loader=edge_eval_loader,
                    device=device,
                    logging_enabled=logging_enabled,
                    experiment_logger=experiment_logger,
                    step=epoch_index + 1,
                    during_edge_memorization=False,
                )

        if (epoch_index + 1) % args.checkpoint_interval_epochs == 0:
            checkpoint_path = (
                f"{args.output_checkpoint_dir}/{args.run_name}"
                f"_joint_training_epoch_{epoch_index + 1}.pt"
            )
            torch.save(model.state_dict(), checkpoint_path)

    print(f"Joint training complete. Best test accuracy: {best_path_accuracy:.2f}%")
    return model
