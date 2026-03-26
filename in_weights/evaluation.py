"""Evaluation utilities for generation and teacher-forced scoring."""

from contextlib import nullcontext

from geometric_memory.utils.experiment_logging import write_all_scalars_once
from geometric_memory.utils.training_utils import AverageMeter
import torch
from geometric_memory.utils.device import is_cuda_device
from tqdm import tqdm


@torch.no_grad()
def evaluate_edge_memorization(
    model,
    loader,
    device,
    logging_enabled=False,
    experiment_logger=None,
    step=None,
    during_edge_memorization=True,
):
    """Evaluate edge memorization performance under teacher forcing.

    Args:
        model: Sequence model to evaluate.
        loader: DataLoader providing edge memorization batches.
        device: Runtime device name (for autocast selection).
        logging_enabled: Whether external metric logging is enabled.
        experiment_logger: Experiment logger implementation.
        step: Optional step/epoch index for scalar logging.
        during_edge_memorization: Whether evaluation occurs in edge stage or later stage.

    Returns:
        object: None. Metrics are printed and optionally logged.
    """
    use_cuda = is_cuda_device(device) and torch.cuda.is_available()
    if use_cuda:
        autocast_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        ctx = torch.amp.autocast(device_type=device, dtype=autocast_dtype)
    else:
        ctx = nullcontext()

    loader_bar = tqdm(loader, desc="Edge Memorization Eval")
    total_loss, total_acc = AverageMeter(), AverageMeter()

    model.eval()
    for x, y in loader_bar:
        with ctx:
            _logits, loss, accs = model(x, y)

        total_loss.update(loss.item(), x.shape[0])
        total_acc.update(accs["acc"], x.shape[0])

        loader_bar.set_description(
            "Edge Memorization | Loss:"
            f" {total_loss.get():.4f} | Acc:"
            f" {total_acc.get(percentage=True):.2f}%"
        )

    print(
        f"Edge Memorization | Loss: {total_loss.get():.4f} | Acc:"
        f" {total_acc.get(percentage=True):.2f}%"
    )

    if logging_enabled and experiment_logger is not None:
        if step is None:
            if during_edge_memorization:
                text = (
                    f"Loss: {total_loss.get():.4f} | "
                    f"Acc: {total_acc.get(percentage=True):.2f}%"
                )
                if hasattr(experiment_logger, "log_text"):
                    experiment_logger.log_text("edge_memorization/training_summary", text)
            else:
                text = (
                    f"Loss: {total_loss.get():.4f} | "
                    f"Acc: {total_acc.get(percentage=True):.2f}%"
                )
                if hasattr(experiment_logger, "log_text"):
                    experiment_logger.log_text("edge_memorization/path_finetuning_summary", text)
        else:
            scalar_dict = {
                "edge_memorization/loss": total_loss.get(),
                "edge_memorization/accuracy": total_acc.get(percentage=True),
            }
            write_all_scalars_once(experiment_logger, scalar_dict, step)


def _update_child_transition_accuracy(
    tokens_child_corr,
    y_pred,
    num_target_tokens,
    adj_list_cache,
    example_idx,
    batch_size,
):
    """ update child transition accuracy.
    
    Args:
        tokens_child_corr: Input parameter.
        y_pred: Input parameter.
        num_target_tokens: Input parameter.
        adj_list_cache: Input parameter.
        example_idx: Input parameter.
        batch_size: Input parameter.
    
    Returns:
        object: Function return value.
    """
    for t in range(1, num_target_tokens):
        parents = y_pred[:, -num_target_tokens + t - 1].tolist()
        children = y_pred[:, -num_target_tokens + t].tolist()

        accs = []
        for i in range(batch_size):
            adj = adj_list_cache[example_idx + i]
            accs.append(1.0 if children[i] in adj.get(parents[i], set()) else 0.0)
        tokens_child_corr[t].update(sum(accs) / max(len(accs), 1), 1)


@torch.no_grad()
def evaluate(model, loader, ctx, temperature, top_k, 
             results=None, mode="test", graph_adjacency_list_cache=None):
    """Evaluates generation accuracy without teacher forcing.

    Args:
        model: Input parameter.
        loader: Input parameter.
        ctx: Input parameter.
        temperature: Input parameter.
        top_k: Input parameter.
        results: Input parameter.
        mode: Input parameter.
        graph_adjacency_list_cache: Input parameter.

    Returns:
        object: Function return value.
    """
    num_prefix_tokens = loader.dataset.num_prefix_tokens
    num_target_tokens = loader.dataset.num_target_tokens

    loader.dataset.eval()
    model.eval()

    total_acc = AverageMeter()
    tokens_corr = {i: AverageMeter() for i in range(num_target_tokens)}
    tokens_child_corr = None
    if graph_adjacency_list_cache is not None:
        tokens_child_corr = {i: AverageMeter() for i in range(1, num_target_tokens)}

    bar = tqdm(loader)
    example_idx = 0
    x, y, y_pred = None, None, None

    for x in bar:
        y = x[:, num_prefix_tokens:].clone()
        x = x[:, :num_prefix_tokens].clone()

        with ctx:
            y_pred = model.generate(x, num_target_tokens, temperature=temperature, top_k=top_k)

        correct = y.eq(y_pred[:, -num_target_tokens:]).float()
        completely_correct = torch.mean(correct.sum(dim=1).eq(num_target_tokens).to(torch.float))
        total_acc.update(completely_correct.item(), x.shape[0])

        per_token_acc = correct.mean(dim=0)
        for i in range(num_target_tokens):
            tokens_corr[i].update(per_token_acc[i].item(), x.shape[0])

        bar.set_description(f"[{mode}] Acc: {total_acc.get(percentage=True):.2f}")

        if graph_adjacency_list_cache is not None:
            _update_child_transition_accuracy(
                tokens_child_corr=tokens_child_corr,
                y_pred=y_pred,
                num_target_tokens=num_target_tokens,
                adj_list_cache=graph_adjacency_list_cache,
                example_idx=example_idx,
                batch_size=x.shape[0],
            )

        example_idx += x.shape[0]

    loader.dataset.train()
    model.train()

    if results is not None:
        results[f"{mode}/accuracy"] = total_acc.get(percentage=True)
        for i in range(num_target_tokens):
            results[f"{mode}/token_{i+1}"] = tokens_corr[i].get(percentage=True)
        if graph_adjacency_list_cache is not None:
            for t in range(1, num_target_tokens):
                key = f"{mode}_child/token_{t+1}"
                results[key] = tokens_child_corr[t].get(percentage=True)

        if x is not None and y is not None and y_pred is not None:
            results["prediction_txt"] = {
                "prefix": x[:20].cpu().tolist(),
                "target": y[:20].cpu().tolist(),
                "pred": y_pred[:20].cpu().tolist(),
            }

    return results


@torch.no_grad()
def evaluate_forced(model, loader, ctx, 
                    results=None, mode="test", graph_adjacency_list_cache=None):
    """Evaluates teacher-forced accuracy and loss.

    Args:
        model: Input parameter.
        loader: Input parameter.
        ctx: Input parameter.
        results: Input parameter.
        mode: Input parameter.
        graph_adjacency_list_cache: Input parameter.

    Returns:
        object: Function return value.
    """
    num_target_tokens = loader.dataset.num_target_tokens
    total_acc, total_loss = AverageMeter(), AverageMeter()
    tokens_corr = {i: AverageMeter() for i in range(num_target_tokens)}
    tokens_child_corr = None
    if graph_adjacency_list_cache is not None:
        tokens_child_corr = {i: AverageMeter() for i in range(1, num_target_tokens)}

    bar = tqdm(loader)
    example_idx = 0

    for x, y in bar:
        with ctx:
            logits, loss, accs = model(x, y)

        y_pred = torch.argmax(logits, dim=-1)
        total_acc.update(val=accs["acc"].item(), num=x.shape[0])
        total_loss.update(val=loss.item(), num=x.shape[0])

        for i in range(num_target_tokens):
            tokens_corr[i].update(accs["token_acc"][i].item(), x.shape[0])

        if graph_adjacency_list_cache is not None:
            for t in range(1, num_target_tokens):
                parents = y_pred[:, t - 1].tolist()
                children = y_pred[:, t].tolist()

                accs_per_batch = []
                for i in range(x.shape[0]):
                    adj = graph_adjacency_list_cache[example_idx + i]
                    is_child = children[i] in adj.get(parents[i], set())
                    accs_per_batch.append(1.0 if is_child else 0.0)
                tokens_child_corr[t].update(sum(accs_per_batch) / len(accs_per_batch), 1)

        example_idx += x.shape[0]
        bar.set_description(
            "Forced Loss: {:.4f} Forced Acc: {:.2f}".format(
                total_loss.get(), total_acc.get(percentage=True)
            )
        )

    if results is not None:
        results[f"{mode}/forced loss"] = total_loss.get()
        results[f"{mode}/forced accuracy"] = total_acc.get(percentage=True)
        for i in range(num_target_tokens):
            key = f"{mode}/forced_token_{i+1}"
            results[key] = tokens_corr[i].get(percentage=True)
        if graph_adjacency_list_cache is not None:
            for t in range(1, num_target_tokens):
                key = f"{mode}_forced_child/token_{t+1}"
                results[key] = tokens_child_corr[t].get(percentage=True)

    return results
