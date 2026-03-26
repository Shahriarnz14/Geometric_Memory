"""Non-evaluation in-weights analysis and probing utilities."""

from geometric_memory.tokenizing.numeral_tokenizer import NumeralTokenizer
import torch

PAUSE_TOKEN = NumeralTokenizer.PAUSE_TOKEN


####################################
# Save Node Embeddings Utility
####################################


def get_node_embeddings(model, num_nodes=None):
    """Extract node embeddings from the model.

    Args:
        model: Input parameter.
        num_nodes: Input parameter.

    Returns:
        object: Function return value.
    """

    with torch.no_grad():
        # [vocab_size, d_model] → take only node rows
        node_embs = model.embed_tokens.weight[:num_nodes]  # [num_nodes, d_model]
        node_embs_np = node_embs.detach().cpu().numpy()

    return node_embs_np


####################################
# Get Top-K Predicted Tokens Utility
####################################
def build_edge_prefix_strings_for_all_nodes(args):
    """Build edge prefix strings for all nodes.
    
    Args:
        args: Input parameter.
    
    Returns:
        object: Function return value.
    """
    prefixes = []
    for u in range(args.total_nodes):
        s = ""
        if args.include_task_token_in_prefix:
            s += "[EDGE],"
        s += f"{u}"
        if not args.edge_memorization_drop_pause_token:
            s += f",{PAUSE_TOKEN}"
        prefixes.append(s)
    return prefixes


def build_batched_prefix_tensor(tokenizer, device, prefix_strings):
    """Build batched prefix tensor.
    
    Args:
        tokenizer: Input parameter.
        device: Input parameter.
        prefix_strings: Input parameter.
    
    Returns:
        object: Function return value.
    """
    encoded = [tokenizer.encode(s) for s in prefix_strings]
    max_len = max(len(e) for e in encoded)

    x = torch.full(
        (len(encoded), max_len),
        fill_value=tokenizer.pad_token_id,
        dtype=torch.long,
        device=device,
    )

    for i, e in enumerate(encoded):
        x[i, : len(e)] = torch.tensor(e, dtype=torch.long, device=device)

    return x


@torch.no_grad()
def get_top_k_predictions_for_all_nodes(model, x_topk, k=5):
    """Returns a [num_nodes, k] array of top-k predictions for v,

    Args:
        model: Input parameter.
        x_topk: Input parameter.
        k: Input parameter.

    Returns:
        object: Function return value.
    """

    model_was_training = model.training
    model.eval()

    logits, _, _ = model(x_topk, None)  # y is unused → pass None
    logits_v = logits[:, -1, :]  # prediction of v at last token

    topk_idx = torch.topk(logits_v, k=min(k, logits_v.size(-1)), dim=-1).indices

    if model_was_training:
        model.train()

    return topk_idx.cpu().numpy()
