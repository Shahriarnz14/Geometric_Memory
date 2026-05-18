import math

from geometric_memory.models.cache import Cache
from geometric_memory.utils.training_utils import accuracy
import torch
import torch.nn as nn
from torch.nn import functional as F


class DeepSequenceModel(nn.Module):
    """DeepSequenceModel definition.
    
    Description:
        Encapsulates related behavior and state.
    """
    def __init__(self, config, block):
        """  init  .
        
        Args:
            config: Input parameter.
            block: Input parameter.
        
        Returns:
            object: Function return value.
        """
        super().__init__()
        self.config = config

        self.vocab_size = config.vocab_size
        self.use_layernorm = config.use_layernorm

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head._geometric_memory_weight_init = (
            "normal_unit"
            if getattr(config, "weight_init_mode", "default") == "non_geometric"
            else "default"
        )
        self.embed_tokens = nn.Embedding(config.vocab_size, config.n_embd)

        self.use_positional_encoding = config.use_positional_encoding
        if self.use_positional_encoding:
            # Positional encoding has to be overwritten in __init__ of subclass
            self.pos_encoding = lambda x: 0

        self.layers = nn.ModuleList(
            [block(config, layer_idx) for layer_idx in range(config.n_layers)]
        )
        if self.use_layernorm:
            self.final_layernorm = nn.LayerNorm(config.n_embd)

        if config.cache:
            # Instantiated but not occupying memory yet
            self.cache = Cache(config)
        else:
            self.cache = None

        # Initialize weights
        self.apply(self._init_weights)

        # apply special scaled init to the residual projections, per GPT-2 paper
        layer_count = getattr(config, "n_layers", None)
        if layer_count is None:
            layer_count = getattr(config, "n_layer", None)
        if layer_count is None:
            raise AttributeError("Model config must define `n_layers`.")

        for pn, p in self.named_parameters():
            if pn.endswith("mlp_projection.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * layer_count))

    def get_num_params(self, non_embedding=True):
        """Return the number of parameters in the model.

        Args:
            non_embedding: Input parameter.

        Returns:
            object: Function return value.
        """
        all_params = sum(p.numel() for p in self.parameters())
        non_emb_params = all_params

        if non_embedding:
            # Remove token embeddings only when they are not tied to lm_head.
            if self.embed_tokens.weight is not self.lm_head.weight:
                non_emb_params -= self.embed_tokens.weight.numel()

            # Remove positional embeddings when present.
            pos_encoding = getattr(self, "pos_encoding", None)
            if isinstance(pos_encoding, nn.Embedding):
                non_emb_params -= pos_encoding.weight.numel()

        return all_params, non_emb_params

    def _print_parameter_report(self):
        """Prints parameter totals after subclass-specific setup is complete.

        Args:
            None: This callable does not take external parameters.

        Returns:
            object: Function return value.
        """
        all_params, non_emb_params = self.get_num_params()
        print(
            f"Number of parameters: %.4fm" % (all_params / 1e6,),
            f" Number of non-embedding parameters: %.4fm" % (non_emb_params / 1e6,),
        )

    def _init_weights(self, module):
        """ init weights.
        
        Args:
            module: Input parameter.
        
        Returns:
            object: Function return value.
        """
        if isinstance(module, nn.Linear):
            weight_init = getattr(module, "_geometric_memory_weight_init", "default")
            if weight_init == "normal_unit": # when we set weight_init_mode to non_geometric for lm_head
                torch.nn.init.normal_(module.weight, mean=0.0, std=1.0)
            else: #default
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding): 
            # this will be overwritten without for token embeddings when we have weight-tying
            # positional embeddings do not use this init and are always PyTorch's default N(0,1)
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, pad_token_id=None):
        """Forward.
        
        Args:
            idx: Input parameter.
            targets: Input parameter.
            pad_token_id: Input parameter.
        
        Returns:
            object: Function return value.
        """
        device = idx.device
        bsz, seq_len = idx.size()
        assert seq_len <= self.config.block_size, (
            f"Cannot forward sequence of length {seq_len}, block size is only "
            f"{self.config.block_size}"
        )

        # --- Create Attention Mask ---
        attn_mask = None
        if pad_token_id is not None:
            attn_mask = idx != pad_token_id

        tok_emb = self.embed_tokens(idx)
        start_pos = (
            0 if self.cache is None or not self.cache.use_caching else self.cache.cur_seq_len[0]
        )
        pos = torch.arange(
            start_pos, seq_len + start_pos, dtype=torch.long, device=device
        ).unsqueeze(0)

        if self.use_positional_encoding:
            pos_emb = self.pos_encoding(pos)
            x = tok_emb + pos_emb
        else:
            x = tok_emb

        for block in self.layers:
            x = block(x, self.cache, attn_mask=attn_mask)

        if self.use_layernorm:
            x = self.final_layernorm(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            # Calculate loss with ignore_index=-1, meaning we skip the gradient contributions from those tokens
            # which is basically the prefix tokens
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
            acc, token_acc = accuracy(logits, targets)
            accs = {"acc": acc, "token_acc": token_acc}
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :])  # note: using list [-1] to preserve the time dim
            loss, accs = None, None

        return logits, loss, accs

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Generate.
        
        Args:
            idx: Input parameter.
            max_new_tokens: Input parameter.
            temperature: Input parameter.
            top_k: Input parameter.
        
        Returns:
            object: Function return value.
        """
        bsz, prefix_len = idx.shape
        seq_len = prefix_len + max_new_tokens - 1
        device = idx.device

        # Decode in parallel if teacherless
        if self.config.teacherless_token is not None:
            idx_next = (
                torch.tensor(self.config.teacherless_token)
                * torch.ones((bsz, max_new_tokens - 1)).long()
            )

            idx_next = idx_next.to(device)
            idx = torch.cat((idx, idx_next), dim=1)
            # if the sequence context is growing too long we must crop it at block size
            idx_cond = (
                idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size :]
            )
            # forward the model to get the logits for the index in the sequence
            logits, _, _ = self(idx_cond, targets=idx_cond)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, :, [-1]]] = -float("inf")

            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            out = torch.multinomial(probs.reshape(bsz * seq_len, -1), num_samples=1).reshape(
                bsz, seq_len
            )

            return out

        out = idx.clone()
        idx_next = idx.clone()

        for i in range(max_new_tokens):
            if self.cache is not None and self.cache.use_caching:
                # if we're caching, only propagate the last token
                idx = idx_next
            # if the sequence context is growing too long we must crop it at block size
            idx_cond = (
                idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size :]
            )
            # forward the model to get the logits for the index in the sequence
            logits, _, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)

            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)

            # append sampled index to the running sequence and continue
            idx = torch.cat((out, idx_next), dim=1)
            out = idx.clone()

        return out

    def set_cache(self, device=None, mode=True):
        """Activates caching.

        Args:
            device: Input parameter.
            mode: Input parameter.

        Returns:
            object: Function return value.
        """
        self.cache.use_caching = mode
        if mode and self.cache.key_cache is None:
            # Allocate memory for caching
            self.cache.build(device)

    def empty_cache(self):
        """Free memory by removing cache.

        Args:
            None: This callable does not take external parameters.

        Returns:
            object: Function return value.
        """
        self.set_cache(mode=False)
        self.cache.delete()

    def reset_cache(self):
        """Set cache back to zero entries

        Args:
            None: This callable does not take external parameters.

        Returns:
            object: Function return value.
        """
        self.cache.empty()
