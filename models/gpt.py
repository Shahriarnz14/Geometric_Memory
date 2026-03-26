"""***Adapted from Andrej Karpathy's nanoGPT***"""

import os
from geometric_memory.models.base_model import DeepSequenceModel
from geometric_memory.models.config import GPTConfig
from geometric_memory.models.lib import Attention, LayerNorm, MLP, MLP_NeuralNet
from geometric_memory.utils.load import load_gpt
import torch
import torch.nn as nn


class Block(nn.Module):
    """Block definition.
    
    Description:
        Encapsulates related behavior and state.
    """
    def __init__(self, config, layer_idx):
        """  init  .
        
        Args:
            config: Input parameter.
            layer_idx: Input parameter.
        
        Returns:
            object: Function return value.
        """
        super().__init__()
        self.use_attention = config.use_attention
        self.use_residual = config.use_residual
        self.use_layernorm = config.use_layernorm
        self.use_neural_net_mlp = config.use_neural_net_mlp

        if self.use_layernorm:
            self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)

        if self.use_attention:
            self.attn = Attention(config, layer_idx, rotary=False)
            if self.use_layernorm:  # only create when attention is calculated
                self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)

        if config.use_neural_net_mlp:
            self.mlp = MLP_NeuralNet(config)
        else:
            self.mlp = MLP(config)

    def forward(self, x, cache=None, attn_mask=None):
        """Forward.
        
        Args:
            x: Input parameter.
            cache: Input parameter.
            attn_mask: Input parameter.
        
        Returns:
            object: Function return value.
        """
        if self.use_layernorm:
            x_ln = self.ln_1(x)
        else:
            x_ln = x

        if self.use_attention:
            attn_out = self.attn(x_ln, cache, attn_mask=attn_mask)
            if self.use_residual:
                x = x + attn_out
            else:
                x = attn_out

            if self.use_layernorm:
                x_ln = self.ln_2(x)
            else:
                x_ln = x

        if self.use_residual:
            return x + self.mlp(x_ln)
        else:
            return self.mlp(x_ln)

    # def forward(self, x, cache=None, attn_mask=None):
    #     if self.use_attention:
    #         if self.use_residual:
    #             x = x + self.attn(self.ln_1(x), cache, attn_mask=attn_mask)
    #             x = x + self.mlp(self.ln_2(x))
    #         else:
    #             x = self.attn(self.ln_1(x), cache, attn_mask=attn_mask)
    #             x = self.mlp(self.ln_2(x))
    #         return x

    #     elif self.use_residual:
    #         return x + self.mlp(self.ln_1(x))
    #     else:
    #         return self.mlp(self.ln_1(x))


class GPT(DeepSequenceModel):
    """GPT definition.
    
    Description:
        Encapsulates related behavior and state.
    """
    def __init__(self, config):
        """  init  .
        
        Args:
            config: Input parameter.
        
        Returns:
            object: Function return value.
        """
        super().__init__(config, block=Block)

        self.use_attention = config.use_attention
        self.freeze_embeddings = config.freeze_embeddings
        self.use_residual = config.use_residual
        self.use_layernorm = config.use_layernorm
        self.use_positional_encoding = config.use_positional_encoding
        self.use_neural_net_mlp = config.use_neural_net_mlp

        # Add positional encoding
        if self.use_positional_encoding:
            self.pos_encoding = nn.Embedding(config.block_size, config.n_embd)

        # Tie weights as in the GPT paper
        if config.use_weight_tying:
            self.embed_tokens.weight = self.lm_head.weight

        if self.freeze_embeddings:
            self.embed_tokens.weight.requires_grad = False
            self.lm_head.weight.requires_grad = False
            if self.use_positional_encoding:
                self.pos_encoding.weight.requires_grad = False

        self._print_parameter_report()

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        """Crop block size.
        
        Args:
            block_size: Input parameter.
        
        Returns:
            object: Function return value.
        """
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        if self.use_positional_encoding:
            self.pos_encoding.weight = nn.Parameter(self.pos_encoding.weight[:block_size])
        for block in self.layers:
            if hasattr(block.attn, "bias"):
                block.attn.bias = block.attn.bias[:, :, :block_size, :block_size]

    @classmethod
    def from_pretrained(cls, model_type, config_model_type=None, teacherless_token=None):
        """Loads GPT-2 model weights from a local path or downloads from Hugging Face.

        Args:
            model_type: Input parameter.
            config_model_type: Input parameter.
            teacherless_token: Input parameter.

        Returns:
            object: Function return value.
        """

        # --- Start of new logic ---
        # If the provided input is a dir, we need to know which configuration to use
        if os.path.isdir(model_type):
            # When loading from a local path, the config type must be explicitly given
            if config_model_type is None:
                raise ValueError(
                    "When loading from a path, 'config_model_type' must be specified"
                    " (e.g., 'gpt2-large')."
                )

            model_config_type = config_model_type
            # The Hugging Face model will be loaded from the local path.
            hf_model_source = model_type
            print(f"Loading weights from local path: {hf_model_source}")
        else:
            # If it's not a path, we assume it's a model name (original behavior).
            model_config_type = model_type
            hf_model_source = model_type
            print(f"Loading weights from pretrained gpt: {hf_model_source}")
        # --- End of new logic ---

        # only dropout can be overridden see more notes below
        from transformers import GPT2LMHeadModel

        assert model_config_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}

        # n_layer, n_head and n_embd are now determined from model_config_type
        config_args = {
            "gpt2": dict(n_layers=12, n_heads=12, n_embd=768),  # 124M params
            "gpt2-medium": dict(
                n_layers=24,
                n_heads=16,
                n_embd=1024
                # 350M params
            ),
            "gpt2-large": dict(n_layers=36, n_heads=20, n_embd=1280),  # 774M params
            "gpt2-xl": dict(n_layers=48, n_heads=25, n_embd=1600),  # 1558M params
        }[model_config_type]

        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args["vocab_size"] = 50257
        config_args["block_size"] = 1024  # always 1024 for GPT model checkpoints
        config_args["bias"] = True  # always True for GPT model checkpoints
        config_args["teacherless_token"] = teacherless_token

        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()

        # init a transformers model from either the local path or the Hub
        model_hf = GPT2LMHeadModel.from_pretrained(hf_model_source)
        sd_hf = model_hf.state_dict()

        # Match the two checkpoints
        sd = load_gpt(sd, sd_hf)
        model.load_state_dict(sd, strict=True)

        return model

    @classmethod
    def from_pretrained_old(cls, model_type, teacherless_token=None):
        """From pretrained old.
        
        Args:
            model_type: Input parameter.
            teacherless_token: Input parameter.
        
        Returns:
            object: Function return value.
        """
        assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}
        # only dropout can be overridden see more notes below
        from transformers import GPT2LMHeadModel

        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            "gpt2": dict(n_layers=12, n_heads=12, n_embd=768),  # 124M params
            "gpt2-medium": dict(
                n_layers=24,
                n_heads=16,
                n_embd=1024
                # 350M params
            ),
            "gpt2-large": dict(n_layers=36, n_heads=20, n_embd=1280),  # 774M params
            "gpt2-xl": dict(n_layers=48, n_heads=25, n_embd=1600),  # 1558M params
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args["vocab_size"] = 50257
        config_args["block_size"] = 1024  # always 1024 for GPT model checkpoints
        config_args["bias"] = True  # always True for GPT model checkpoints
        config_args["teacherless_token"] = teacherless_token

        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # Match the two checkpoints
        sd = load_gpt(sd, sd_hf)
        model.load_state_dict(sd, strict=True)

        return model


if __name__ == "__main__":
    import types
    from tokenizers import get_tokenizer

    args = types.SimpleNamespace()
    args.model = "gpt2"
    tokenizer = get_tokenizer(args)

    model = GPT.from_pretrained(model_type="gpt2")
    model.eval()
    text = "Hello my name is"
    idx = torch.tensor(tokenizer.encode(text), dtype=torch.int32).unsqueeze(0)
    # model.set_cache(device='cpu')
    out = model.generate(idx, max_new_tokens=24, top_k=1)
    print(tokenizer.decode(out.numpy().squeeze()))
