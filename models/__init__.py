import os
from geometric_memory.models.config import GPTConfig, MambaConfig
from geometric_memory.models.gpt import GPT
from geometric_memory.models.pythia import Pythia


def _build_gpt_model(args):
    """ build gpt model.
    
    Args:
        args: Input parameter.
    
    Returns:
        object: Function return value.
    """
    config = GPTConfig(
        n_layers=args.transformer_layer_count,
        n_heads=args.attention_head_count,
        n_embd=args.embedding_dimension,
        block_size=args.block_size,
        bias=True,
        vocab_size=args.vocab_size,
        dropout=args.dropout_rate,
        use_flash=args.use_flash,
        teacherless_token=args.teacherless_token,
        use_attention=args.use_attention,
        use_residual=args.use_residual_connections,
        freeze_embeddings=args.freeze_token_embeddings,
        use_layernorm=args.use_layer_norm,
        use_positional_encoding=args.use_positional_encoding,
        use_neural_net_mlp=args.use_mlp_only_blocks,
        use_weight_tying=args.tie_input_output_embeddings,
    )
    return GPT(config)


def _build_gpt2_model(args):
    """ build gpt2 model.
    
    Args:
        args: Input parameter.
    
    Returns:
        object: Function return value.
    """
    model_name_in_path = f"openai-community/{args.model_family}"
    model_path = os.path.join(args.ws_gcs_path, "hf_models", model_name_in_path)

    model = GPT.from_pretrained(
        model_path,
        config_model_type=args.model_family,
        teacherless_token=args.teacherless_token,
    )
    if args.block_size < 1024:
        model.crop_block_size(args.block_size)
    return model


def _build_pythia_model(args):
    """ build pythia model.
    
    Args:
        args: Input parameter.
    
    Returns:
        object: Function return value.
    """
    return Pythia.from_pretrained(args.model_family, teacherless_token=args.teacherless_token)


def _build_mamba_model(args):

    """ build mamba model.
    
    Args:
        args: Input parameter.
    
    Returns:
        object: Function return value.
    """
    from geometric_memory.models.mamba import MambaModel

    config = MambaConfig(
        n_layers=args.transformer_layer_count,
        n_embd=args.embedding_dimension,
        vocab_size=args.vocab_size,
        block_size=args.block_size,
        teacherless_token=args.teacherless_token,
        d_state=args.mamba_state_dimension,
        d_conv=args.mamba_convolution_kernel,
        expand=args.mamba_expand_factor,
    )
    return MambaModel(config)


def get_model(args):
    """Get model.
    
    Args:
        args: Input parameter.
    
    Returns:
        object: Function return value.
    """
    if args.model_family == "gpt":
        return _build_gpt_model(args)
    if args.model_family.startswith("gpt2"):
        return _build_gpt2_model(args)
    if args.model_family.startswith("pythia"):
        return _build_pythia_model(args)
    elif args.model_family == "mamba":
        return _build_mamba_model(args)
    raise ValueError(f"Unknown model family: {args.model_family}")
