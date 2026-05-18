from dataclasses import dataclass

import torch


@dataclass
class PhiConfig:
    """PhiConfig definition.
    
    Description:
        Encapsulates related behavior and state.
    """
    name: str = "phi_2"
    block_size: int = 2048
    vocab_size: int = 51200  #
    n_layers: int = 32
    n_heads: int = 32
    n_embd: int = 2560
    dropout: float = 0.0
    bias: bool = (
        True  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    )
    use_flash: bool = True
    cache: bool = False
    base: int = 10000
    rope_dim: int = int(0.4 * 2560 // 32)  # n_heads
    initializer_range: float = 0.02
    max_bsz: int = 16
    resid_drop: float = 0.1
    dtype = torch.bfloat16
    # PhiConfig = PhiConfig()
    # Phi1_5Config = PhiConfig(
    #     name='phi_1_5',
    #     n_embd=2048,
    #     n_layers=24,
    #     rope_dim=int(0.5 * 2048 // 32)
    # )


@dataclass
class GPTConfig:
    """GPTConfig definition.
    
    Description:
        Encapsulates related behavior and state.
    """
    block_size: int = 1024
    vocab_size: int = (
        50304  # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    )
    n_layers: int = 12
    n_heads: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = (
        True  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    )
    use_flash: bool = True if torch.cuda.is_available() else False
    teacherless_token: int = None
    dtype = torch.bfloat16
    cache: bool = False
    max_bsz: int = 16
    use_attention: bool = True
    use_residual: bool = True
    use_layernorm: bool = True
    use_positional_encoding: bool = True
    use_neural_net_mlp: bool = False
    freeze_embeddings: bool = False
    use_weight_tying: bool = True
    weight_init_mode: str = "default"


@dataclass
class PythiaConfig:
    """PythiaConfig definition.
    
    Description:
        Encapsulates related behavior and state.
    """
    block_size: int = 1024
    vocab_size: int = (
        50304  # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    )
    n_layers: int = 12
    n_heads: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = (
        True  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    )
    use_flash: bool = True if torch.cuda.is_available() else False
    teacherless_token: int = None
    dtype = torch.bfloat16
    cache: bool = False
    max_bsz: int = 16
    base: int = 10000
    rope_dim: int = int(0.25 * n_embd // n_heads)


@dataclass
class MambaConfig:
    """MambaConfig definition.
    
    Description:
        Encapsulates related behavior and state.
    """
    n_embd: int = 2560
    n_layers: int = 64
    vocab_size: int = 50277
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    block_size: int = 1024
    teacherless_token: int = None
    bias: bool = True
    cache: bool = True
    use_layernorm: bool = False
    use_positional_encoding: bool = False
