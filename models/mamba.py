from mamba_ssm import Mamba
from geometric_memory.models.base_model import DeepSequenceModel
from geometric_memory.models.config import MambaConfig
import torch.nn as nn


class MambaBlock(nn.Module):
    """MambaBlock definition.
    
    Description:
        Encapsulates related behavior and state.
    """
    def __init__(self, config, layer_idx=None):
        """  init  .
        
        Args:
            config: Input parameter.
            layer_idx: Input parameter.
        
        Returns:
            object: Function return value.
        """
        super().__init__()
        self.mamba_layer = Mamba(
            d_model=config.n_embd,
            d_state=config.d_state,
            d_conv=config.d_conv,
            expand=config.expand,
        )
        self.norm = nn.LayerNorm(config.n_embd)

    def forward(self, x, cache=None, attn_mask=None):
        """Forward.
        
        Args:
            x: Input parameter.
            cache: Input parameter.
            attn_mask: Input parameter.
        
        Returns:
            object: Function return value.
        """
        return self.mamba_layer(self.norm(x))


class MambaModel(DeepSequenceModel):
    """MambaModel definition.
    
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
        config.cache = False

        super().__init__(config, block=MambaBlock)

        # Tie embedding and output layer weights
        self.embed_tokens.weight = self.lm_head.weight
        self._print_parameter_report()

    def crop_block_size(self, block_size):
        """Crop block size.
        
        Args:
            block_size: Input parameter.
        
        Returns:
            object: Function return value.
        """
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
