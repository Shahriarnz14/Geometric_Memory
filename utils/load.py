import torch


def remap(key, mapping):
    """Maps a key through a dictionary when a mapping entry exists.

    Args:
        key: Original key string.
        mapping: Dictionary of key-part substitutions.

    Returns:
        str: Remapped key if present, else the original key.
    """
    return mapping[key] if key in mapping else key


def load_gpt(state_dict, hf_state_dict):
    """Loads GPT-style Hugging Face weights into internal GPT state dict.

    Args:
        state_dict: Target model state dict to populate in-place.
        hf_state_dict: Source Hugging Face state dict.

    Returns:
        dict: Updated target state dict.
    """
    _mapping = {
        'wte': 'embed_tokens',
        'wpe': 'pos_encoding',
        'h': 'layers',
        'ln_f': 'final_layernorm',
        'self_attn': 'attn',
        'c_proj': 'proj',
        'c_fc': 'expand',
    }

    check_keys_loaded = {key: False for key in state_dict}
    for key, val in hf_state_dict.items():
        _mapped_key = key
        if key.startswith('transformer.'):
            _mapped_key = key.split('transformer.')[1]
        mapped_key = '.'.join(remap(s, _mapping) for s in _mapped_key.split('.'))

        # fused qkv in GPT-style "c_attn"
        if mapped_key in state_dict or mapped_key.endswith('c_attn.weight') or mapped_key.endswith('c_attn.bias'):
            if mapped_key.endswith('c_attn.weight') or mapped_key.endswith('c_attn.bias'):
                dim = hf_state_dict[key].shape[-1] // 3
                with torch.no_grad():
                    state_dict['queries_linear'.join(mapped_key.split('c_attn'))]\
                        .copy_(hf_state_dict[key][..., :dim].t())
                    check_keys_loaded['queries_linear'.join(mapped_key.split('c_attn'))] = True
                    state_dict['keys_linear'.join(mapped_key.split('c_attn'))]\
                        .copy_(hf_state_dict[key][..., dim:2 * dim].t())
                    check_keys_loaded['keys_linear'.join(mapped_key.split('c_attn'))] = True
                    state_dict['values_linear'.join(mapped_key.split('c_attn'))]\
                        .copy_(hf_state_dict[key][..., 2 * dim:3 * dim].t())
                    check_keys_loaded['values_linear'.join(mapped_key.split('c_attn'))] = True
            else:
                try:
                    # weight-transpose exceptions (match your screenshots)
                    if mapped_key.endswith('mlp.expand.weight') or mapped_key.endswith('proj.weight'):
                        with torch.no_grad():
                            state_dict[mapped_key].copy_(hf_state_dict[key].t())
                            check_keys_loaded[mapped_key] = True
                    else:
                        with torch.no_grad():
                            state_dict[mapped_key].copy_(hf_state_dict[key])
                            check_keys_loaded[mapped_key] = True
                except RuntimeError:
                    print(key, 'does not match in shape')
        else:# KeyError:
            print(key, 'was not found')

    for k, v in check_keys_loaded.items():
        if not v:
            print(k, 'was not loaded')

    return state_dict


def load_pythia(state_dict, hf_state_dict, config):
    """Loads Pythia Hugging Face weights into internal model state dict.

    Args:
        state_dict: Target model state dict to populate in-place.
        hf_state_dict: Source Hugging Face state dict.
        config: Pythia config object used for head-dimension math.

    Returns:
        dict: Updated target state dict.
    """
    _mapping = {
        'embed_in': 'embed_tokens',
        'embed_out': 'lm_head',
        'input_layernorm': 'ln1',
        'post_attention_layernorm': 'ln2',
        'final_layernorm': 'final_layernorm',
        'attention': 'attn',
        'dense_h_to_4h': 'expand',
        'dense_4h_to_h': 'proj',
    }

    check_keys_loaded = {key: False for key in state_dict}
    check_keys_hf_loaded = {key: False for key in hf_state_dict}

    for key, val in hf_state_dict.items():
        _mapped_key = key
        if key.startswith('gpt_neox.'):
            _mapped_key = key.split('gpt_neox.')[1]
        mapped_key = '.'.join(remap(s, _mapping) for s in _mapped_key.split('.'))

        # Pythia fused qkv: "query_key_value.{weight,bias}"
        if mapped_key.endswith('query_key_value.weight') or mapped_key.endswith('query_key_value.bias'):
            with torch.no_grad():
                head_dim = config.n_embd // config.n_heads

                base = mapped_key.rsplit('query_key_value', 1)[0] + 'attn.'
                tensor = hf_state_dict[key]

                if mapped_key.endswith('.weight'):
                    # HF: (in_dim, 3*out_dim). Split along last dim, transpose to (out_dim, in_dim)
                    dim = tensor.shape[-1] // 3
                    q_w = tensor[:, :dim].t()
                    k_w = tensor[:, dim:2 * dim].t()
                    v_w = tensor[:, 2 * dim:3 * dim].t()

                    state_dict[base + 'queries_linear.weight'].copy_(q_w)
                    check_keys_loaded[base + 'queries_linear.weight'] = True
                    check_keys_hf_loaded[key] = True

                    state_dict[base + 'keys_linear.weight'].copy_(k_w)
                    check_keys_loaded[base + 'keys_linear.weight'] = True

                    state_dict[base + 'values_linear.weight'].copy_(v_w)
                    check_keys_loaded[base + 'values_linear.weight'] = True
                else:
                    dim = tensor.shape[-1] // 3
                    q_b = tensor[:dim]
                    k_b = tensor[dim:2 * dim]
                    v_b = tensor[2 * dim:3 * dim]

                    state_dict[base + 'queries_linear.bias'].copy_(q_b)
                    check_keys_loaded[base + 'queries_linear.bias'] = True
                    check_keys_hf_loaded[key] = True

                    state_dict[base + 'keys_linear.bias'].copy_(k_b)
                    check_keys_loaded[base + 'keys_linear.bias'] = True

                    state_dict[base + 'values_linear.bias'].copy_(v_b)
                    check_keys_loaded[base + 'values_linear.bias'] = True
        else:
            try:
                with torch.no_grad():
                    # Transpose for standard Linear weights that come as (in,out) from HF
                    if mapped_key.endswith('.weight'):
                        state_dict[mapped_key].copy_(hf_state_dict[key].t())
                    else:
                        state_dict[mapped_key].copy_(hf_state_dict[key])
                    check_keys_loaded[mapped_key] = True
                    check_keys_hf_loaded[key] = True
            except RuntimeError:
                print(key, 'does not match in shape')
            except KeyError:
                print(key, 'was not found')

    for k, v in check_keys_loaded.items():
        if not v:
            print(k, 'was not loaded')
    for k, v in check_keys_hf_loaded.items():
        if not v:
            print(k, 'was not loaded')

    return state_dict


def load_mamba(state_dict, hf_state_dict):
    """Loads Mamba-style checkpoints into internal Mamba state dict.

    Args:
        state_dict: Target model state dict to populate in-place.
        hf_state_dict: Source checkpoint state dict.

    Returns:
        dict: Updated target state dict.
    """
    _mapping = {
        "embedding": "embed_tokens",
        "mixer": "mamba_layer",
        "norm_f": "final_layernorm",
    }
    prefixes_to_strip = ("backbone.", "model.")

    check_keys_loaded = {key: False for key in state_dict}
    check_keys_hf_loaded = {key: False for key in hf_state_dict}

    for key, val in hf_state_dict.items():
        mapped_key = key
        for prefix in prefixes_to_strip:
            if mapped_key.startswith(prefix):
                mapped_key = mapped_key[len(prefix) :]
                break
        mapped_key = ".".join(remap(part, _mapping) for part in mapped_key.split("."))

        if mapped_key not in state_dict:
            print(key, "was not found")
            continue

        target_tensor = state_dict[mapped_key]
        source_tensor = val

        try:
            with torch.no_grad():
                if target_tensor.shape == source_tensor.shape:
                    target_tensor.copy_(source_tensor)
                elif target_tensor.ndim == 2 and target_tensor.shape == source_tensor.t().shape:
                    target_tensor.copy_(source_tensor.t())
                else:
                    print(key, "does not match in shape")
                    continue
            check_keys_loaded[mapped_key] = True
            check_keys_hf_loaded[key] = True
        except RuntimeError:
            print(key, "does not match in shape")

    for loaded_key, was_loaded in check_keys_loaded.items():
        if not was_loaded:
            print(loaded_key, "was not loaded")
    for hf_key, was_loaded in check_keys_hf_loaded.items():
        if not was_loaded:
            print(hf_key, "was not loaded")

    return state_dict
