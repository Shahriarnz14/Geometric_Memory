"""Reproducibility helpers for tiny-graph notebooks."""

from __future__ import annotations

import random

import numpy as np
import torch


def set_all_seeds(seed: int) -> None:
    """Sets deterministic seeds for Python, NumPy, and PyTorch.

    Args:
        seed: Integer random seed.

    Returns:
        None: Global random states are updated in-place.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
