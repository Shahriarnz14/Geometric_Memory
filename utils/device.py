"""Device selection helpers for training and notebook runtimes."""

from __future__ import annotations

import torch


def resolve_default_device() -> str:
    """Resolves the preferred runtime device in fallback order.

    Args:
        None: This helper takes no external parameters.

    Returns:
        str: ``"cuda"`` when available, otherwise ``"mps"`` when available,
            otherwise ``"cpu"``.
    """
    if torch.cuda.is_available():
        return "cuda"
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_built() and mps_backend.is_available():
        return "mps"
    return "cpu"


def is_cuda_device(device: str) -> bool:
    """Checks whether a runtime device string points to CUDA.

    Args:
        device: Runtime device string.

    Returns:
        bool: ``True`` when the device is CUDA-like (``"cuda"`` or ``"cuda:*"``).
    """
    return str(device).lower().startswith("cuda")
