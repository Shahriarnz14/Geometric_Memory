"""Compatibility package shim for absolute imports.

This repository historically used absolute imports like:
`from geometric_memory.in_weights import ...`

When users run scripts from the repository root directory, Python cannot resolve
`geometric_memory` unless the parent directory is on `PYTHONPATH`.
This shim makes imports work in both common workflows:

1) Running from the parent directory (traditional package path)
2) Running directly from the repository root
"""

from pathlib import Path
import pkgutil

# Support namespace-package behavior and append repository root as a submodule
# search location so `geometric_memory.in_weights` resolves from repo root.
__path__ = pkgutil.extend_path(__path__, __name__)  # type: ignore[name-defined]
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in __path__:
    __path__.append(str(repo_root))

