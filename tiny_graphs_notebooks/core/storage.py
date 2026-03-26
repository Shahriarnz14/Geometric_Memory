"""Small JSON/pickle and manifest helpers for tiny notebook artifacts."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Mapping


def load_json(path: Path | str) -> dict[str, object]:
    """Loads a JSON object from disk.

    Args:
        path: JSON file path.

    Returns:
        dict[str, object]: Parsed JSON payload.
    """
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(path: Path | str, payload: Mapping[str, object]) -> None:
    """Writes a dictionary payload to disk as pretty JSON.

    Args:
        path: Output JSON path.
        payload: JSON-serializable mapping.

    Returns:
        None: File is written to disk.
    """
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def write_pickle(path: Path | str, payload: object) -> None:
    """Serializes an object to pickle.

    Args:
        path: Output pickle path.
        payload: Pickle-serializable object.

    Returns:
        None: File is written to disk.
    """
    with open(path, 'wb') as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


def read_pickle(path: Path | str):
    """Loads a pickled object from disk.

    Args:
        path: Pickle file path.

    Returns:
        object: Deserialized object.
    """
    with open(path, 'rb') as f:
        return pickle.load(f)


def find_latest_manifest(manifest_dir: Path) -> Path | None:
    """Finds the newest manifest file in a directory.

    Args:
        manifest_dir: Directory containing `*.json` manifests.

    Returns:
        Path | None: Most recently modified manifest, or `None` when missing.
    """
    manifest_paths = sorted(manifest_dir.glob('*.json'), key=lambda p: p.stat().st_mtime)
    return manifest_paths[-1] if manifest_paths else None


def find_manifest_by_checkpoint(manifest_dir: Path, checkpoint_path: str) -> Path | None:
    """Finds a manifest whose checkpoint path matches a checkpoint file.

    Args:
        manifest_dir: Directory containing `*.json` manifests.
        checkpoint_path: Checkpoint file path to match.

    Returns:
        Path | None: Matching manifest path or `None`.
    """
    checkpoint_resolved = str(Path(checkpoint_path).resolve())
    for manifest_path in manifest_dir.glob('*.json'):
        payload = load_json(manifest_path)
        payload_checkpoint = str(Path(str(payload.get('checkpoint_path', ''))).resolve())
        if payload_checkpoint == checkpoint_resolved:
            return manifest_path
    return None
