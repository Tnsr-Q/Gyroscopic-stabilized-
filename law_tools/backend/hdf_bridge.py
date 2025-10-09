"""Utility helpers for reading RCC checkpoint data structures."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import h5py
import numpy as np


@dataclass
class LawCheckpoint:
    """Lightweight container describing the loaded checkpoint content."""

    law_tensor: np.ndarray
    metadata: Dict[str, Any]


def _read_attrs(handle: h5py.File) -> Dict[str, Any]:
    """Return a dictionary copy of the attributes on *handle*."""

    attrs: Dict[str, Any] = {}
    for key, value in handle.attrs.items():
        if isinstance(value, bytes):
            attrs[key] = value.decode("utf-8")
        else:
            attrs[key] = value
    return attrs


def load_checkpoint(path: str | Path, dataset: str = "law_tensor") -> LawCheckpoint:
    """Load ``dataset`` from *path* and return :class:`LawCheckpoint`."""

    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    with h5py.File(checkpoint_path, "r") as handle:
        if dataset not in handle:
            raise KeyError(f"Dataset '{dataset}' missing in checkpoint {checkpoint_path}")
        tensor = handle[dataset][:]
        metadata = _read_attrs(handle)
    return LawCheckpoint(law_tensor=tensor, metadata=metadata)


def iter_dataset_names(path: str | Path) -> Iterable[str]:
    """Yield dataset names contained in the checkpoint."""

    with h5py.File(path, "r") as handle:
        for name in handle.keys():
            yield name


def load_additional_field(path: str | Path, field: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Load a secondary field from the checkpoint returning data and attrs."""

    with h5py.File(path, "r") as handle:
        if field not in handle:
            raise KeyError(f"Field '{field}' not present in checkpoint {path}")
        data = handle[field][:]
        attrs = _read_attrs(handle[field]) if isinstance(handle[field], h5py.Dataset) else {}
    return data, attrs


__all__ = ["LawCheckpoint", "load_checkpoint", "iter_dataset_names", "load_additional_field"]
