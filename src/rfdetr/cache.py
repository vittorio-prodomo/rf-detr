# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""
Centralized cache directory for downloaded model weights.

By default, weights are stored in ``~/.cache/roboflow/``.
Override with :func:`set_cache_dir` or the ``ROBOFLOW_CACHE_DIR``
environment variable.

Usage::

    from rfdetr.cache import get_cache_dir, set_cache_dir, resolve_weight_path

    # Query the current cache directory
    print(get_cache_dir())  # ~/.cache/roboflow

    # Override programmatically
    set_cache_dir("/my/custom/path")

    # Resolve a bare filename to a full cache path
    path = resolve_weight_path("rf-detr-seg-nano.pt")
    # -> /my/custom/path/rf-detr-seg-nano.pt
"""

import os
from pathlib import Path

_DEFAULT_CACHE_DIR = Path.home() / ".cache" / "roboflow"
_cache_dir: Path | None = None


def get_cache_dir() -> Path:
    """Return the current weight cache directory.

    Resolution order:

    1. Value set via :func:`set_cache_dir` (highest priority)
    2. ``ROBOFLOW_CACHE_DIR`` environment variable
    3. ``~/.cache/roboflow/`` (default)
    """
    if _cache_dir is not None:
        return _cache_dir
    env = os.environ.get("ROBOFLOW_CACHE_DIR")
    if env:
        return Path(env).expanduser().resolve()
    return _DEFAULT_CACHE_DIR


def set_cache_dir(path: str | Path) -> None:
    """Override the weight cache directory.

    Args:
        path: New cache directory. Supports ``~`` expansion.
    """
    global _cache_dir
    _cache_dir = Path(path).expanduser().resolve()


def resolve_weight_path(filename: str) -> str:
    """Turn a bare weight filename into a full path inside the cache dir.

    If *filename* already contains a directory component (e.g.
    ``/my/custom/model.pt`` or ``./model.pt``), it is returned as-is
    after ``~`` expansion.

    Args:
        filename: Weight filename, e.g. ``"rf-detr-seg-nano.pt"``.

    Returns:
        Absolute path string, e.g.
        ``"/home/user/.cache/roboflow/rf-detr-seg-nano.pt"``.
    """
    expanded = os.path.expanduser(filename)
    # If the user supplied a path with directory components, respect it.
    if os.path.dirname(expanded):
        return os.path.realpath(expanded)
    # Bare filename → resolve into cache directory.
    return str(get_cache_dir() / expanded)
