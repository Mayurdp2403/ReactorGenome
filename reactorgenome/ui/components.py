"""Python side of the two browser components: the animation player and the design board."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import streamlit.components.v1 as components

_ROOT = Path(__file__).resolve().parent.parent / "components"
_player = components.declare_component("rg_player", path=str(_ROOT / "player"))
_board = components.declare_component("rg_board", path=str(_ROOT / "flowboard"))


def _clean(v):
    if isinstance(v, np.ndarray):
        return _clean(v.tolist())
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    if isinstance(v, dict):
        return {k: _clean(x) for k, x in v.items()}
    if isinstance(v, (np.floating, float)):
        return float(v) if np.isfinite(v) else None
    if isinstance(v, np.integer):
        return int(v)
    return v


def player(data: dict, key: str):
    """data: kind, t, and the per-kind arrays described in components/player/index.html."""
    d = _clean(data)
    d["sig"] = hashlib.md5(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()
    _player(data=d, key=key, default=None, height=440)


def board(plant: dict, results: dict | None, key: str):
    return _board(plant=_clean(plant), results=_clean(results or {}), key=key, default=None, height=520)
