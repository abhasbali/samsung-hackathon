"""YAML configuration loading with ``extends`` inheritance and dotted overrides."""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

from codefusion.config.schema import INTENTS, CodeFusionConfig


def _deep_merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _load_yaml_tree(path: Path, seen: set[Path] | None = None) -> dict[str, Any]:
    seen = seen or set()
    path = path.resolve()
    if path in seen:
        raise ValueError(f"circular 'extends' in {path}")
    seen.add(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    parent = data.pop("extends", None)
    if parent:
        base = _load_yaml_tree((path.parent / parent), seen)
        data = _deep_merge(base, data)
    return data


def _parse_scalar(v: str) -> Any:
    return yaml.safe_load(v)


def apply_overrides(data: dict[str, Any], overrides: list[str] | None) -> dict[str, Any]:
    """Apply ``section.key=value`` overrides (values parsed as YAML)."""
    data = copy.deepcopy(data)
    for ov in overrides or []:
        if "=" not in ov:
            raise ValueError(f"override must look like a.b=c, got {ov!r}")
        key, val = ov.split("=", 1)
        node = data
        parts = key.strip().split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = _parse_scalar(val)
    return data


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> CodeFusionConfig:
    """``CODEFUSION_DEVICE`` (e.g. ``cuda``) sets the dense and reranker device; explicit overrides still win."""
    data: dict[str, Any] = {}
    if path is not None:
        data = _load_yaml_tree(Path(path))
    device = os.environ.get("CODEFUSION_DEVICE")
    if device:
        overrides = [f"dense.device={device}", f"reranker.device={device}", *(overrides or [])]
    data = apply_overrides(data, overrides)
    return CodeFusionConfig.model_validate(data)


def config_to_dict(cfg: CodeFusionConfig) -> dict[str, Any]:
    return cfg.model_dump(mode="json")


__all__ = ["INTENTS", "CodeFusionConfig", "apply_overrides", "config_to_dict", "load_config"]
