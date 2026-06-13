"""Load YAML config + apply CLI overrides.

Usage trong eval/run_experiment.py:
    cfg = load_config("configs/default.yaml", overrides=["model.name=lvface"])
    print(cfg["model"]["name"])  # 'lvface'
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def load_config(
    path: str | Path = "configs/default.yaml",
    overrides: list[str] | None = None,
) -> dict[str, Any]:
    """Read YAML file then apply `key.subkey=value` overrides from CLI."""
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if overrides:
        for item in overrides:
            apply_override(cfg, item)

    return cfg


def apply_override(cfg: dict[str, Any], item: str) -> None:
    """Mutate cfg in place: 'a.b.c=value' → cfg['a']['b']['c'] = parsed(value)."""
    if "=" not in item:
        raise ValueError(f"Override phải có dạng key=value, nhận: {item!r}")
    key, raw_value = item.split("=", 1)
    keys = key.split(".")

    node = cfg
    for k in keys[:-1]:
        if k not in node or not isinstance(node[k], dict):
            raise KeyError(f"Override key không tồn tại trong config: {key!r}")
        node = node[k]

    leaf = keys[-1]
    if leaf not in node:
        raise KeyError(f"Override key không tồn tại trong config: {key!r}")

    # Parse value with YAML để tự handle int/float/bool/list
    node[leaf] = yaml.safe_load(raw_value)


def merge_configs(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep-merge override vào copy của base. Hữu ích khi có configs/exp_X.yaml
    chỉ chứa các field khác default."""
    result = deepcopy(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = merge_configs(result[k], v)
        else:
            result[k] = v
    return result
