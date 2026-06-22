from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping

import yaml


def deep_update(base: dict[str, Any], update: Mapping[str, Any]) -> dict[str, Any]:
    for key, value in update.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base


def _parse_value(value: str) -> Any:
    return yaml.safe_load(value)


def apply_overrides(config: dict[str, Any], overrides: list[str] | None) -> dict[str, Any]:
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Invalid override '{item}', expected key=value")
        dotted_key, raw_value = item.split("=", 1)
        keys = dotted_key.split(".")
        cursor = config
        for key in keys[:-1]:
            child = cursor.setdefault(key, {})
            if not isinstance(child, dict):
                raise ValueError(f"Cannot override nested key below '{key}'")
            cursor = child
        cursor[keys[-1]] = _parse_value(raw_value)
    return config


def load_config(
    path: str | Path,
    overrides: list[str] | None = None,
    defaults: str | Path | None = None,
) -> dict[str, Any]:
    config: dict[str, Any] = {}
    if defaults is not None:
        with Path(defaults).open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}
    with Path(path).open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    deep_update(config, loaded)
    return apply_overrides(config, overrides)


def load_project_config(
    path: str | Path,
    overrides: list[str] | None = None,
) -> dict[str, Any]:
    path = Path(path)
    default_path = Path(__file__).resolve().parents[1] / "configs" / "ultralite_dce.yaml"
    defaults = None if path.resolve() == default_path.resolve() else default_path
    return load_config(path, overrides=overrides, defaults=defaults)


def save_config(config: Mapping[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(dict(config), handle, sort_keys=False, allow_unicode=True)
