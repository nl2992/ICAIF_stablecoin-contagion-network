"""Configuration loader for stressnet.

Reads YAML configs and environment variables. All configs are loaded lazily
and cached on first access.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

load_dotenv()

_CONFIGS_DIR = Path(__file__).parent.parent.parent / "configs"


def _load_yaml(name: str) -> dict[str, Any]:
    path = _CONFIGS_DIR / name
    with open(path) as fh:
        return yaml.safe_load(fh)


def load_events() -> dict[str, Any]:
    """Load event window definitions from configs/events.yaml."""
    return _load_yaml("events.yaml")["events"]


def load_nodes() -> dict[str, Any]:
    """Load node taxonomy from configs/nodes.yaml."""
    return _load_yaml("nodes.yaml")["nodes"]


def load_features() -> dict[str, Any]:
    """Load feature definitions from configs/features.yaml."""
    return _load_yaml("features.yaml")["features"]


def load_sources() -> dict[str, Any]:
    """Load data source definitions from configs/sources.yaml."""
    return _load_yaml("sources.yaml")["sources"]


def load_models() -> dict[str, Any]:
    """Load model configurations from configs/models.yaml."""
    return _load_yaml("models.yaml")["models"]


def load_paper() -> dict[str, Any]:
    """Load paper metadata from configs/paper.yaml."""
    return _load_yaml("paper.yaml")["paper"]


def get_env(key: str, default: str | None = None) -> str | None:
    """Return an environment variable value with an optional default."""
    return os.environ.get(key, default)


def data_root() -> Path:
    return Path(get_env("DATA_DIR", "./data"))


def bronze_root() -> Path:
    return data_root() / "bronze"


def silver_root() -> Path:
    return data_root() / "silver"


def gold_root() -> Path:
    return data_root() / "gold"


def manifests_root() -> Path:
    return data_root() / "manifests"


def results_root() -> Path:
    return Path(get_env("RESULTS_DIR", "./results"))


def load_config() -> dict[str, Any]:
    """Return a merged configuration dict."""
    return {
        "events": load_events(),
        "nodes": load_nodes(),
        "features": load_features(),
        "sources": load_sources(),
        "models": load_models(),
        "paper": load_paper(),
    }
