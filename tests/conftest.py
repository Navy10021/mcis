from __future__ import annotations

from pathlib import Path

import pytest
import yaml

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "settings.yaml"


@pytest.fixture(scope="session")
def config() -> dict:
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="session")
def sample_config(config) -> dict:
    return config


@pytest.fixture(scope="session")
def project_root() -> Path:
    return Path(__file__).resolve().parent.parent
