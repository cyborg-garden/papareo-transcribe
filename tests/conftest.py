"""Load hermes-plugin/ the way the harness does: as a package from its dir."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

PLUGIN_DIR = Path(__file__).resolve().parents[1] / "hermes-plugin"


def load_plugin(name: str = "papareo_transcribe") -> types.ModuleType:
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        name, PLUGIN_DIR / "__init__.py", submodule_search_locations=[str(PLUGIN_DIR)]
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def plugin() -> types.ModuleType:
    return load_plugin()
