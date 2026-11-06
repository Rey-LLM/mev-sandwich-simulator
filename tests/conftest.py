"""Shared pytest setup: keep every test independent of the developer's environment."""

import pytest


@pytest.fixture(autouse=True)
def _hermetic_settings(monkeypatch):
    """Ignore any sandwich.toml and $MEV_SIM_WORKERS unless a test opts in."""
    monkeypatch.setenv("MEV_SIM_CONFIG", "none")
    monkeypatch.delenv("MEV_SIM_WORKERS", raising=False)
