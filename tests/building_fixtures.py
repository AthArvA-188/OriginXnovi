"""Shared SYNTHETIC tower fixture for the building tests (building_spec section 13).

Import it into a test module with `from building_fixtures import small_tower  # noqa: F401`. The tower is
synthetic (labelled so in every file) and is never used to claim accuracy.
"""

from __future__ import annotations

import pytest

from cascade.building.model import BuildingData
from cascade.building.synthetic import TowerConfig, generate_tower

SMALL_CFG = TowerConfig(floors=10, days=120, seed=7)


@pytest.fixture(scope="session")
def small_tower(tmp_path_factory) -> BuildingData:
    """10 floors, 120 days, seed 7, generated once per test session."""
    return generate_tower(tmp_path_factory.mktemp("small_tower"), SMALL_CFG)
