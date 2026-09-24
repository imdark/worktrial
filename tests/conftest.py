from __future__ import annotations

from pathlib import Path

import pytest

import teleop_sim
from teleop_sim.core.clock import ManualClock
from teleop_sim.core.spec import RobotSpec
from tests import fakes  # noqa: F401  -- import registers the fake components

PACKAGE_ROOT = Path(teleop_sim.__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent
SPEC_PATH = PACKAGE_ROOT / "robots" / "specs" / "so101.yaml"
CONFIG_DIR = PACKAGE_ROOT / "configs"

#: The Kronos model is generated from proprietary CAD and is not in the
#: repository; scripts/build_kronos_assets.py + build_kronos_model.py rebuild it
#: from the Hardware Archive. Without it, every test that needs it skips.
KRONOS_DIR = REPO_ROOT / "assets" / "end_effectors" / "kronos"
requires_kronos = pytest.mark.skipif(
    not (KRONOS_DIR / "kronos.xml").exists() or not (KRONOS_DIR / "kronos_extract.json").exists(),
    reason="Kronos assets not generated (proprietary CAD; see README)",
)


@pytest.fixture
def spec() -> RobotSpec:
    return RobotSpec.from_yaml(SPEC_PATH)


@pytest.fixture
def clock() -> ManualClock:
    return ManualClock()
