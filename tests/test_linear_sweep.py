"""The straight-line sweep used to test the safety monitors, in MuJoCo."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import teleop_sim.builtins  # noqa: F401
from teleop_sim.core.clock import ManualClock
from teleop_sim.core.config import RunConfig, build_system
from teleop_sim.core.types import Outcome
from teleop_sim.robots.kinematics import Kinematics
from teleop_sim.telemetry import TelemetryRecorder
from tests.conftest import requires_kronos

SIM = Path(__file__).resolve().parent.parent / "teleop_sim/configs/yam_kronos_sweep_sim.yaml"


@requires_kronos
def test_the_sweep_moves_the_tip_along_straight_lines(tmp_path):
    config = RunConfig.from_yaml(SIM)
    config.source["policy"]["cycles"] = 1
    system = build_system(config, clock=ManualClock())
    policy = system.source.policy
    system.loop.recorder = TelemetryRecorder(
        tmp_path, phase_of=lambda: policy.phase, control_hz=system.spec.control_hz
    )
    result = system.loop.run_episode(seed=0)
    assert result.outcome is Outcome.SUCCESS
    import json

    rows = [json.loads(line) for line in (tmp_path / "telemetry.jsonl").open()]
    assert {"forward", "back", "down", "up"} <= {r["phase"] for r in rows}
    kin = Kinematics(system.spec)
    for phase, along, across in (("forward", 0, [1, 2]), ("down", 2, [0, 1])):
        tips = np.array(
            [
                kin.frame(np.array(r["cmd_joint_pos"]), "site", system.spec.ee_site)[0]
                for r in rows
                if r["phase"] == phase
            ]
        )
        assert np.abs(tips[:, across] - tips[0, across]).max() < 0.002  # straight to 2 mm
        assert abs(tips[-1, along] - tips[0, along]) > 0.1  # and it really travelled


def test_the_down_stroke_must_stay_clear_of_the_table():
    config = RunConfig.from_yaml(SIM)
    config.source["policy"].update(center=[0.42, -0.05, 0.12], travel_z=0.06)
    with pytest.raises(ValueError, match="min_z"):
        build_system(config, clock=ManualClock())
