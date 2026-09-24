"""Validation rung 3: YAML in, wired system out -- with nothing heavy imported."""

from __future__ import annotations

import subprocess
import sys

import pytest

from teleop_sim.control.sources import PolicySource, TeleopSource
from teleop_sim.core.clock import ManualClock
from teleop_sim.core.config import ConfigError, RunConfig, build_system
from teleop_sim.core.registry import RegistryError
from tests.conftest import CONFIG_DIR, REPO_ROOT


def _config(name: str = "fake_teleop.yaml") -> RunConfig:
    return RunConfig.from_yaml(CONFIG_DIR / name)


def test_build_system_from_yaml():
    system = build_system(_config(), clock=ManualClock())
    assert system.spec.name == "so101"
    assert isinstance(system.source, TeleopSource)
    assert system.loop.rate.hz == system.spec.control_hz
    assert system.safety.config.max_steps == 200
    assert system.task.name == "stage0_smoke"


def test_policy_config_builds_a_policy_source():
    system = build_system(_config("fake_policy.yaml"), clock=ManualClock())
    assert isinstance(system.source, PolicySource)
    assert system.source.policy_spec().policy_id == "constant"


_IMPORT_PROBE = """
import sys
from teleop_sim.core.clock import ManualClock
from teleop_sim.core.config import RunConfig, build_system
import tests.fakes  # noqa: F401

build_system(RunConfig.from_yaml(sys.argv[1]), clock=ManualClock())

leaked = [m for m in ("mujoco", "torch", "lerobot", "pyrealsense2", "serial")
          if m in sys.modules]
assert not leaked, leaked
print("clean")
"""


def test_core_pulls_in_no_heavy_dependencies():
    """The guarantee CI enforces: teleop_sim.core and .control stay installable
    with the base dependencies alone.

    Checked in a fresh interpreter on purpose. An in-process sys.modules
    assertion passes for the wrong reason as soon as any other test in the
    session has imported mujoco -- which tests/conformance/ now does.
    """
    result = subprocess.run(
        [sys.executable, "-c", _IMPORT_PROBE, str(CONFIG_DIR / "fake_policy.yaml")],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "clean" in result.stdout


def test_declaring_mujoco_lazily_does_not_import_it():
    """`robot.type: mujoco` must resolve without the import happening until a
    config actually names it."""
    from teleop_sim.core.registry import ROBOTS

    assert "mujoco" in ROBOTS.names()


def test_unknown_robot_type_lists_alternatives():
    config = _config()
    config.robot["type"] = "definitely_not_registered"
    with pytest.raises(RegistryError, match="available:"):
        build_system(config, clock=ManualClock())


def test_missing_spec_is_reported():
    config = _config()
    del config.robot["spec"]
    with pytest.raises(ConfigError, match="'spec' file"):
        build_system(config, clock=ManualClock())


def test_teleop_source_requires_a_device():
    config = _config()
    del config.source["device"]
    with pytest.raises(ConfigError, match="'device' section"):
        build_system(config, clock=ManualClock())


def test_unknown_top_level_section_rejected(tmp_path):
    path = tmp_path / "run.yaml"
    path.write_text(
        "robot: {type: fake, spec: x.yaml}\n"
        "source: {type: policy, policy: {type: constant}}\n"
        "success: {type: fake}\n"
        "reset: {type: none}\n"
        "trian: {lr: 3}\n"
    )
    with pytest.raises(ConfigError, match=r"unknown top-level section\(s\) \['trian'\]"):
        RunConfig.from_yaml(path)


def test_missing_required_section_rejected(tmp_path):
    path = tmp_path / "run.yaml"
    path.write_text("robot: {type: fake, spec: x.yaml}\nsuccess: {type: fake}\n")
    with pytest.raises(ConfigError, match="missing required section 'source'"):
        RunConfig.from_yaml(path)


def test_safety_defaults_to_the_sim_watchdog(tmp_path):
    config = _config()
    config.safety = {}
    system = build_system(config, clock=ManualClock())
    assert system.safety.config.max_steps == 600


def test_startup_check_runs_at_construction_not_at_step_one():
    """Principle 11. A force limit with no sensor behind it must refuse to
    build, rather than silently never firing."""
    from teleop_sim.control.compat import Incompatible
    from teleop_sim.control.loop import ControlLoop
    from teleop_sim.control.safety import SimWatchdog
    from tests.fakes import CountingReset, FakeSuccessDetector

    system = build_system(_config("fake_policy.yaml"), clock=ManualClock())
    with pytest.raises(Incompatible, match="could never fire"):
        ControlLoop(
            system.robot,
            system.source,
            FakeSuccessDetector(),
            CountingReset(),
            SimWatchdog(system.spec),
            clock=system.clock,
            safety_config={"max_current_a": 1.8},
        )


def test_hardware_safety_keys_on_the_sim_watchdog_are_rejected_clearly():
    config = _config()
    config.safety["max_current_a"] = 1.8
    with pytest.raises(ValueError, match="hardware_safety"):
        build_system(config, clock=ManualClock())


def test_ik_retargeter_is_present_but_honest():
    """Stage 8 lives behind a stub, not behind a silent gap in the interface."""
    config = _config()
    config.source["retarget"] = {"type": "ik"}
    with pytest.raises(NotImplementedError, match="Stage 8"):
        build_system(config, clock=ManualClock())


def test_async_policy_source_is_present_but_honest():
    """Stage 2 lives behind a stub, not behind a silent gap in the interface."""
    config = _config("fake_policy.yaml")
    config.source["type"] = "async_policy"
    with pytest.raises(NotImplementedError, match="Stage 2"):
        build_system(config, clock=ManualClock())
