"""Validation rung 5 -- the one that matters.

The Stage 2 exit criterion, pulled forward into Stage 0 against fakes: the same
loop, driven by the same config file, runs with a human attached and with no
human attached. If this ever needs a change inside ControlLoop to pass, the
ActionSource abstraction is wrong -- and the entire autonomy track (Stages 6,
7, 11) is built on it.
"""

from __future__ import annotations

import yaml

from teleop_sim.control.sources import PolicySource, TeleopSource
from teleop_sim.core.clock import ManualClock
from teleop_sim.core.config import RunConfig, build_system
from teleop_sim.core.types import Outcome
from tests.conftest import CONFIG_DIR

TELEOP_CONFIG = CONFIG_DIR / "fake_teleop.yaml"
POLICY_CONFIG = CONFIG_DIR / "fake_policy.yaml"


def test_the_two_configs_differ_only_in_the_source_block():
    teleop = yaml.safe_load(TELEOP_CONFIG.read_text())
    policy = yaml.safe_load(POLICY_CONFIG.read_text())

    assert set(teleop) == set(policy)
    differing = {key for key in teleop if teleop[key] != policy[key]}
    assert differing == {"source"}, (
        "autonomy must be a one-section config change; these configs also differ in "
        f"{sorted(differing - {'source'})}"
    )


def _run(path):
    config = RunConfig.from_yaml(path)
    system = build_system(config, clock=ManualClock())
    return system, system.loop.run_episode(seed=config.seed)


def test_same_loop_runs_driven_and_autonomous():
    teleop_system, teleop_result = _run(TELEOP_CONFIG)
    policy_system, policy_result = _run(POLICY_CONFIG)

    assert isinstance(teleop_system.source, TeleopSource)
    assert isinstance(policy_system.source, PolicySource)
    assert type(teleop_system.loop) is type(policy_system.loop)

    for result in (teleop_result, policy_result):
        assert result.outcome is Outcome.SUCCESS
        assert result.steps == 39

    # Identical episodes, opposite provenance -- which is what makes
    # intervention data (Stage 7) separable from autonomous rollouts.
    assert teleop_result.robot_spec_hash == policy_result.robot_spec_hash
    assert teleop_result.policy_spec_hash is None
    assert policy_result.policy_spec_hash is not None

    assert teleop_result.intervention_frac == 1.0
    assert policy_result.intervention_frac == 0.0
    assert teleop_result.autonomous is False
    assert policy_result.autonomous is True
