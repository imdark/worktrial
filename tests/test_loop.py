"""Validation rung 4: the loop runs, paces itself, and logs what was sent."""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest

from teleop_sim.control.loop import ControlLoop
from teleop_sim.control.safety import SimWatchdog
from teleop_sim.control.sources import PolicySource, TeleopSource
from teleop_sim.core.types import ActionOrigin, Button, ControlMode, Outcome
from teleop_sim.policies.constant import ConstantPolicy
from teleop_sim.retarget.identity import IdentityRetargeter
from tests.fakes import (
    CollectingRecorder,
    CountingReset,
    FakeRobot,
    FakeSuccessDetector,
    FakeTeleop,
    SlowPolicy,
)


def _loop(spec, clock, source, recorder=None, success_after=None, max_steps=100, robot=None):
    return ControlLoop(
        robot or FakeRobot(spec, clock),
        source,
        FakeSuccessDetector(success_after=success_after),
        CountingReset(),
        SimWatchdog(spec, max_steps=max_steps),
        clock=clock,
        recorder=recorder,
    )


def _teleop(spec, clock, **kwargs):
    return TeleopSource(FakeTeleop(spec, clock, **kwargs), IdentityRetargeter(spec))


def _policy(spec, clock):
    return PolicySource(ConstantPolicy(spec, clock), clock=clock)


def test_loop_runs_to_the_step_limit(spec, clock):
    recorder = CollectingRecorder()
    result = _loop(spec, clock, _teleop(spec, clock), recorder).run_episode(seed=0)

    assert result.outcome is Outcome.TIMEOUT
    assert result.failure_tag == "max_steps"
    assert result.steps == 100
    assert len(recorder.actions) == 100
    assert recorder.episodes_started == 1
    assert recorder.results == [result]


def test_timestamps_are_monotonic(spec, clock):
    recorder = CollectingRecorder()
    _loop(spec, clock, _teleop(spec, clock), recorder).run_episode(seed=0)
    stamps = [obs.timestamp for obs in recorder.observations]
    assert all(b >= a for a, b in pairwise(stamps))
    assert stamps[-1] > stamps[0]


def test_loop_holds_its_nominal_rate(spec, clock):
    loop = _loop(spec, clock, _teleop(spec, clock))
    result = loop.run_episode(seed=0)
    stats = result.extra["rate"]
    assert stats["mean_hz"] == pytest.approx(30.0)
    assert stats["jitter"] < 1e-9
    assert loop.rate.overruns == 0


def test_recorded_actions_are_what_the_robot_accepted(spec, clock):
    """The leader ramp eventually demands joint targets beyond the limits.
    What lands in the dataset must be the clipped command, not the request."""
    recorder = CollectingRecorder()
    _loop(spec, clock, _teleop(spec, clock, step=0.5), recorder).run_episode(seed=0)

    values = np.stack([a.values for a in recorder.actions])
    assert np.all(values <= spec.upper_limits + 1e-9)
    assert np.any(np.isclose(values[-1], spec.upper_limits))
    assert all(a.mode is ControlMode.JOINT_POSITION for a in recorder.actions)


def test_reset_strategy_runs_every_episode(spec, clock):
    reset = CountingReset()
    loop = ControlLoop(
        FakeRobot(spec, clock),
        _policy(spec, clock),
        FakeSuccessDetector(success_after=3),
        reset,
        SimWatchdog(spec, max_steps=50),
        clock=clock,
    )
    loop.run(episodes=3, seed=0)
    assert reset.calls == 3


def test_provenance_separates_human_from_policy(spec, clock):
    recorder = CollectingRecorder()
    result = _loop(spec, clock, _teleop(spec, clock), recorder).run_episode(seed=0)
    assert all(a.source is ActionOrigin.HUMAN_TELEOP for a in recorder.actions)
    assert result.intervention_frac == 1.0

    auto_recorder = CollectingRecorder()
    auto = _loop(
        spec, clock, _policy(spec, clock), auto_recorder, max_steps=20
    ).run_episode(seed=0)
    assert all(a.source is ActionOrigin.POLICY for a in auto_recorder.actions)
    assert auto.intervention_frac == 0.0


def test_episode_result_carries_provenance_hashes(spec, clock):
    result = _loop(spec, clock, _policy(spec, clock), max_steps=5).run_episode(seed=0)
    assert result.robot_spec_hash == spec.content_hash()
    assert result.policy_spec_hash is not None
    assert result.code_version is not None
    assert result.staleness_ms  # p50/p95/max populated from Action.obs_timestamp
    assert result.underruns == 0


def test_teleop_runs_carry_no_policy_hash(spec, clock):
    result = _loop(spec, clock, _teleop(spec, clock), max_steps=5).run_episode(seed=0)
    assert result.policy_spec_hash is None


def test_chunked_policy_calls_predict_once_per_chunk(spec, clock):
    """Buffering lives in the source, so a chunking policy needs no threading
    or bookkeeping of its own."""
    policy = SlowPolicy(spec, clock, horizon=5)
    source = PolicySource(policy, clock=clock)
    result = _loop(spec, clock, source, max_steps=20).run_episode(seed=0)

    assert result.steps == 20
    assert policy.predict_calls == 4  # 20 steps / 5-action chunks


def test_estop_aborts_before_the_action_is_sent(spec, clock):
    recorder = CollectingRecorder()
    source = _teleop(spec, clock, button=Button.ESTOP, button_at=5)
    result = _loop(spec, clock, source, recorder).run_episode(seed=0)

    assert result.outcome is Outcome.ABORTED
    assert result.failure_tag == "estop"
    assert result.steps == 4
    # The fifth action was generated but never reached the robot.
    assert len(recorder.actions) == 4


def test_discard_event_discards_the_episode(spec, clock):
    recorder = CollectingRecorder()
    source = _teleop(spec, clock, button=Button.DISCARD, button_at=3)
    result = _loop(spec, clock, source, recorder).run_episode(seed=0)

    assert result.failure_tag == "discard"
    assert recorder.episodes_discarded == 1
    assert recorder.results == []


def test_success_clears_the_failure_tag(spec, clock):
    result = _loop(spec, clock, _policy(spec, clock), success_after=10).run_episode(seed=0)
    assert result.outcome is Outcome.SUCCESS
    assert result.failure_tag is None
    assert result.steps == 9
