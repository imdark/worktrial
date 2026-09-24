"""Validation rung 1: the canonical types refuse out-of-contract data."""

from __future__ import annotations

import numpy as np
import pytest

from teleop_sim.core.types import (
    Action,
    ActionChunk,
    ActionOrigin,
    ControlMode,
    EpisodeResult,
    Observation,
    Outcome,
    TeleopCommand,
    TypeValidationError,
)


def _image(h: int = 4, w: int = 4) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def _action(**overrides) -> Action:
    base = {
        "mode": ControlMode.JOINT_POSITION,
        "values": [0.1, 0.2],
        "gripper": 0.0,
        "timestamp": 1.0,
        "obs_timestamp": 0.9,
    }
    base.update(overrides)
    return Action(**base)


# ------------------------------------------------------------- Observation


def test_observation_coerces_joint_pos_to_float_array():
    obs = Observation(images={}, joint_pos=[0, 1, 2], gripper=0.0, timestamp=0.0)
    assert obs.joint_pos.dtype == np.float64
    assert obs.dof == 3


def test_observation_rejects_unnormalised_gripper():
    with pytest.raises(TypeValidationError, match="normalised"):
        Observation(images={}, joint_pos=[0.0], gripper=4095.0, timestamp=0.0)


@pytest.mark.parametrize(
    "image, message",
    [
        (np.zeros((4, 4, 3), dtype=np.float32), "uint8"),
        (np.zeros((4, 4), dtype=np.uint8), "HxWx3"),
        (np.zeros((4, 4, 4), dtype=np.uint8), "HxWx3"),
    ],
)
def test_observation_rejects_bad_images(image, message):
    with pytest.raises(TypeValidationError, match=message):
        Observation(images={"wrist": image}, joint_pos=[0.0], gripper=0.0, timestamp=0.0)


@pytest.mark.parametrize("field", ["joint_vel", "joint_current", "joint_torque"])
def test_per_joint_sensing_must_match_dof(field):
    with pytest.raises(TypeValidationError, match=field):
        Observation(
            images={}, joint_pos=[0.0, 1.0], gripper=0.0, timestamp=0.0, **{field: [0.0]}
        )


def test_ee_wrench_must_be_six_vector():
    with pytest.raises(TypeValidationError, match="ee_wrench"):
        Observation(
            images={}, joint_pos=[0.0], gripper=0.0, timestamp=0.0, ee_wrench=[0.0, 1.0]
        )


def test_observation_rejects_non_finite_values():
    with pytest.raises(TypeValidationError, match="non-finite"):
        Observation(images={}, joint_pos=[0.0, np.nan], gripper=0.0, timestamp=0.0)


def test_has_reports_which_fields_are_actually_present():
    obs = Observation(
        images={"wrist": _image()},
        joint_pos=[0.0],
        gripper=0.0,
        timestamp=0.0,
        joint_current=[0.5],
    )
    assert obs.has("images") and obs.has("joint_current") and obs.has("joint_pos")
    assert not obs.has("joint_torque")
    assert not obs.has("ee_wrench")
    with pytest.raises(TypeValidationError, match="unknown observation key"):
        obs.has("telepathy")


def test_observation_record_round_trip_carries_sensing():
    obs = Observation(
        images={"wrist": _image()},
        joint_pos=[0.1, -0.2],
        joint_vel=[0.0, 0.0],
        joint_current=[0.3, 0.4],
        ee_wrench=[0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        ee_pose=[0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0],
        gripper=0.25,
        timestamp=1.5,
    )
    restored = Observation.from_record(obs.to_record())
    np.testing.assert_allclose(restored.joint_pos, obs.joint_pos)
    np.testing.assert_allclose(restored.joint_current, obs.joint_current)
    np.testing.assert_allclose(restored.ee_wrench, obs.ee_wrench)
    assert restored.joint_torque is None
    assert obs.to_record()["camera_names"] == ["wrist"]


# ------------------------------------------------------------------ Action


def test_action_record_round_trip_preserves_provenance_and_staleness():
    action = _action(source=ActionOrigin.HUMAN_CORRECTION)
    restored = Action.from_record(action.to_record())
    np.testing.assert_allclose(restored.values, action.values)
    assert restored.mode is ControlMode.JOINT_POSITION
    assert restored.source is ActionOrigin.HUMAN_CORRECTION
    assert restored.obs_timestamp == 0.9


def test_staleness_is_the_age_of_the_observation_used():
    action = _action(timestamp=1.0, obs_timestamp=0.9)
    assert action.staleness_ms(now=1.0) == pytest.approx(100.0)
    assert _action(obs_timestamp=None).staleness_ms(now=1.0) is None


def test_replace_values_keeps_provenance_and_timestamps():
    action = _action(values=[5.0], source=ActionOrigin.HUMAN_TELEOP)
    clipped = action.replace_values(np.array([1.0]))
    assert clipped.source is ActionOrigin.HUMAN_TELEOP
    assert clipped.timestamp == action.timestamp
    assert clipped.obs_timestamp == action.obs_timestamp
    assert clipped.values[0] == 1.0


def test_action_rejects_unknown_origin():
    with pytest.raises(ValueError, match="alien"):
        _action(source="alien")


def test_origins_classify_human_versus_machine():
    assert ActionOrigin.HUMAN_TELEOP.is_human
    assert ActionOrigin.HUMAN_CORRECTION.is_human
    assert not ActionOrigin.POLICY.is_human
    assert not ActionOrigin.POLICY_HIGH_LEVEL.is_human


# ------------------------------------------------------------- ActionChunk


def test_chunk_backfills_obs_timestamp_onto_its_actions():
    chunk = ActionChunk(
        actions=[_action(obs_timestamp=None), _action(obs_timestamp=None)],
        obs_timestamp=7.0,
        horizon_hz=30.0,
    )
    assert len(chunk) == 2
    assert all(a.obs_timestamp == 7.0 for a in chunk.actions)


def test_empty_chunk_rejected():
    with pytest.raises(TypeValidationError, match="at least one action"):
        ActionChunk(actions=[], obs_timestamp=0.0, horizon_hz=30.0)


def test_chunk_rate_must_be_positive():
    with pytest.raises(TypeValidationError, match="horizon_hz"):
        ActionChunk(actions=[_action()], obs_timestamp=0.0, horizon_hz=0.0)


# ------------------------------------------------- TeleopCommand / results


def test_teleop_command_events():
    cmd = TeleopCommand(
        kind="joint",
        values=[0.0],
        gripper=0.0,
        timestamp=0.0,
        buttons={"record": True, "estop": False},
    )
    assert cmd.pressed("record")
    assert not cmd.pressed("estop")
    assert cmd.pressed_events() == {"record"}


def test_teleop_command_rejects_unknown_kind():
    with pytest.raises(TypeValidationError, match="kind"):
        TeleopCommand(kind="telepathy", values=[0.0], gripper=0.0, timestamp=0.0)


def test_episode_result_flags():
    good = EpisodeResult(Outcome.SUCCESS, steps=10, duration=1.0)
    assert good.succeeded and good.autonomous
    assisted = EpisodeResult(Outcome.SUCCESS, 10, 1.0, intervention_frac=0.3)
    assert assisted.succeeded and not assisted.autonomous


def test_episode_result_record_carries_provenance():
    record = EpisodeResult(
        Outcome.WATCHDOG,
        steps=5,
        duration=0.5,
        robot_spec_hash="abc123",
        policy_spec_hash="def456",
        code_version="deadbee",
        staleness_ms={"p50": 12.0, "p95": 40.0, "max": 51.0},
        underruns=3,
    ).to_record()

    assert record["outcome"] == "watchdog"
    assert record["robot_spec_hash"] == "abc123"
    assert record["policy_spec_hash"] == "def456"
    assert record["code_version"] == "deadbee"
    assert record["staleness_ms"]["p95"] == 40.0
    assert record["underruns"] == 3
