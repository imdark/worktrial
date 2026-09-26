"""Motor-torque contact detection, on synthetic data and the stock YAM model."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import teleop_sim.builtins  # noqa: F401
from teleop_sim.control.safety.contact import (
    ContactDetector,
    ContactMonitor,
    DetectorParams,
    RunArrays,
    TorqueModel,
    fit_params,
    sustained_max,
    with_contact_monitor,
)
from teleop_sim.core.registry import SAFETY, build
from teleop_sim.core.spec import RobotSpec
from teleop_sim.core.types import Action, ControlMode, Observation, Outcome
from teleop_sim.perception.vision import HazardReview

YAM = Path(__file__).resolve().parent.parent / "teleop_sim/robots/specs/yam.yaml"
DOF = 6


def params(**kw) -> DetectorParams:
    base = dict(
        coef=[[1, 0, 0, 0, 0]] * DOF,
        sigma=[0.5] * DOF,
        sigma_jump=[0.2] * DOF,
        track_max_deg=[2.0] * DOF,
    )
    base.update(kw)
    return DetectorParams(**base)


def test_noise_alone_does_not_trip():
    det = ContactDetector(params())
    rng = np.random.default_rng(0)
    for _ in range(500):
        assert det.update(rng.normal(0, 0.1, DOF), "survey") is None


def test_a_sudden_push_trips_within_three_steps():
    det = ContactDetector(params())
    for _ in range(40):
        assert det.update(np.zeros(DOF), "survey") is None
    push = np.zeros(DOF)
    push[1] = 2.0  # 10 sigma_jump on J2
    trips = [det.update(push, "survey") for _ in range(3)]
    assert trips[:2] == [None, None]
    assert trips[2] is not None and trips[2].joint == 1 and trips[2].rule == "jump"


def test_limits_are_looser_where_contact_is_expected():
    push = np.zeros(DOF)
    push[1] = 1.6  # 8 sigma_jump: over 6, under 6 x 2.5
    for phase, expect in (("survey", True), ("close", False)):
        det = ContactDetector(params())
        for _ in range(40):
            det.update(np.zeros(DOF), phase)
        tripped = any(det.update(push, phase) is not None for _ in range(5))
        assert tripped is expect, phase


def test_lagging_the_command_trips_only_past_free_motion_lag():
    det = ContactDetector(params())
    q = np.zeros(DOF)
    ok = np.radians([4.0] + [0.0] * 5)  # under 2 + 3 deg margin
    bad = np.radians([6.0] + [0.0] * 5)
    assert all(det.update(np.zeros(DOF), "above", ok, q) is None for _ in range(5))
    trips = [det.update(np.zeros(DOF), "above", bad, q) for _ in range(3)]
    assert trips[2] is not None and trips[2].rule == "tracking"


def test_sustained_max_needs_consecutive_masked_steps():
    x = np.array([[0.0], [5.0], [5.0], [5.0], [9.0], [0.0]])
    mask = np.ones(6, dtype=bool)
    assert sustained_max(x, mask, 3)[0] == 5.0
    mask[2] = False
    assert sustained_max(x, mask, 3)[0] == 0.0


def synthetic_run(model: TorqueModel, n=300, seed=0, name="run") -> RunArrays:
    rng = np.random.default_rng(seed)
    t = np.arange(n) / 28.0
    q = np.stack([0.3 * np.sin(0.2 * t + j) + (0.8 if j in (1, 2) else 0.0) for j in range(DOF)], 1)
    qd = np.gradient(q, t, axis=0)
    grav = np.stack([model.dynamics(q[i], np.zeros(DOF), np.zeros(DOF))[0] for i in range(n)])
    eff = -1.3 * grav + 0.2 + rng.normal(0, 0.05, grav.shape)  # opposite sign, scaled
    return RunArrays(name, t, ["survey"] * n, q, q.copy(), qd, eff)


def test_the_fit_recovers_a_sign_flipped_scaled_gravity_model():
    spec = RobotSpec.from_yaml(YAM)
    model = TorqueModel(spec)
    runs = [synthetic_run(model, seed=s, name=f"r{s}") for s in (0, 1)]
    fitted, p, residuals = fit_params(runs, spec)
    loaded = np.array(p.coef)
    loaded_gravity = loaded[1:3, 0]  # J2/J3 carry the load
    assert np.allclose(loaded_gravity, -1.3, atol=0.05)
    assert max(np.abs(r).std(axis=0).max() for r in residuals) < 0.1


def test_the_monitor_logs_by_default_and_stops_only_when_enforcing(tmp_path):
    spec = RobotSpec.from_yaml(YAM)
    path = tmp_path / "params.json"
    params(coef=[[1, 0, 0, 0, 0]] * DOF).save(path)
    for mode, stops in (("log", False), ("enforce", True)):
        cfg = with_contact_monitor({"type": "sim_watchdog"}, mode, params=str(path))
        cfg.pop("review")  # no second opinion: a contact in enforce mode stops at once
        monitor = build(SAFETY, cfg, spec=spec)
        assert isinstance(monitor, ContactMonitor)
        monitor.bind(phase_of=lambda: "above")
        monitor.log_dir = tmp_path / mode
        monitor.reset()
        q = np.array([0.0, 0.8, 0.8, 0.0, 0.0, 0.0])
        expected = monitor.model.predict(q, np.zeros(DOF), np.zeros(DOF))
        verdicts = []
        for step in range(40):
            eff = expected + (np.array([0, 5.0, 0, 0, 0, 0]) if step >= 30 else 0.0)
            obs = Observation(
                images={},
                joint_pos=q,
                gripper=0.0,
                timestamp=step / 28.0,
                extra={"motors": {"joint_eff": eff.tolist(), "joint_vel": [0.0] * DOF}},
            )
            verdicts.append(monitor.check(obs, step))
        tripped = [v for v in verdicts if v is not None]
        assert bool(tripped) is stops, mode
        if stops:
            assert tripped[0].outcome is Outcome.WATCHDOG and tripped[0].safe_stop
        assert monitor.trips, mode  # both modes detect; only enforce acts
        assert (tmp_path / mode / "contact_checks.jsonl").exists()


def test_the_monitor_ignores_robots_without_torque_readings(tmp_path):
    spec = RobotSpec.from_yaml(YAM)
    path = tmp_path / "params.json"
    params().save(path)
    monitor = ContactMonitor(spec, params=str(path), mode="enforce")
    monitor.reset()
    obs = Observation(images={}, joint_pos=np.zeros(DOF), gripper=0.0, timestamp=0.0)
    assert monitor.check(obs, 0) is None


def test_missing_parameters_say_how_to_fit_them(tmp_path):
    with pytest.raises(FileNotFoundError, match="fit_torque_model.py"):
        ContactMonitor(RobotSpec.from_yaml(YAM), params=str(tmp_path / "none.json"))


class FakeRig:
    """Records what the monitor sends; remembers one earlier command to back off to."""

    def __init__(self, q):
        self.q = q
        self.sent = []
        self.stopped = False
        self.earlier = Action(
            mode=ControlMode.JOINT_POSITION, values=np.zeros(DOF), gripper=0.0, timestamp=0.0
        )

    def command_before(self, seconds):
        self.asked = seconds
        return self.earlier

    def send_action(self, action):
        self.sent.append(action)

    def get_observation(self):
        return Observation(
            images={"wrist": np.zeros((4, 4, 3), np.uint8)},
            joint_pos=self.q,
            gripper=0.0,
            timestamp=0.0,
        )

    def safe_stop(self):
        self.stopped = True


def reviewer_saying(*decisions):
    calls = []

    def review(images, context):
        calls.append((sorted(images), context))
        decision, hazard = decisions[len(calls) - 1]
        return HazardReview(decision, hazard, 0.9, reason=f"fake {decision}", model="fake")

    review.calls = calls
    return review


def contact_run(monitor, q, push_from=30, steps=40):
    expected = monitor.model.predict(q, np.zeros(DOF), np.zeros(DOF))
    verdicts = []
    for step in range(steps):
        eff = expected + (np.array([0, 5.0, 0, 0, 0, 0]) if step >= push_from else 0.0)
        obs = Observation(
            images={"wrist": np.zeros((4, 4, 3), np.uint8)},
            joint_pos=q,
            gripper=0.0,
            timestamp=step / 28.0,
            extra={"motors": {"joint_eff": eff.tolist(), "joint_vel": [0.0] * DOF}},
        )
        verdicts.append(monitor.check(obs, step))
        if push_from <= step and (verdicts[-1] is not None or monitor.trips):
            break
    return verdicts


def enforced(tmp_path, reviewer):
    path = tmp_path / "params.json"
    params().save(path)
    monitor = ContactMonitor(
        RobotSpec.from_yaml(YAM),
        params=str(path),
        mode="enforce",
        review={"wait_s": 1.0},
        reviewer=reviewer,
        sleep=lambda s: None,
    )
    q = np.array([0.0, 0.8, 0.8, 0.0, 0.0, 0.0])
    rig = FakeRig(q)
    monitor.bind(robot=rig, phase_of=lambda: "above", context="Picking up the frosted cup.")
    monitor.reset()
    return monitor, rig, q


def test_a_contact_backs_off_then_claude_says_no_collision_and_the_pick_goes_on(tmp_path):
    review = reviewer_saying(("resume", False))
    monitor, rig, q = enforced(tmp_path, review)
    verdicts = contact_run(monitor, q)
    assert all(v is None for v in verdicts)  # carried on
    assert rig.sent[0] is rig.earlier  # backed off first
    glide = rig.sent[8:]  # then, on resume, glided back to where it stopped
    assert glide and np.allclose(glide[-1].values, rig.earlier.values)
    assert rig.asked == 1.0
    images, context = review.calls[0]
    assert "wrist (at the moment of contact)" in images and "wrist (now)" in images
    assert "motors" in context and "J2 (shoulder)" in context and "frosted cup" in context
    assert not rig.stopped


def test_claude_calling_it_a_collision_stops_the_arm(tmp_path):
    monitor, rig, q = enforced(tmp_path, reviewer_saying(("abort", True)))
    verdicts = contact_run(monitor, q)
    stop = [v for v in verdicts if v is not None]
    assert stop and stop[0].safe_stop and "Claude: fake abort" in stop[0].detail
    monitor.on_trip(rig, stop[0])
    assert rig.stopped


def test_claude_can_hold_the_arm_and_look_again(tmp_path):
    review = reviewer_saying(("wait", True), ("resume", False))
    monitor, rig, q = enforced(tmp_path, review)
    assert all(v is None for v in contact_run(monitor, q))
    assert len(review.calls) == 2
