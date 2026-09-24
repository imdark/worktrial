"""Startup capability negotiation (§4.3).

Runs once at construction, before step 0. A policy and a robot that disagree
about control mode, camera names, sensing or rate must refuse to start rather
than fail at step 1 with a KeyError -- or, worse, run degraded and quietly
produce bad numbers.

Errors are collected and reported together: discovering three mismatches one
run at a time is the slowest possible way to configure an experiment.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from teleop_sim.core.policy_spec import PolicySpec
from teleop_sim.core.spec import RobotSpec

#: Observation keys that exist only when the robot declares the sensor.
_SENSED_OBS_KEYS = ("joint_current", "joint_torque", "ee_wrench")

#: Safety-config keys and the sensing capability each one needs behind it.
_SAFETY_REQUIREMENTS = {
    "max_current_a": "joint_current",
    "max_torque_nm": "joint_torque",
    "max_wrench_n": "ee_wrench",
    "estop": "estop",
}

#: Fractional control-rate disagreement tolerated before warning.
HZ_TOLERANCE = 0.1


class Incompatible(ValueError):
    """A policy, robot, camera or safety configuration cannot work together."""


@dataclass
class CompatibilityReport:
    warnings: list[str] = field(default_factory=list)

    def log_lines(self) -> list[str]:
        return [f"compatibility warning: {w}" for w in self.warnings]


def check_compatibility(
    policy_spec: PolicySpec | None,
    robot_spec: RobotSpec,
    camera_config: dict[str, tuple[int, int]] | None = None,
    safety_config: dict[str, object] | None = None,
) -> CompatibilityReport:
    """Validate a policy/robot/camera/safety combination.

    Raises ``Incompatible`` listing every problem found. Returns warnings for
    conditions that are expected at some point in the plan -- notably a policy
    trained against an older RobotSpec, which is the normal state after the
    Stage 9 spec revision and exactly the signal you want in the log when
    evaluation numbers move.
    """
    errors: list[str] = []
    warnings: list[str] = []

    if camera_config is None:
        camera_config = {c.name: c.resolution for c in robot_spec.cameras}

    _check_safety(safety_config or {}, robot_spec, errors)

    if policy_spec is not None:
        _check_control_mode(policy_spec, robot_spec, errors)
        _check_cameras(policy_spec, camera_config, errors)
        _check_obs_keys(policy_spec, robot_spec, camera_config, errors)
        _check_rate(policy_spec, robot_spec, warnings)
        _check_spec_lineage(policy_spec, robot_spec, warnings)

    if errors:
        raise Incompatible(
            "incompatible configuration:\n  - " + "\n  - ".join(errors)
        )
    return CompatibilityReport(warnings=warnings)


def _check_control_mode(
    policy_spec: PolicySpec, robot_spec: RobotSpec, errors: list[str]
) -> None:
    if not robot_spec.supports(policy_spec.control_mode):
        errors.append(
            f"policy {policy_spec.policy_id!r} commands control mode "
            f"{policy_spec.control_mode.value!r}, but robot {robot_spec.name!r} "
            f"supports {[m.value for m in robot_spec.supported_modes]}"
        )


def _check_cameras(
    policy_spec: PolicySpec, camera_config: dict[str, tuple[int, int]], errors: list[str]
) -> None:
    for required in policy_spec.cameras:
        actual = camera_config.get(required.name)
        if actual is None:
            errors.append(
                f"policy {policy_spec.policy_id!r} needs camera {required.name!r}, "
                f"which is not configured; available: {sorted(camera_config)}. "
                "Camera names are a hard contract."
            )
        elif tuple(actual) != required.resolution:
            errors.append(
                f"camera {required.name!r} is {actual[0]}x{actual[1]} but policy "
                f"{policy_spec.policy_id!r} expects "
                f"{required.resolution[0]}x{required.resolution[1]}"
            )


def _check_obs_keys(
    policy_spec: PolicySpec,
    robot_spec: RobotSpec,
    camera_config: dict[str, tuple[int, int]],
    errors: list[str],
) -> None:
    for key in policy_spec.obs_keys:
        if key in _SENSED_OBS_KEYS and not robot_spec.sensing.provides(key):
            errors.append(
                f"policy {policy_spec.policy_id!r} consumes {key!r}, but robot "
                f"{robot_spec.name!r} declares sensing.{key}: false"
            )
        if key == "images" and not camera_config:
            errors.append(
                f"policy {policy_spec.policy_id!r} consumes images, but no camera "
                "is configured"
            )


def _check_rate(
    policy_spec: PolicySpec, robot_spec: RobotSpec, warnings: list[str]
) -> None:
    expected = policy_spec.expected_control_hz
    actual = robot_spec.control_hz
    if abs(expected - actual) > HZ_TOLERANCE * expected:
        warnings.append(
            f"policy {policy_spec.policy_id!r} expects {expected:g} Hz but robot "
            f"{robot_spec.name!r} runs at {actual:g} Hz; actions will be applied at "
            "a different cadence than they were trained at"
        )


def _check_spec_lineage(
    policy_spec: PolicySpec, robot_spec: RobotSpec, warnings: list[str]
) -> None:
    trained = policy_spec.trained_against_robot_spec
    if trained is None:
        return
    live = robot_spec.content_hash()
    if trained != live:
        warnings.append(
            f"policy {policy_spec.policy_id!r} was trained against RobotSpec "
            f"{trained}, live spec is {live}. Expected after a spec revision -- "
            "note it when comparing evaluation numbers."
        )


def _check_safety(
    safety_config: dict[str, object], robot_spec: RobotSpec, errors: list[str]
) -> None:
    for key, capability in _SAFETY_REQUIREMENTS.items():
        if safety_config.get(key) in (None, False):
            continue
        if not robot_spec.sensing.provides(capability):
            errors.append(
                f"safety config sets {key!r}, but robot {robot_spec.name!r} declares "
                f"sensing.{capability}: false -- that limit could never fire"
            )
