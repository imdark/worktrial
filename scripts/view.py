"""Watch the arm move under scripted joint targets.

    mjpython scripts/view.py      # macOS
    python   scripts/view.py      # Linux

On macOS the passive viewer must own the main thread, so it only runs under
`mjpython` (ships with MuJoCo); plain `python` fails there.

Opens the MuJoCo viewer and sweeps each joint through a slow sine, driving the
arm through the same Robot interface a policy would use -- not by writing qpos
directly, which would prove nothing about the driver.

Needs a display. The headless equivalent is tests/conformance/.
"""

from __future__ import annotations

import argparse
import math
import time

from teleop_sim.core.clock import WallClock
from teleop_sim.core.spec import RobotSpec
from teleop_sim.core.types import Action
from teleop_sim.robots.sim.mujoco_robot import MujocoRobot

DEFAULT_SPEC = "teleop_sim/robots/specs/yam.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", default=DEFAULT_SPEC)
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--amplitude", type=float, default=0.12, help="fraction of each range")
    args = parser.parse_args()

    import mujoco.viewer

    spec = RobotSpec.from_yaml(args.spec)
    clock = WallClock()
    robot = MujocoRobot(spec, clock, render_cameras=[])
    robot.connect()
    robot.reset(seed=0)

    home = spec.home_joints()
    span = (spec.upper_limits - spec.lower_limits) / 2.0 * args.amplitude
    period = 1.0 / spec.control_hz

    print(f"{spec.name}: {spec.dof} joints, {robot.substeps} substeps per control step")
    print("close the viewer window to stop")

    with mujoco.viewer.launch_passive(robot.model, robot.data) as viewer:
        start = clock.now()
        while viewer.is_running() and clock.now() - start < args.seconds:
            tick = clock.now()
            phase = 2 * math.pi * 0.2 * (tick - start)
            targets = spec.clip_joints(
                home + span * [math.sin(phase + i) for i in range(spec.dof)]
            )
            gripper = 0.5 * (1 + math.sin(phase * 2))

            robot.send_action(
                Action(
                    mode=spec.default_control_mode,
                    values=targets,
                    gripper=gripper,
                    timestamp=tick,
                    obs_timestamp=tick,
                )
            )
            viewer.sync()
            remaining = period - (clock.now() - tick)
            if remaining > 0:
                time.sleep(remaining)

    robot.disconnect()


if __name__ == "__main__":
    main()
