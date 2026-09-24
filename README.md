# teleop-simulator

A hardware-agnostic pipeline for robot manipulation: teleoperated
demonstrations → imitation learning → **autonomous manipulation** → sim2real
transfer.

The simulated robot today is an **I2RT YAM** arm at a table with two water
glasses, a camera on top of its gripper and an overhead camera. The same code
drives a placeholder SO-101 arm, which is the point: a robot is a YAML spec
plus a driver, not a rewrite.

> Architecture and the full 14-stage plan: [DESIGN.md](DESIGN.md).
> Why YAM, and what public data exists for it: [YAM_RESEARCH.md](YAM_RESEARCH.md).

## Status

| Stage | | |
|---|---|---|
| 0 — skeleton and contracts | ✅ | the seams, the control loop, startup compatibility check, conformance suite |
| 1 — simulated robot and scene | ✅ | MuJoCo YAM + glasses scene; SO-101 as a second arm |
| 2 — teleop loop, async inference | next | keyboard teleop, `AsyncPolicySource`, per-step staleness |

```
pytest          180 passed, 4 skipped
ruff check .    clean
```

---

## Setup

Tested on macOS (Apple Silicon) with Python 3.12. Linux works the same way.

```bash
# 1. git (macOS: Apple's Command Line Tools)
xcode-select --install

# 2. uv, which installs Python 3.12 without touching the system Python
curl -LsSf https://astral.sh/uv/install.sh | sh

# 3. environment + dependencies (from the repo root)
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e ".[dev,sim]"

# 4. check everything works
pytest
```

The Command Line Tools also ship a Python 3.9. Don't use it: it is past end
of life and too old for MuJoCo 3, LeRobot and current PyTorch.

Optional extras, so machines only install what they use:

| Extra | Pulls in | Needed for |
|---|---|---|
| `dev` | pytest, ruff | tests, lint |
| `sim` | MuJoCo, mink | the YAM / SO-101 simulation |
| `train` | torch, LeRobot | training (Stage 5) |
| `real-feetech`, `realsense`, `vr` | hardware SDKs | real robots, cameras, VR teleop |

---

## Spawning the YAM + glasses environment

The scene is [`assets/i2rt_yam/glasses_scene.xml`](assets/i2rt_yam/glasses_scene.xml);
the robot spec that describes it is
[`teleop_sim/robots/specs/yam.yaml`](teleop_sim/robots/specs/yam.yaml).

### 1. Just look at it — no code

```bash
python -m mujoco.viewer --mjcf=assets/i2rt_yam/glasses_scene.xml
```

Opens MuJoCo's interactive viewer. Drag to orbit, scroll to zoom, double-click
a body and Ctrl-drag to push it around. The *Control* panel on the right has a
slider per actuator (`joint1`–`joint6`, `gripper`) if you want to move the arm
by hand.

The viewer starts from MuJoCo's default joint positions, not from the spec's
home pose. Use one of the options below to see the arm as the code drives it.

### 2. Watch it move through the `Robot` interface

```bash
mjpython scripts/view.py            # macOS
python   scripts/view.py            # Linux
```

Drives the arm through a slow sweep around its home pose via `MujocoRobot`,
the same interface a policy uses. On macOS the viewer must own the main thread,
so MuJoCo's passive viewer only runs under `mjpython`, which ships with
MuJoCo. Plain `python` fails there.

Options: `--seconds 60`, `--amplitude 0.2` (fraction of each joint's range),
`--spec teleop_sim/robots/specs/so101.yaml` for the other arm.

### 3. Headless: no display needed

```bash
python scripts/snapshot.py --out yam.png
```

Renders a contact sheet: one row per camera (`top`, `wrist`), one column per
sampled step. Useful over SSH, in CI, or for checking camera framing.

### 4. From Python

```python
from teleop_sim.core.clock import WallClock
from teleop_sim.core.spec import RobotSpec
from teleop_sim.core.types import Action
from teleop_sim.robots.sim.mujoco_robot import MujocoRobot

spec = RobotSpec.from_yaml("teleop_sim/robots/specs/yam.yaml")
robot = MujocoRobot(spec, WallClock(), image_size=(224, 224))
obs = robot.reset(seed=0)           # arm at home, gripper open, glasses on the table

obs.images["wrist"]                 # (224, 224, 3) uint8, camera on top of the gripper
obs.images["top"]                   # (224, 224, 3) uint8, overhead
obs.joint_pos                       # 6 joint angles, radians
obs.gripper                         # 0.0 = open, 1.0 = closed
obs.ee_pose                         # grasp point: xyz + wxyz quaternion

# one control step (1/30 s): hold the home pose, start closing the gripper
robot.send_action(Action(mode=spec.default_control_mode, values=spec.home_joints(),
                         gripper=1.0, timestamp=0.0))
```

### 5. Through the control loop, from a config

```python
from teleop_sim.core.config import RunConfig, build_system

system = build_system(RunConfig.from_yaml("teleop_sim/configs/yam_glasses.yaml"))
result = system.loop.run_episode(seed=0)
print(result.outcome, result.steps, result.robot_spec_hash)
```

[`yam_glasses.yaml`](teleop_sim/configs/yam_glasses.yaml) wires the YAM to a
trivial hold-position policy with the safety watchdog on. Swapping in a teleop
device or a trained policy is an edit to its `source:` section, not a code
change.

### What's in the scene

| | |
|---|---|
| **Arm** | I2RT YAM, 6 DoF, from MuJoCo Menagerie (MIT), unmodified. Home pose `[0, 1.047, 1.047, 0, 0, 0]` rad. |
| **Gripper** | Parallel two-finger jaw; opens to ~76 mm between the pads. One actuator drives both fingers. |
| **`wrist` camera** | D405-sized (42×42×23 mm) on **top of the gripper**, looking past the fingertips. 58° vertical FOV. |
| **`top` camera** | Overhead workspace view, named to match the public YAM datasets. 55° vertical FOV. |
| **Glasses** | `glass_left` and `glass_right`: 65 mm across, 100 mm tall, 199 g, 45 mm apart, 0.40 m in front of the arm. |
| **Physics** | 1/600 s timestep, so 20 substeps per 30 Hz control step; elliptic friction cones so a held glass doesn't slide out. |

A scripted side grasp lifts `glass_right` 100 mm. That's a test
(`tests/test_yam_scene.py`), so the scene is checked to be usable for the
task, not just to render. How that trajectory is generated is in
[`tests/yam_grasp.py`](tests/yam_grasp.py).

### Changing the scene

- **Move or resize the glasses, or move the wrist camera:** edit
  [`teleop_sim/envs/yam_assets.py`](teleop_sim/envs/yam_assets.py), then run
  `python scripts/build_yam_assets.py`. Don't hand-edit `yam_follower.xml` or
  `glasses.xml`; a test fails if they drift from the generator.
- **Table, lighting, overhead camera, physics options:** edit
  `glasses_scene.xml` directly.
- **Never edit `assets/i2rt_yam/upstream/`.** It's the vendored Menagerie model,
  checksum-verified by a test. See [VENDOR.md](assets/i2rt_yam/VENDOR.md).
- **After moving a camera, update its pose in `yam.yaml` too.**
  `tests/test_mjcf_spec.py` checks that the spec and the model agree.

---

## How it fits together

Eight seams get an interface: `Robot`, `Teleoperator`, `Retargeter`, `Camera`,
`Policy`, `SuccessDetector`, `ResetStrategy`, `SafetyMonitor`. Everything else
is concrete.

- **Hardware is data.** A `RobotSpec` YAML holds joints, limits, gripper
  calibration, cameras, sensing and control rate. It drives both the simulated
  robot and the real driver.
- **A human and a policy look the same to the loop.** Both are `ActionSource`s,
  so autonomy is a config change:
  ```yaml
  source: {type: teleop, device: {type: fake}, retarget: {type: identity}}   # human drives
  source: {type: policy, policy: {type: constant}}                            # no human
  ```
- **Mismatches fail at startup, not at step 1.** A policy that wants a camera,
  control mode or sensor this robot doesn't have refuses to start, listing
  every problem at once.
- **Every episode is stamped** with a content hash of the robot spec and model
  files, the policy spec and the code version, so datasets recorded against
  different robot revisions can't silently mix.

## Layout

| Path | What |
|---|---|
| `teleop_sim/core/` | canonical types, robot and policy specs, protocols, registry, clock, hashing |
| `teleop_sim/builtins.py` | populates the registries; heavy backends are declared, not imported |
| `teleop_sim/control/` | the one control loop, action sources, compatibility check, latency, safety |
| `teleop_sim/robots/sim/` | `MujocoRobot` |
| `teleop_sim/robots/specs/` | `yam.yaml`, `so101.yaml` |
| `teleop_sim/envs/` | scene handles, task composition, the YAM asset generator |
| `teleop_sim/configs/` | run configs: `yam_glasses`, `sim_policy` (SO-101), fake configs |
| `assets/i2rt_yam/` | vendored YAM (`upstream/`, untouched), generated camera and glasses, the scene |
| `assets/so101/` | placeholder SO-101 model and scene |
| `scripts/` | `view.py`, `snapshot.py`, `build_yam_assets.py` |
| `tests/conformance/` | the shared contract every `Robot` implementation must pass |
| `tests/fakes.py` | every seam with no physics, hardware or rendering |

## Tests

```bash
pytest                                # everything
pytest tests/test_yam_scene.py        # the glasses environment, including the grasp
pytest tests/conformance/             # the Robot contract, for fake, SO-101 and YAM
```

Two conventions that matter when adding code:

- **Registration must not depend on import order.** Registries are filled by
  `teleop_sim/builtins.py`, and heavy backends are declared, not imported. CI
  runs every test file in its own process to catch violations.
- **Every new `Robot` must pass `tests/conformance/`.** It checks radians in and
  out, a gripper that reads exactly 0.0 and 1.0 at its extremes, clipped
  commands, and honest control modes and sensing. A unit bug in a driver looks
  exactly like bad policy performance.

## Known limitations

- **The grasp works but isn't robust yet.** The simulated gripper squeezes only
  ~1.5 N per side, and some grasp variants tip the glass. The robust scripted
  expert is Stage 4.
- **A hard strike on the table can jam YAM's fingers** until the next reset.
  This comes from the upstream model.
- **The glasses are empty.** MuJoCo doesn't simulate liquid.
- **Sensing is declared off** (no torque or current) until checked against
  I2RT's driver.
- **The SO-101 model is a placeholder.** Its link lengths are guesses; YAM's
  are I2RT's own.

## Licensing

The YAM model in `assets/i2rt_yam/upstream/` is MIT-licensed, © 2025 i2rt
robotics; see its `LICENSE`. No license has been chosen yet for this
repository's own code.
