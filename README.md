# teleop-simulator

A hardware-agnostic pipeline for robot manipulation: teleoperated
demonstrations → imitation learning → **autonomous manipulation** → sim2real
transfer.

The simulated robot today is an **I2RT YAM** arm at a table with two water
glasses, a camera on top of its gripper and an overhead camera. It runs with
either the arm's stock gripper or **Kronos**, the production end effector,
modelled from its CAD for sim2real. The same code
drives a placeholder SO-101 arm, which is the point: a robot is a YAML spec
plus a driver, not a rewrite.

The first **sim-to-real** result is in: a Claude-guided pick / lift / place
that runs unchanged in MuJoCo and on a real YAM + Kronos arm driven through a
`robots_realtime` session. See [Sim-to-real](#sim-to-real-a-claude-guided-pick-on-a-real-yam).

> Architecture and the full 14-stage plan: [DESIGN.md](DESIGN.md).
> Why YAM, and what public data exists for it: [research/yam_research.md](research/yam_research.md).

## Status

| Stage | | |
|---|---|---|
| 0 — skeleton and contracts | ✅ | the seams, the control loop, startup compatibility check, conformance suite |
| 1 — simulated robot and scene | ✅ | MuJoCo YAM + glasses scene; composable arm / end effector / scene descriptions; Kronos production gripper |
| 2 — teleop loop, async inference | next | keyboard teleop, `AsyncPolicySource`, per-step staleness |
| VLM pick MVP (ahead of plan: Stages 9 + 12, first cut) | ✅ | Claude-guided pick / lift / place, same code in sim and on a real YAM + Kronos; 3 real picks, 2 successful |

```
pytest          294 passed, 11 skipped   (250 passed, 55 skipped without the Kronos assets)
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
| `vlm` | anthropic | Claude as the vision model of the pick task |
| `real-rr` | pyzmq, msgpack, msgpack-numpy | driving an arm through a running `robots_realtime` session |
| `real-feetech`, `realsense`, `vr` | hardware SDKs | real robots, cameras, VR teleop |

---

## Spawning the YAM + glasses environment

The robot is [`teleop_sim/robots/specs/yam.yaml`](teleop_sim/robots/specs/yam.yaml)
(the YAM arm + its stock gripper) and the cell is
[`teleop_sim/scenes/glasses_table.yaml`](teleop_sim/scenes/glasses_table.yaml).
They're separate descriptions, assembled at load time; see
[Robot descriptions](#robot-descriptions-arm--end-effector--scene).

### 1. Just look at it — no code

```bash
python scripts/export_model.py --out build/yam_glasses.xml
python -m mujoco.viewer --mjcf=build/yam_glasses.xml
```

The robot and the table are separate descriptions, so there's no single model
file on disk until you export one. `export_model.py` writes it (to `build/`,
which is gitignored; the file uses absolute paths, so it's machine-specific).
`--scene none` exports the bare robot. The second command opens MuJoCo's
interactive viewer. Drag to orbit, scroll to zoom, double-click
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
`--scene none` for the bare robot, `--spec teleop_sim/robots/specs/so101.yaml`
for the other arm.

### 3. Headless: no display needed

```bash
python scripts/snapshot.py --out yam.png
```

Renders a contact sheet: one row per camera (`top`, `wrist`), one column per
sampled step. Useful over SSH, in CI, or for checking camera framing.

### 4. From Python

```python
from teleop_sim.core.clock import WallClock
from teleop_sim.core.parts import SceneSpec
from teleop_sim.core.spec import RobotSpec
from teleop_sim.core.types import Action
from teleop_sim.robots.sim.mujoco_robot import MujocoRobot

robot_spec = RobotSpec.from_yaml("teleop_sim/robots/specs/yam.yaml")        # arm + gripper
spec = robot_spec.with_scene(SceneSpec.from_yaml("teleop_sim/scenes/glasses_table.yaml"))
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

### 5. Record a run: multi-view video + stills

```bash
python scripts/record_demo.py                       # YAM + Kronos (needs the Kronos assets)
python scripts/record_demo.py --spec teleop_sim/robots/specs/yam.yaml --grasp stock
```

Runs the scripted pick of `glass_right` and writes, to `outputs/<robot>_glass_pick/`
(gitignored), an MP4 with six views per frame (3/4, side, gripper close-up,
top-down, and the `top` and `wrist` cameras a policy sees), a still at the end of each
phase, and a contact sheet of the stills.

### 6. Through the control loop, from a config

```python
from teleop_sim.core.config import RunConfig, build_system

system = build_system(RunConfig.from_yaml("teleop_sim/configs/yam_glasses.yaml"))
result = system.loop.run_episode(seed=0)
print(result.outcome, result.steps, result.robot_spec_hash)
```

[`yam_glasses.yaml`](teleop_sim/configs/yam_glasses.yaml) places the YAM in the
glasses table (`robot.scene`) and wires it to a trivial hold-position policy
with the safety watchdog on. Swapping in a teleop
device or a trained policy is an edit to its `source:` section, not a code
change.

### What's in the scene

| | |
|---|---|
| **Arm** | I2RT YAM, 6 DoF, from MuJoCo Menagerie (MIT). Home pose `[0, 1.047, 1.047, 0, 0, 0]` rad. |
| **Gripper** | `yam_linear` (`yam.yaml`): the stock gripper Menagerie ships, a parallel two-finger jaw with ~76 mm opening. Or `kronos` (`yam_kronos.yaml`): the production end effector, [below](#the-production-end-effector-kronos). |
| **`wrist` camera** | D405-sized (42×42×23 mm) on **top of the gripper**, looking past the fingertips. 58° vertical FOV. |
| **`top` camera** | Overhead workspace view, named to match the public YAM datasets. 55° vertical FOV. |
| **Glasses** | `glass_left` and `glass_right`: 65 mm across, 100 mm tall, 199 g, 45 mm apart, 0.40 m in front of the arm. |
| **Physics** | 1/600 s timestep, so 20 substeps per 30 Hz control step; elliptic friction cones so a held glass doesn't slide out. |

A scripted side grasp lifts `glass_right` 100 mm. That's a test
(`tests/test_yam_scene.py`), so the scene is checked to be usable for the
task, not just to render. How that trajectory is generated is in
[`teleop_sim/envs/scripted_grasp.py`](teleop_sim/envs/scripted_grasp.py).

### Changing the scene

- **Move or resize the glasses, or move the wrist camera:** edit
  [`teleop_sim/envs/yam_assets.py`](teleop_sim/envs/yam_assets.py), then run
  `python scripts/build_yam_assets.py`. Don't hand-edit the generated files
  (`yam_arm.xml`, `yam_linear.xml`, `glasses.xml`); a test fails if they drift
  from the generator.
- **Table, lighting, overhead camera, physics options:** edit
  [`assets/scenes/glasses_table/scene.xml`](assets/scenes/glasses_table/scene.xml).
- **Never edit `assets/i2rt_yam/upstream/`.** It's the vendored Menagerie model,
  checksum-verified by a test. See [VENDOR.md](assets/i2rt_yam/VENDOR.md).
- **After moving a camera, update its pose in the description that owns it**:
  the gripper's for the wrist camera, the scene's for `top`.
  `tests/test_mjcf_spec.py` checks that descriptions and models agree.

---

## The production end effector: Kronos

[`robots/specs/yam_kronos.yaml`](teleop_sim/robots/specs/yam_kronos.yaml) is the
YAM with Kronos in place of the stock gripper: two Dynamixel XL430-W250-T servos
driving two pivoting fingers through 1:1 meshed gear hubs, with TPU pads and grip
tape, and a camera on a neck above the fingers.

**Its model files are not in this repository.** They are generated from the
proprietary CAD (the Hardware Archive), which stays private, and this repository
is public. To build them locally, see
[`assets/end_effectors/kronos/README.md`](assets/end_effectors/kronos/README.md).
Until you do, `yam_kronos` won't load and every Kronos test skips.

```
Hardware Archive.zip ─ scripts/build_kronos_assets.py (gmsh) ─▶ meshes/*.stl + kronos_extract.json
kronos_extract.json ── scripts/build_kronos_model.py ────────▶ kronos.xml
```

The first step needs the CAD and gmsh (GPL, build-time only). The second is plain
Python ([`teleop_sim/envs/kronos_assets.py`](teleop_sim/envs/kronos_assets.py)),
so the model can be regenerated and reviewed without either.

| | Value | Source |
|---|---|---|
| Geometry, pivots (31.0 mm apart), CoM, inertia | from the STEP assembly | CAD, measured |
| Mass | housing 258.7 g, each finger 38.0 g | slicer projects + servo datasheet |
| Finger motion | exact mirror (gear coupling); 28.0° from open to closed, video shows 25.5 ± 3.5° | video + STL + CAD |
| "Open" (gripper 0.0) | the widest pose in the teleop video, 18.85° in from the CAD pose | video + STL |
| "Closed" (gripper 1.0) | where the pads meet, 46.8°, measured from the pad geometry | computed |
| Grip torque limit | 2.8 N·m (2 × XL430 stall at 11.1 V) | datasheet, **voltage assumed** |
| Pad friction 1.2, shell 0.6 | TPU + grip tape vs PETG-CF | **assumed** |
| Wrist camera | RealSense D405 size, 58° FOV | **model assumed** |
| Mount on the YAM flange | orientation read off the video | **inferred, needs confirming** |

`tests/test_kronos.py` pins each measured value, so the model can't drift from
the hardware evidence without a failing test. The same scripted side grasp (with
Kronos-specific offsets, `KRONOS_GRASP`) lifts `glass_right` about 100 mm.

---

## Robot descriptions: arm + end effector + scene

Each physical thing has its own description, and a robot is a combination of
them:

| Part | Description | Model | Holds |
|---|---|---|---|
| **Arm** | [`robots/arms/yam.yaml`](teleop_sim/robots/arms/yam.yaml) | `assets/i2rt_yam/yam_arm.xml` | joints, limits, home, control rate, the `flange` body an end effector bolts to |
| **End effector** | [`yam_linear.yaml`](teleop_sim/robots/end_effectors/yam_linear.yaml), [`kronos.yaml`](teleop_sim/robots/end_effectors/kronos.yaml) | `assets/end_effectors/<name>/` | gripper calibration, grasp point, mount pose on the flange, the cameras it carries |
| **Scene** | [`scenes/glasses_table.yaml`](teleop_sim/scenes/glasses_table.yaml) | `assets/scenes/glasses_table/` | table, objects, fixed cameras, workspace, physics options |
| **Robot** | [`yam.yaml`](teleop_sim/robots/specs/yam.yaml), [`yam_kronos.yaml`](teleop_sim/robots/specs/yam_kronos.yaml) | — | which arm + which end effector |

```yaml
# robots/specs/yam.yaml: the whole robot description
name: yam
arm: ../arms/yam.yaml
end_effector: ../end_effectors/yam_linear.yaml
```

- **Swapping the gripper is that one `end_effector:` line.** The arm and scene
  are untouched, and so is everything downstream: loop, policies, recorder.
- **The real robot uses the same arm + end effector with no scene.** The scene
  is chosen per run (`robot.scene` in a run config), because the glasses aren't
  part of the robot.
- **Parts are assembled with MuJoCo's own composition API** (`MjSpec`), in
  [`robots/sim/assembly.py`](teleop_sim/robots/sim/assembly.py).
- **Physics options come from the scene.** A part asking for a different
  integrator or timestep is refused; MuJoCo would only warn.
- **Every part's model files, meshes included, feed the robot's content hash.**
  Data recorded with one gripper can't pass for another's.

Splitting Menagerie's single YAM file into arm + stock gripper is tested to be
lossless: same joints, actuators and per-body masses, and identical grasp-point
kinematics. The added mass is the 60 g wrist camera, which is intentional. One
thing couldn't be split: Menagerie fuses the wrist motor and the stock
gripper's mounting plate into one mesh, so that thin plate stays drawn on the
arm.

### Adding an end effector

1. Put its meshes in `assets/end_effectors/<name>/`, in metres, one file per
   moving part, each under 200,000 faces (MuJoCo's STL limit).
2. Write `<name>.xml`: one root body holding the gripper's bodies, joints,
   collision shapes, grasp site and any camera, plus its actuator and finger
   coupling. **Prefix every material, mesh and default class with `<name>_`**,
   or composition stops on a name clash.
3. Write `robots/end_effectors/<name>.yaml`: mjcf path, root body, mount pose
   on the flange, gripper joint/actuator/travel/deadband, grasp site, cameras.
4. Point a robot description at it. The tests then check it:
   - it passes `tests/conformance/`;
   - its description agrees with its model;
   - it touches nothing on the arm, open or closed.

`tests/assets/test_jaw/` is a minimal worked example: a second gripper on the
same YAM arm that passes the full conformance suite.

---

## Sim-to-real: a Claude-guided pick on a real YAM

The task: find a cup or glass with a vision model, pick it up, lift it, verify
the grasp, put it back where it was, and return. One policy, one control loop,
one config switch between simulation and the real arm.

### Architecture

```
          ┌──────────────── Claude (Anthropic API) ─────────────────┐
          │ Opus 5.5 · plan: which object does the instruction mean? │  once per episode
          │ Sonnet 5 · locate: base / rim pixels of that object      │  every view
          │ Sonnet 5 · verify: is it held?  (unsure → Opus 5.5)      │  after lift
          └───────────────▲──────────────────────────┬───────────────┘
                          │ images                    │ pixels, never metres
┌─────────────────────────┴──────────────────────────▼──────────────────────┐
│ PickLiftPlacePolicy (tasks/pick_lift_place.py) — a Policy like any other  │
│   survey view → plan → locate → closer view → locate → grasp waypoints    │
│   pixel → ray → table plane (perception/camera.py; glass depth is useless)│
│   → IK on the same RobotSpec model (robots/kinematics.py) → joint chunks  │
│   jaw reading + model → held?  arrival check + gravity-sag compensation   │
└──────────────────────────────────┬────────────────────────────────────────┘
                     ControlLoop (unchanged) · PolicySource · TaskEvent ends the episode
                                   │
            robot.type: mujoco ────┴──── robot.type: robots_realtime
        MujocoRobot (sim)                 RobotsRealtimeRobot (robots/real/rr_bridge.py)
        OracleVision = ground truth         ZMQ bus of a running rr-session:
                                            hud/teleop/<arm>/joint_pos  →  arm
                                            <arm>/joint_state, <cam>/rgb →  obs
                                            (rr keeps its own safety layers, gravity comp,
                                             gripper logic; silence = hold position)
```

Design decisions that matter:

- **The model never commands the arm.** Claude answers in pixels. Geometry turns
  a pixel into a table position: the camera ray through the point where the
  glass meets the table, intersected with the table plane. A transparent glass
  gives stereo depth nothing to measure, and the table reads fine.
- **Model routing.** Frequent narrow questions go to the fast model at low
  effort, rare scene reasoning to the smart model at high effort, and a
  low-confidence fast answer is asked again of the smart model. Answers come
  back through structured outputs (a JSON schema), not forced tool use.
- **Physics before vision for "is it held?".** Jaws that closed all the way
  hold nothing (1.00 in sim); jaws that stopped part-way hold something (0.12
  in sim, 0.75–0.78 on the real cup). The model is asked only in the second case, and only a
  confident "no" overrides the jaws. A camera-only check failed in sim: Sonnet
  saw the other glass on the table and called a held glass "not held".
- **Never let go in the air.** A failed grasp is put back down before opening.
- **Slow is safe.** A vision call blocks the loop while the arm stands still
  between segments. `robots_realtime` holds position once commands pause for
  0.5 s, and ramps back in from the measured pose when they resume, so pauses,
  and link drops of up to 90 s, cost only time.

### The sim-to-real ladder

Each rung runs everything the next one will, minus one risk:

| Rung | Command | What it proves |
|---|---|---|
| 1. sim, oracle vision | `run_pick.py configs/yam_kronos_pick_sim.yaml --vision oracle --fast` | geometry, IK, motion, loop (no API key) |
| 2. sim, Claude | `run_pick.py configs/yam_kronos_pick_sim.yaml` | prompts, parsing, routing, verification |
| 3. real config vs a sim rig | `python -m teleop_sim.robots.real.sim_rig` + `run_pick.py configs/yam_kronos_pick_real.yaml --host 127.0.0.1` | the bridge and the bus protocol, without the arm |
| 4. real, read-only | `... --probe` | link, clock skew, joint + gripper conventions, cameras (sends nothing) |
| 5. real, look only | `... --look-only` | perception on the real cameras; arm never near the object |
| 6. real, approach only | `... --approach-only` | reach and alignment at the standoff pose, no contact |
| 7. real, full | `... ` (asks y/N before every motion unless `--no-confirm`) | the task |

`tests/test_pick_task.py` runs rungs 1 and 3 in CI (Claude stubbed).

### Optional: laya-vision for "is it held?" (experimental, non-commercial)

[laya-vision](https://huggingface.co/thaitea/laya-vision) is a 201M model that
answers typed yes/no questions about an image in one forward pass: ~110 ms on
an M5, against 2.5–4.8 s for Sonnet on `verify`. It cannot answer `locate`
(no pixel output), so `plan` and `locate` stay with the fallback backend.

```bash
bash scripts/setup_laya_vision.sh                                  # once: own venv, pinned weights
third_party/laya-vision/.venv/bin/python scripts/laya_server.py    # leave running
python scripts/run_pick.py teleop_sim/configs/yam_kronos_pick_sim.yaml --vision laya-oracle --fast
python scripts/run_pick.py teleop_sim/configs/yam_kronos_pick_real.yaml --vision laya   # falls back to Claude
```

Zero-shot it may only *confirm* a grasp; any other answer goes to the
fallback. Its "no" is not trustworthy yet: it said P(held) = 0.17 about a
glass held in sim. It runs in its own venv because it needs
`transformers >= 5.3`, which lerobot 0.3.2 rejects. **The weights are
CC BY-NC-SA 4.0 — not for commercial use.**

#### Fine-tuning it on the rig's own frames

`scripts/collect_real_picks.py` records every episode's wrist frames with a
label from the gripper, not from a model (`teleop_sim/telemetry.py`: jaws
closed and stopped short → held; jaws open → not held; jaws moving →
uncertain, dropped). Then:

```bash
python scripts/export_laya_dataset.py runs/collect_<ts> --val 8 9 --test 10   # -> data/laya/gem13_held
PY=third_party/laya-vision/.venv/bin/python
$PY scripts/train_laya_verify.py --eval-only                     # zero-shot baseline
$PY scripts/train_laya_verify.py --steps 300 --freeze last_n     # fine-tune, calibrate, save
$PY scripts/laya_server.py --model third_party/laya-vision-ft/<run>/checkpoint
```

Splits are by episode. On the first batch (10 real picks, 1,149 frames),
zero-shot laya-vision scored AUROC 0.83 on val / test and could not separate
the cup sitting between open fingers (reach-in, P(held) 0.58) from a held cup;
a 20-step smoke fine-tune reached 0.885 / 0.884. That batch has **no hard
negatives** (jaws closed on nothing), so a fine-tuned model may learn
"closed jaws = held": collect deliberate misses before trusting its "no".

### Optional: vision safety loop (opt-in, `--hazard-monitor`)

Off unless asked for. With `--hazard-monitor` on `run_pick.py` or
`collect_real_picks.py`, the run's safety monitor is wrapped in
`vision_hazard` (`teleop_sim/control/safety/vision_hazard.py`, settings in
`teleop_sim/configs/safety/vision_hazard.yaml`):

1. In a background thread, laya-vision asks a few times a second whether a
   person's hand or arm is in the wrist or overhead view (~150 ms per answer).
2. One confident "yes" does nothing. **Three in a row (~1 s) pause the arm**:
   the loop stops commanding and the rig holds its pose.
3. **Claude decides.** Sonnet 5 sees the frame that raised the alarm and
   fresh frames from both cameras, then answers *resume* (false alarm or
   hazard gone; the motion continues where it stopped), *wait* (hold, look
   again in 3 s) or *abort* (safe stop, episode over). A resume it is unsure
   of goes to Opus 5.5.
4. It fails closed. No answer from Claude, still unclear after 60 s, more
   than 6 alarms in an episode, or a laya server that stops answering for
   2 s all stop the arm. A run with no laya server refuses to start.

```bash
third_party/laya-vision/.venv/bin/python scripts/laya_server.py    # leave running
python scripts/run_pick.py teleop_sim/configs/yam_kronos_pick_real.yaml --hazard-monitor
python scripts/eval_hazard_monitor.py runs/collect_<ts>            # false alarms on recorded runs
python scripts/watch_hazards.py teleop_sim/configs/yam_kronos_pick_real.yaml \
    --host <rig> --camera-port 5557 --label hazard --note "hand near gripper"   # arm untouched
```

Measured zero-shot on the 10-episode clean batch: **0 false alarms in 12.9
min**. Claude cleared staged false alarms on real frames in 2.5–3.4 s. Only
*person* questions ship. Asking "any object other than the cup in the path?"
misfires on 87–99 % of clean frames: zero-shot, the model cannot tell the
target cup, the spare cup or the gripper from an obstacle. Object checks
need fine-tuning on frames recorded with `watch_hazards.py --label`.
Detection rate is **not yet measured**: no recorded frame contains a hazard.
This supplements the e-stop and robots_realtime's limits; it does not
replace them.

### Running on a robots_realtime rig

1. On the rig, the operator launches the session that owns CAN, the motors and
   the cameras (it needs sudo), e.g. `./scripts/launch_gem13.sh`.
2. Reach its bus. A direct Ethernet link to the rig here is IPv6 link-local
   only, which ZMQ will not use, and it drops every minute or so, so tunnel:
   ```bash
   ssh -N -L 15555:127.0.0.1:5555 -L 15556:127.0.0.1:5556 pantheon@<rig>.local
   ```
3. Set `ANTHROPIC_API_KEY` in the environment (never in a file), then climb
   rungs 4–7 with `--host 127.0.0.1 --ports 15555 15556`.

Every run writes `runs/<timestamp>/`: `events.jsonl`, `calls.jsonl` (each model
call with its image, answer, latency, tokens and request id) and the annotated
images. That log is the raw material for evaluation.

### What the real rig taught us (2026-09-24, YAM + Kronos, frosted tapered cup)

- **Bus protocol**, read from `robots_realtime`: XSUB/XPUB broker on 5555/5556,
  `[topic, msgpack({ts, src, data})]` with msgpack-numpy arrays, sender-clock
  timestamps checked against a 0.5 s command timeout (the bridge measures skew),
  and a normalised gripper with **1.0 = open**, the reverse of this repo's.
- **Kinematics agree with the vendor's**: identical joint signs and zeros; link
  geometry differs from i2rt's `yam.xml` by ~6 mm mean / ~12 mm worst at the
  flange. Targets seen by the wrist camera and reached by the same model mostly
  cancel it.
- **Gravity sag**: 2.5–6° below command on the rig's fleet config.
  `sag_compensation` walks the command off by the measured shortfall; the arm
  then arrives within ~1° and a few mm.
- **Tool pitch**: with the joints exactly at a level pose, the wrist camera and
  the operator put the real tool ~2° more nose-down than the model predicts
  (the Kronos mount is inferred, not measured). `grasp.pitch_trim_deg: 2.0`.
- **Grasp geometry**: 45° nose-down, gripping 2 cm past the centre toppled the tapered
  cup. Near-level and centred works: 20° at 50 mm (run 2) and, gripping higher,
  10° at 80 mm with the 2° trim (run 3). Level (0°) is unreachable that low
  that far out; lower hits the table with the Kronos body.
- **Localization**: survey and closer-view estimates agree to 7–23 mm.
  A full real pick takes 75–90 s.

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
| `teleop_sim/control/` | the one control loop, action sources, compatibility check, latency, safety (`sim_watchdog`, opt-in `vision_hazard`) |
| `teleop_sim/robots/sim/` | `MujocoRobot` |
| `teleop_sim/robots/real/` | `RobotsRealtimeRobot` (bridge to a running `robots_realtime` session), its wire format, and `SimRig`, a MuJoCo rig speaking the same protocol |
| `teleop_sim/robots/kinematics.py` | FK / IK on a RobotSpec's model, off the physics state (shared by sim and real) |
| `teleop_sim/perception/` | pinhole geometry, the `Vision` interface, `ClaudeVision` (Opus 5.5 / Sonnet 5), `OracleVision` (sim ground truth) |
| `teleop_sim/tasks/` | `PickLiftPlacePolicy`, the VLM-guided pick / lift / place |
| `teleop_sim/runlog.py` | per-run events, model calls and images |
| `teleop_sim/core/parts.py` | arm / end effector / scene descriptions and how they compose |
| `teleop_sim/robots/arms/`, `end_effectors/` | part descriptions |
| `teleop_sim/robots/specs/` | robots: `yam.yaml`, `yam_kronos.yaml` (compositions), `so101.yaml` (monolithic) |
| `teleop_sim/robots/sim/assembly.py` | builds one MuJoCo model from the parts |
| `teleop_sim/scenes/` | scene descriptions |
| `teleop_sim/envs/` | scene handles, task composition, asset generators (YAM, Kronos), the scripted grasp |
| `teleop_sim/configs/` | run configs: `yam_glasses`, `yam_kronos_pick_sim`, `yam_kronos_pick_real`, `sim_policy` (SO-101), fake configs |
| `assets/i2rt_yam/` | vendored YAM (`upstream/`, untouched) and the generated arm |
| `assets/end_effectors/` | end-effector models: `yam_linear/` (generated from upstream), `kronos/` (generated locally, not tracked) |
| `assets/scenes/` | scene models (`glasses_table/`, `cup_table/` with the real rig's tapered cup) |
| `assets/so101/` | placeholder SO-101 model and scene |
| `scripts/` | tasks: `run_pick.py`; viewing: `view.py`, `snapshot.py`, `record_demo.py`, `export_model.py`; asset builds: `build_yam_assets.py`, `build_kronos_assets.py`, `build_kronos_model.py` |
| `tests/conformance/` | the shared contract every `Robot` implementation must pass |
| `tests/fakes.py` | every seam with no physics, hardware or rendering |
| `research/` | background: the YAM survey, the hierarchical-VLA thesis |

## Tests

```bash
pytest                                # everything
pytest tests/test_yam_scene.py        # the glasses environment, including the grasp
pytest tests/conformance/             # the Robot contract: fake, SO-101, YAM + each gripper
pytest tests/test_composition.py      # arm / end effector / scene composition
pytest tests/test_kronos.py           # Kronos against its hardware evidence (skips without its assets)
pytest tests/test_pick_task.py        # the VLM pick: geometry, sim, the real bridge against a sim rig
pytest tests/test_vision_hazard.py    # the vision safety loop: debounce, pause, Claude's decisions (fake models)
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

- **The VLM pick is an MVP.** Three real runs, two successes, one object. The
  table height, cup dimensions and Kronos jaw opening are estimates, the grasp
  was tuned by hand at the rig, and the loop blocks during vision calls
  (`AsyncPolicySource` is still Stage 2). The bridge drives one arm; the core
  types are single-arm by design.
- **Sim and the rig disagree** on the Kronos jaw opening (the real one grips a
  cup the model says is too wide) and on tool pitch (~2°). Both belong in the
  Kronos description once measured.

- **The grasp works but isn't robust yet.** The stock gripper squeezes only
  ~1.5 N per side. With Kronos, 12 of 16 swept grasp variants lift the glass but
  only 3 keep it within 10° of upright. The robust scripted expert is Stage 4.
- **The arm's position servos sag under load** (~0.05 rad at joints 2–3 holding a
  glass). To hold a pose, command the planned pose, not the measured one;
  commanding where the arm sagged to makes it sag again.
- **Kronos's mount orientation, camera model and servo voltage are assumptions**
  (see [the table](#the-production-end-effector-kronos)) until checked on the hardware.
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
