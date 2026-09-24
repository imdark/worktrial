# Teleop Simulator — Design Document

**Status:** Draft v3.1 · 2026-09-24
**Scope:** End-to-end pipeline: teleoperated demonstration collection → imitation learning → **autonomous manipulation** → sim2real transfer, with a two-level (slow planner / fast controller) policy as the end state.

> **v3 changes.** Seam list re-derived and made honest (§2). Asynchronous / multi-rate inference made a first-class loop concern (§3.4). `PolicySpec` and startup capability negotiation added (§4.3). Force/current sensing added to `Observation` and declared in `RobotSpec` (§4.1). `Task` decomposed into `SuccessDetector` + `ResetStrategy` + concrete scene code (§3.3). Dataset provenance stamping added (§4.5). Interface conformance suite added (§5.1). Full list in §11.
>
> **v3.1 changes.** Concurrency model made explicit (§3.5): which thread owns what, threads-not-processes and why, and the 30–100 Hz question. Per-stream image capture timestamps added (§4.1). Capture concurrency is per-component and config-driven, because threaded capture costs bit-exact replay (§6).

---

## 1. Objectives and constraints

### Objectives

1. Collect teleoperation demonstrations in **simulation and on real hardware through one code path**.
2. Train visuomotor policies and evaluate them in sim against reproducible success metrics.
3. **Execute manipulation autonomously** — no human in the control loop — in sim and then on hardware, including autonomous reset, failure detection, and retry.
4. Transfer trained policies to a real arm. **Sim2real is a hard requirement**, deferred in schedule but not in design.
5. Support a hierarchical policy: a slow high-level planner (~1–5 Hz) driving a fast low-level controller (~30–100 Hz).

Teleoperation is the **data-acquisition mechanism and the fallback**, not the end product. The end product is objective 3.

### Constraints

| Constraint | Consequence |
|---|---|
| Target hardware not yet identified | Robot definition must be **data**, not code. Swapping arms = new spec file + driver. |
| Primary development on macOS / Apple Silicon | MuJoCo is the simulator. Isaac Sim/Lab is out unless a Linux+NVIDIA box appears. |
| Training on Mac initially, GPU later | Device must be a config field. Datasets must be portable (HF Hub). |
| Leader-arm teleop first, VR/SpaceMouse later | A retargeting layer must exist from day one, even if only one backend is implemented. |
| Autonomous operation removes the human safety layer | Watchdog, timeout, stall detection, and workspace limits are infrastructure, not polish. |
| Objective 5 mixes a ~2 Hz planner with a ~50 Hz controller | Inference **cannot** be assumed to fit inside a control tick. Asynchrony is infrastructure, not an optimization (§3.4). |

### Non-goals for v1

Bimanual manipulation · mobile bases · RL from scratch · photorealistic rendering · multi-robot coordination · dexterous (>2 finger) hands · explicit 6-DoF pose-estimation module (see §8).

These are excluded to keep the first working loop small.

**What that costs us, stated honestly.** Most of these are additive — a mobile base is a new `Robot`, more tasks are new `SuccessDetector`s, RL is a new `Policy`. Two are **not**:

- **Bimanual** and **dexterous hands** are precluded by the core types, deliberately. `Observation.joint_pos` is a flat array and `Observation.gripper` is a scalar (§4.1). Adding a second arm or a 5-finger hand is a **breaking revision of `Observation` and `Action`**, and therefore of every dataset recorded before it.
- The alternative — keying every leaf by limb name (`joint_pos: dict[str, ndarray]`) — taxes every line of code we write for two years to buy an option we have explicitly deferred. We are not paying that tax.

This is a priced decision, not an oversight. If bimanual becomes real, budget a core-type migration and a dataset re-write, and see §4.5 on why the spec hash makes that survivable. *(This replaces the v2 claim that none of the non-goals were precluded by the architecture, which was false.)*

---

## 2. Design principles

1. **A seam must be earned, and the list is closed.** An interface exists only where we will genuinely swap an implementation, and only where we can already name the second implementation. Eight seams qualify (§2.1). Everything else is written concretely. Adding a ninth requires naming the two implementations that force it.
2. **Hardware as data.** A `RobotSpec` YAML (joints, limits, end-effector, gripper calibration, camera mounts, **sensing capabilities**, control rate) drives both the simulated robot and the real driver. One source of truth means sim and real cannot silently diverge.
3. **Canonical units at every boundary.** Radians for joints. Gripper normalized to `[0, 1]`. Monotonic timestamps on every sample. Native units (servo ticks, degrees, metres of finger travel) exist only inside a driver. **This is enforced by a conformance suite (§5.1), not by convention.**
4. **One control loop — one code path, not one thread.** Sim teleop, real teleop, autonomous sim rollout, and autonomous real execution are the same function with different objects passed in, so bugs get fixed once. Sensor capture, inference, and disk writes run *off* that thread (§3.5); the loop thread's own work stays small enough to be boring.
5. **A human and a policy are the same thing to the loop.** Both are `ActionSource`s (§3.2). Autonomy is therefore a config change, not a second code path — and shared control is a third `ActionSource` rather than a special case.
6. **Sim and real implement the same interface.** `MujocoRobot` and `FeetechRobot` are interchangeable. So is `ZMQRobotClient`, which is how a cloud GPU drives a local arm with zero code change.
7. **The loop owns time; nothing else may block it.** A source that cannot answer within a tick returns its freshest answer and reports staleness. Inference latency is measured and logged, never absorbed silently (§3.4).
8. **Concrete until it hurts.** Do not abstract the physics engine, do not build a plugin system, do not generalize the reward system before there are two tasks. Over-abstraction is the more likely failure here than under-abstraction.
9. **Record what was commanded, not what was requested.** `send_action` returns the clipped/actually-sent action, and that is what gets logged. Otherwise the dataset teaches the policy to command targets the robot will never reach.
10. **Every autonomous episode terminates on its own.** Success, failure, timeout, or watchdog trip — never "until someone notices." This is what makes unattended data generation and batch evaluation possible.
11. **Compatibility fails at construction, not at step 1.** A policy and a robot that disagree about control mode, camera names, or rate must refuse to start (§4.3).

### 2.1 The seam list

v2 claimed six seams and then quietly swapped implementations at two more. The honest list is eight. Each row names the second implementation that earned it.

| # | Seam | Implementations | Earned by |
|---|---|---|---|
| 1 | `Robot` | mujoco · feetech/dynamixel · zmq_client | sim vs. real vs. remote |
| 2 | `Teleoperator` | keyboard · leader_arm · spacemouse · vr_quest | Stage 2 vs. Stage 8 |
| 3 | `Retargeter` | joint_map · ik · delta | leader arm emits joints, VR emits poses |
| 4 | `Camera` | sim · opencv · realsense | sim vs. real |
| 5 | `Policy` | scripted · act · remote · hierarchical · world_model | Stage 4 vs. 5 vs. 12 vs. 13 |
| 6 | `SuccessDetector` | sim_state · learned · fixture · human | **sim reads privileged state; hardware cannot** |
| 7 | `ResetStrategy` | sim_rerandomize · scripted_replace · fixture · human_batch | Stage 6 vs. Stage 11 |
| 8 | `SafetyMonitor` | sim_watchdog · hardware_safety | Stage 6 vs. Stage 11 (force/current/e-stop) |

**Explicitly not seams:**

- `ActionSource` — an *adapter*, not a seam. It is a thin union over `Teleoperator + Retargeter` and `Policy` so one loop serves all four modes (§3.2).
- `Task` — a *concrete composition* of a `SuccessDetector`, a `ResetStrategy`, and scene-setup code. It is a convenience object, not an interface.
- `Scene` / `randomization` — sim-only, concrete. There is no real-hardware counterpart, and pretending there is would be the exact over-abstraction principle 8 warns about.
- `Recorder` — one implementation plus one LeRobot adapter module, per the dependency-isolation risk in §9.
- The physics engine — see principle 8.

---

## 3. High-level architecture

```
      scripts/  teleop_record · train · eval · run_auto · replay · calibrate
                            │
                  ┌─────────▼──────────┐
                  │  control/loop.py   │  one loop, all modes · owns the clock
                  └─┬──────┬───────┬───┘
      ┌─────────────┘      │       └──────────────┬──────────────┐
 ┌────▼────┐    ┌──────────▼─────┐        ┌───────▼───────┐ ┌────▼─────┐
 │  Robot  │    │  ActionSource  │        │ SafetyMonitor │ │ Recorder │
 └────┬────┘    └──┬──────────┬──┘        └───────────────┘ └──────────┘
      │            │          │            timeout · stall ·
 sim│real│remote   │          │            workspace · force
                   │          │
     ┌─────────────┘          │
     │                 ┌──────▼────────────────────┐
┌────▼─────────┐       │  PolicySource             │  owns chunk buffer
│ TeleopSource │       │  AsyncPolicySource  (§3.4)│  owns worker thread
└────┬─────────┘       └──────┬────────────────────┘  reports staleness
     │                        │
┌────▼─────────┐  ┌──────────┐│  ┌────────┐
│ Teleoperator │─▶│Retargeter ││─▶│ Policy │ scripted│act│remote│hierarchical
└──────────────┘  └──────────┘   └────────┘

 ┌────────┐  ┌─────────────────┐  ┌───────────────┐  ┌───────────────┐
 │ Camera │  │ SuccessDetector │  │ ResetStrategy │  │ Scene + rand. │
 └────────┘  └─────────────────┘  └───────────────┘  └───────────────┘
 sim│usb│rs   sim_state│learned    sim│scripted│      concrete,
              fixture│human        fixture│human      sim only
```

### 3.1 Component responsibilities

| Component | Owns | Does **not** own |
|---|---|---|
| `Robot` | Hardware/sim I/O, unit conversion, joint clipping, safety limits, declaring `supported_modes` and sensing capabilities | Task logic, rendering policy, recording |
| `Teleoperator` | Reading a human input device, button state | Any knowledge of the target robot |
| `Retargeter` | Mapping a teleop command onto a specific robot's action space (constructed with that robot's `RobotSpec`) | Device I/O, robot I/O |
| `Camera` | Producing frames + intrinsics | Resizing for a policy (that is the policy's transform) |
| `Policy` | Observation → **action chunk**. Declares its requirements via `PolicySpec` | Timing, threading, buffering, recording, resets |
| `PolicySource` | Chunk buffering and replay, action provenance tagging | Inference |
| `AsyncPolicySource` | Worker thread, freshest-chunk selection, staleness reporting | Inference, timing policy |
| `SuccessDetector` | Deciding whether the episode's goal was achieved | Why it failed, scene setup |
| `ResetStrategy` | Returning the world to a start state between episodes | Success, control |
| `SafetyMonitor` | Episode termination and safe-stop: timeout, stall, out-of-workspace, force/current limit, e-stop | Deciding *why* it failed beyond a coarse label |
| `Recorder` | Serializing steps + metadata + provenance hashes | Everything else |
| `control/loop.py` | Rate limiting, wiring, latency and staleness accounting, event dispatch, startup compatibility check | Any component-specific logic |

The `Retargeter` seam is what makes VR a later config change rather than a refactor. The leader arm emits joint targets and uses an affine `JointMapRetargeter`; a VR headset emits an end-effector pose and uses an `IKRetargeter`. The loop, the robot, and the recorder are unaffected by which one is active.

### 3.2 Operating modes

The loop does not distinguish a human from a policy. Both satisfy:

```python
class ActionSource(ABC):
    def reset(self) -> None: ...
    def get_action(self, obs: Observation) -> Action: ...   # MUST NOT block past a tick
```

| Mode | `ActionSource` | Human role | Introduced |
|---|---|---|---|
| **Teleoperated** | `TeleopSource` = Teleoperator + Retargeter | drives every step | Stage 2 |
| **Scripted autonomous** | `PolicySource(ScriptedPolicy)` | none | Stage 4 |
| **Autonomous** | `AsyncPolicySource(LearnedPolicy)` | supervises, e-stop only | Stage 6 |
| **Shared / intervention** | `SharedSource` — policy acts, human pre-empts | corrects on failure | Stage 7 |

`ActionSource` is a thin union over the `Teleoperator + Retargeter` pair and `Policy`. It is not a ninth seam — it is the adapter that lets one loop serve all four modes. Shared autonomy falls out of it almost free, and it is the mechanism that makes intervention data (Stage 7) a recording concern rather than a new subsystem.

### 3.3 Why `Task` was decomposed

v2 gave a single `Task` object four responsibilities: scene randomization, reset, success detection, and autonomous reset. Those four split cleanly along the sim/real line, and bundling them hid a missing work item:

| Responsibility | In sim | On hardware |
|---|---|---|
| Scene randomization | `envs/randomization.py` | **does not exist** — a human places objects |
| Success detection | read privileged `mj_data` | **no privileged state** — needs a learned classifier, an instrumented fixture, or a human |
| Reset | teleport + re-randomize | physical: fixture, scripted re-place, or human between batches |

Stage 11's exit criterion ("N consecutive autonomous episodes on hardware") is unmeasurable without a real `SuccessDetector`, and v2 never built one. It is now an explicit seam (§2.1 row 6) with a scheduled implementation (Stage 11a). `Task` survives as a concrete composition object that bundles the three for a given experiment; it carries no interface of its own.

### 3.4 Asynchronous inference (multi-rate execution)

**Problem.** v2's `Policy.get_action(obs) -> Action` is a blocking call at the loop rate, and three things in this plan violate it: the Stage 12 VLM (~200–500 ms at 1–5 Hz), `remote_policy` over ZMQ (network RTT), and Stage 13 latent planning. v2 hid this inside "`Policy` owns action chunking and buffering," which pushes threading into every policy implementation and leaves the loop unable to say how stale the action it just applied was.

**Resolution.** Buffering and concurrency move out of `Policy` and into the source, where there is exactly one implementation of each.

```python
class Policy(ABC):
    spec: PolicySpec
    def reset(self) -> None: ...
    def predict(self, obs: Observation) -> ActionChunk: ...   # may be slow; called off-thread

ActionChunk: actions: list[Action]
             obs_timestamp: float        # the obs this chunk was computed from
             horizon_hz: float           # rate the chunk's actions are spaced at
```

- `PolicySource` — synchronous. Calls `predict` when its buffer empties. Correct for `ScriptedPolicy` and for small ACT configs that fit in a tick.
- `AsyncPolicySource` — runs `predict` on a worker thread, serves the freshest buffered action every tick, and stamps `Action.obs_timestamp`. The loop computes `staleness_ms = now - action.obs_timestamp` and logs it on every step. Underrun (buffer exhausted before the next chunk lands) is an explicit, counted event, and its fallback — hold last action, or decay toward it — is a config field, not a silent behavior.

One wrapper covers the VLM, the remote policy, any slow diffusion/VLA checkpoint, and Stage 13. **The hierarchical policy of objective 5 is then just two sources at two rates**, the slow one feeding the fast one's conditioning, rather than a special case in the loop.

`staleness_ms` is recorded per step from Stage 2. It is the single most useful number for diagnosing a policy that works in eval and fails live, and for the Stage 10 sim2real latency model.

### 3.5 Concurrency model — which thread owns what

Principle 4 says *one control loop*. That is one **code path**, not one thread, and the distinction starts to matter the moment real hardware is attached.

**The arithmetic.** At 30 Hz the tick budget is 33.3 ms. Run sequentially on the hardware path:

| Step | Rough order | Nature |
|---|---|---|
| 2 × USB camera `read()` | 10–30 ms | **blocking wait** for the next frame |
| Feetech sync-read, 6 servos @ 1 Mbaud | 2–5 ms | serial I/O |
| ACT inference on MPS | 10–30 ms | native compute |
| Servo write | 1–2 ms | serial I/O |
| Recorder append + video encode | 2–20 ms, spiky | disk |

These are order-of-magnitude estimates to size the problem, not measurements — Stage 2's latency logging produces the real ones. The shape is not in doubt: roughly 50 ms sequentially, so ~18 Hz with jitter, and the single largest item is a camera call that spends its time doing nothing but waiting.

**Off the loop thread** — anything that waits on I/O or takes unbounded time:

| Work | Mechanism | Hand-off |
|---|---|---|
| Camera capture | one grabber thread per camera | single-slot latest-frame buffer, stale frames dropped |
| Policy inference | `AsyncPolicySource` (§3.4) | chunk buffer |
| Recording, video encode | writer thread + subprocess encoder | bounded queue; a full queue is a counted event, never a silent block |
| Leader-arm polling | device thread, once it is a serial device | single-slot latest-command buffer |

**On the loop thread, always.** The joint read inside `get_observation()` and `send_action()` — both fast, and their ordering is the determinism worth protecting. The `SafetyMonitor` check, because it *gates* the action: a watchdog that trips 40 ms late on another thread is not a watchdog. The clock and the rate limiter. Target ≤5 ms of loop-thread work, with everything else a read from a slot someone else filled.

**Threads, not processes — and the GIL is not the obstacle.** The GIL serializes Python bytecode, and almost nothing expensive here is Python bytecode: camera grabs, serial I/O, torch inference and MuJoCo stepping are native calls that release it while they run. (Worth verifying per binding rather than assuming, but that is the shape.) Processes earn their place in exactly one part of this project, and it is not the control loop: **Stage 6's batch evaluation** is embarrassingly parallel *across episodes* — N worker processes, each running an ordinary sequential loop. Latency hiding and throughput are different problems; do not reach for the same mechanism twice.

**No asyncio.** The libraries here are blocking C calls, not awaitables. Wrapping them in `run_in_executor` is threads again, bought at the price of colouring the whole codebase `async`.

**The 30–100 Hz question.** Objective 5's upper bound leaves a 10 ms budget, where CPython on macOS — GC pauses, no real-time scheduling — is genuinely marginal. We largely dodge it by construction: with `JOINT_POSITION` as the default action space (§4.2), the servos close their own PID loop internally at kHz rates and Python is a **setpoint generator, not a servo controller**. 30–50 Hz of setpoints is sufficient. If a real >100 Hz software loop is ever needed, that is a signal to push it into the driver or the firmware, not to optimize Python. Cheap insurance meanwhile: preallocated arrays, `gc.freeze()` after setup, and `gc.disable()` during an episode with a manual collect between episodes.

**Concurrency is per-component and config-driven, because determinism has a price.** Threaded capture destroys bit-exact replay, and Stage 3's exit criterion depends on replay. So simulation defaults to synchronous capture and hardware to threaded (§6). The loop code is identical either way — the threads live *inside* components behind the `Camera`, `ActionSource` and `Recorder` seams, which is exactly what those seams are for.

**Sequence this work by measurement, not by guess.** Build Stage 2 sequential, read the per-stage latency breakdown it already logs, then thread the largest item. Expected order: cameras, then policy, then recorder.

---

## 4. Core contracts

Condensed; full definitions land in `teleop_sim/core/`.

### 4.1 Observation, Action, and friends

```python
Observation:  images: dict[str, ndarray]   # name -> HxWx3 uint8
              image_timestamps: dict[str, float]   # per-stream CAPTURE time
              joint_pos: ndarray           # radians
              gripper: float               # 0 = open, 1 = closed
              timestamp: float             # monotonic, when obs was assembled
              joint_vel: ndarray | None
              joint_current: ndarray | None  # amps    — None when unsensed
              joint_torque: ndarray | None   # Nm      — None when unsensed
              ee_wrench: ndarray | None      # 6-vector — None when unsensed
              ee_pose, extra               # optional / sim ground truth

Action:       mode: ControlMode            # joint_position | joint_velocity
                                           # | ee_pose_abs | ee_pose_delta
              values: ndarray
              gripper: float
              timestamp: float             # when the action was issued
              obs_timestamp: float         # obs it was computed from -> staleness
              source: ActionOrigin         # enum, see below

ActionOrigin: HUMAN_TELEOP | SCRIPTED | POLICY | POLICY_HIGH_LEVEL
              | POLICY_LOW_LEVEL | HUMAN_CORRECTION
              # v2 used a 2-valued string; Stage 7 needs base-vs-correction
              # and Stage 12 needs which level emitted the action.

TeleopCommand: kind: "joint" | "ee_pose" | "ee_delta"
               values, gripper
               buttons: dict[str, bool]    # record · reset · discard · estop · takeover
               timestamp: float

EpisodeResult: outcome: "success" | "failure" | "timeout" | "watchdog" | "aborted"
               failure_tag: str | None     # e.g. "missed_grasp", "dropped", "stalled"
               steps, duration, seed
               intervention_frac: float    # 0.0 for fully autonomous
               robot_spec_hash: str        # provenance, see §4.5
               policy_spec_hash: str | None
               code_version: str
               staleness_ms: {p50, p95, max}
               underruns: int
```

**Per-stream capture timestamps are the fix for a hole that opens the moment capture goes threaded.** A single scalar `timestamp` is only honest while capture is synchronous. Once cameras run on their own grabber threads (§3.5), each frame carries its own capture time, lagging the joint read by a variable 10–30 ms. Stamped at capture and recorded, that offset is visible and modellable — and it feeds the Stage 10 latency model. Collapsed into one scalar, it becomes an image/state misalignment that jitters frame to frame and looks exactly like a bad policy.

**Sensing fields are the fix for a v2 hole:** `SafetyMonitor` was given ownership of a "force limit" while `Observation` carried nothing to read. Force and current are now first-class and explicitly nullable, because most position-controlled hobby servos cannot report them.

**Morphology assumption, stated once, deliberately.** These types encode *one kinematic chain with one scalar-DoF end effector*. That is a priced v1 decision (§1, Non-goals). Bimanual and dexterous hands require revising `Observation` and `Action`; the `robot_spec_hash` in every episode (§4.5) is what makes that migration detectable rather than silent.

### 4.2 RobotSpec

```python
RobotSpec:    name, version, mjcf_path, urdf_path
              joints: [JointSpec]          # name, limits, vel/effort limits
              ee_link, gripper_joint, gripper_range
              cameras: [CameraSpec]        # name, mount, pose, fov, resolution
              workspace_bounds             # autonomous safety envelope
              default_control_mode, supported_modes: [ControlMode]
              control_hz
              sensing: {joint_current: bool, joint_torque: bool,
                        ee_wrench: bool, estop: bool}
```

`supported_modes` and `sensing` are new. Between them they let the startup check (§4.3) reject two whole classes of silent failure: a policy commanding a mode the arm cannot execute, and a configured force limit with no sensor behind it.

**Action space decision.** Default to `JOINT_POSITION` for position-controlled servos. It removes IK mismatch and controller-gain mismatch, the two largest avoidable sim2real gaps. Use `EE_POSE_DELTA` only for arms with a well-characterized Cartesian impedance controller.

### 4.3 PolicySpec and startup capability negotiation

`RobotSpec` made hardware a data contract. v2 had no counterpart for policies, so a checkpoint's requirements — camera *names*, resolutions, control mode, rate — were implicit, and a mismatch surfaced as a `KeyError` at best or quiet degradation at worst. That is the central risk in "swap the model."

```python
PolicySpec:   policy_id, checkpoint_hash
              control_mode: ControlMode
              cameras: [{name, height, width}]   # names are a hard contract
              obs_keys: [str]                    # which Observation fields are consumed
              obs_history: int
              action_horizon: int                # chunk length
              expected_control_hz: float
              language_conditioned: bool
              trained_against_robot_spec: str    # RobotSpec hash, if known
```

Written into every checkpoint at train time; read at load time. `control/compat.py` runs **once at construction, before step 0**:

```python
check_compatibility(policy.spec, robot.spec, camera_config) -> None | raises Incompatible
```

It asserts: the policy's `control_mode` is in `robot.spec.supported_modes`; every camera the policy names exists in the camera config at the resolution it expects; every `obs_key` is actually populated by this robot (a policy consuming `joint_current` on an arm with `sensing.joint_current: false` is a hard error); and `expected_control_hz` is within tolerance of `robot.spec.control_hz`, warning otherwise. A `trained_against_robot_spec` that differs from the live spec hash is a **warning**, not an error — it is the expected state after the Stage 9 spec revision, and it is exactly the signal you want in the log when eval numbers move.

The safety config is checked here too: a configured force limit against `sensing.joint_torque: false` fails loudly rather than never firing.

### 4.4 Rate and staleness contract

The loop owns the clock (principle 7). Every step records `dt`, `inference_ms`, `staleness_ms` (action age, §3.4), per-stream **sensor age** (`now - image_timestamps[name]`), and `underrun` flags for both the action buffer and any camera slot that served a repeated frame. A source or sensor that consistently underruns is a configuration error the harness reports, not a mystery. `max_staleness_ms` and `cameras.max_age_ms` are config bounds; on hardware, exceeding them is a `SafetyMonitor` trip rather than a log line.

### 4.5 Dataset provenance

§9 predicts one `RobotSpec` revision at Stage 9. Datasets recorded before it stay on disk and stay trainable — and become subtly wrong in a way that degrades a policy rather than raising. Therefore:

Every episode records **the full `RobotSpec` plus its content hash**, the `PolicySpec` hash where a policy produced the data, and the code version. `scripts/train.py` refuses to silently mix spec hashes: it prints the distribution and requires an explicit `--allow-mixed-specs`. Cheap now; forensically painful otherwise.

---

## 5. Repository layout

```
teleop_sim/
  core/       types · spec · policy_spec · protocols · registry · clock · hashing
  robots/     sim/mujoco_robot · real/{feetech,dynamixel} · remote/{zmq_client,zmq_server}
              specs/*.yaml
  teleop/     leader_arm · keyboard · spacemouse · vr_quest
  retarget/   joint_map · ik · delta
  cameras/    sim_cam · opencv_cam · realsense_cam · async_capture   # grabber thread + slot
  envs/       scene · randomization · tasks/pick_place       # sim-only, concrete
  success/    sim_state · learned · fixture · human          # seam 6
  reset/      sim_rerandomize · scripted_replace · human_batch  # seam 7
  policies/   scripted · act · remote_policy · hierarchical
  control/    loop · rate · latency · compat · concurrency · safety/{sim_watchdog,hardware_safety}
              sources/{teleop_source,policy_source,async_policy_source,shared_source}
  eval/       harness · failures · report
  data/       recorder · lerobot_adapter · replay
  configs/    *.yaml
scripts/      teleop_record · run_auto · train · eval · replay · calibrate
tests/        fakes · conformance/ · test_loop · test_retarget · test_safety · test_compat
assets/       <robot>/ · objects/ · textures/
```

`success/` and `reset/` are lifted out of `envs/` because `envs/` reads as sim-only and both have real-hardware implementations (§3.3). `control/safety/` replaces the single `watchdog.py` for the same reason. `cameras/async_capture.py` and `control/concurrency.py` hold every thread in the system (§3.5) — one place to look when a rate goes wrong.

### 5.1 The conformance suite

`tests/fakes.py` provides `FakeRobot` / `FakeTeleop` implementing the protocols with no physics and no hardware, so CI runs anywhere and catches interface drift early.

`tests/conformance/` is new and is the highest-leverage artifact for hardware modularity. It is a **shared parameterized suite that every `Robot` implementation must pass** — `FakeRobot`, `MujocoRobot`, `ZMQRobotClient` in CI, and the real driver on a bench:

- joint positions are radians in and radians out, within limits
- gripper reads exactly `0.0` and `1.0` at both physical extremes, monotonic between
- `send_action` returns the **clipped** action, and an out-of-limit command clips rather than raising
- timestamps are monotonic and never duplicated
- `supported_modes` is honest: every declared mode is accepted, every undeclared one is rejected
- fields declared `false` in `spec.sensing` are `None` in the observation, and vice versa

Principle 3 was stated in v2 but enforced nowhere. A unit bug in a new driver costs a week and looks exactly like bad policy performance. Equivalent (smaller) suites exist for `Camera`, `Retargeter`, and `SuccessDetector`.

---

## 6. Configuration model

A dict registry (`@register(ROBOTS, "mujoco")`) plus a typed config file. No plugin/entrypoint machinery.

```yaml
robot:    {type: mujoco,     spec: robots/specs/<arm>.yaml, render_cameras: [wrist, front]}
source:   {type: teleop,     device: leader_arm, port: /dev/tty.usbmodemXXXX,
           retarget: {type: joint_map, calibration: calib/leader_to_follower.json}}
success:  {type: sim_state,  tolerance_m: 0.03}
reset:    {type: sim_rerandomize, xy_range: 0.12, yaw: true}
task:     {type: pick_place, objects: [cube_red]}
safety:   {type: sim_watchdog, max_steps: 600, stall_secs: 3.0, enforce_workspace: true}
cameras:  {async: false}          # synchronous capture — bit-exact replay (§3.5)
record:   {fps: 30, root: data/pickplace_sim_v1}
```

Autonomous, with a slow remote policy:

```yaml
source:   {type: async_policy, policy: {type: remote, endpoint: tcp://gpu:5555},
           on_underrun: hold_last, max_staleness_ms: 150}
safety:   {type: hardware_safety, max_current_a: 1.8, enforce_workspace: true,
           estop: /dev/tty.estop, on_trip: safe_stop}
cameras:  {async: true, max_age_ms: 40}   # grabber threads, stale frames dropped
record:   {fps: 30, queue_depth: 120, encoder: subprocess}
```

Switching from teleoperated to autonomous is `source.type: teleop` → `source.type: policy` (or `async_policy`) plus a checkpoint path. Switching from sim to hardware is `robot.type: mujoco` → `robot.type: feetech`, and — because §3.3 separated them — `success.type` and `reset.type` change with it. Neither touches code. The startup check (§4.3) then validates the combination before anything moves.

**Dependency extras** (`pip install -e ".[sim,train]"`) keep RealSense, CUDA, and VR dependencies off machines that do not need them.

---

## 7. Implementation stages

Four milestones, fifteen stages. Each stage has an exit criterion that can be demonstrated, not just claimed.

The narrative arc: **teleoperate in sim → run autonomously in sim → teleoperate on hardware → run autonomously on hardware → add hierarchy.**

### Milestone A — Teleoperated loop in simulation

#### Stage 0 · Skeleton and contracts
**Goal.** Establish the seams before any behavior exists.
**Builds.** `core/` (types, spec, policy_spec, protocols, registry, clock, hashing), `control/compat.py`, `tests/fakes.py`, `tests/conformance/`, `pyproject.toml` with extras, CI running tests on macOS.
**Exit.** `pytest` passes; a `FakeRobot` and `FakeTeleop` can be constructed from config and stepped 100 times; **`FakeRobot` passes the full conformance suite**; `check_compatibility` rejects a deliberately mismatched `PolicySpec`.
**Note.** Write the `IKRetargeter`, `AsyncPolicySource`, and hardware-`SuccessDetector` stubs here, raising `NotImplementedError`. They force the interfaces to be honest about VR, asynchrony, and real-hardware success detection before any of the three exists.

#### Stage 1 · Simulated robot and scene
**Goal.** A MuJoCo arm, table, gripper, and cameras, driven through the `Robot` interface.
**Builds.** `robots/sim/mujoco_robot.py`, `cameras/sim_cam.py`, `robots/specs/<arm>.yaml`, `assets/`, `envs/scene.py`.
**Exit.** Scripted joint targets move the arm in the viewer; `get_observation()` returns correctly-shaped images from every camera declared in the spec; **`MujocoRobot` passes the conformance suite**, including gripper normalization at both extremes.
**Risk.** Camera FOV and mount pose are guesses until real hardware exists. Keep them in the spec file so correcting them later is a one-line edit.

#### Stage 2 · Control loop, action sources, and asynchrony
**Goal.** A human can drive the simulated arm, through the abstraction that will later carry slow policies.
**Builds.** `control/loop.py`, `control/rate.py`, `control/latency.py`, `control/concurrency.py`, `control/sources/{teleop_source,policy_source,async_policy_source}`, `teleop/keyboard.py`, `retarget/joint_map.py`.
**Exit.** Keyboard teleop moves the arm at a stable 30 Hz *via a* `TeleopSource`; swapping in a trivial constant-action `PolicySource` runs the identical loop with no human attached; **a deliberately slow (100 ms) policy behind `AsyncPolicySource` holds 30 Hz with staleness and underruns logged per step**; rate jitter and a **per-stage latency breakdown** (capture · joint read · source · send · record) are logged.
**Note.** The two swaps *are* the exit criterion. If either needs loop changes, the abstraction is wrong, and it is far cheaper to fix here than at Stage 6 (autonomy) or Stage 12 (the VLM), by which point trained policies depend on the old timing. Build the loop sequential and thread nothing yet: the per-stage breakdown is what decides *what* to thread, and guessing wastes the measurement (§3.5).

#### Stage 3 · Recording, replay, inspection
**Goal.** Demonstrations become a dataset.
**Builds.** `data/recorder.py`, `data/lerobot_adapter.py`, `data/replay.py`, provenance stamping (§4.5).
**Exit.** Record 10 episodes; replay them open-loop into a fresh sim and observe substantially the same trajectory; dataset loads with the LeRobot dataset API; **every episode carries its `RobotSpec` and hash, and a deliberate spec edit makes the hash change**; a visualizer shows synchronized camera streams, joint traces, staleness, and sensor age.
**Note.** Replay determinism requires `cameras.async: false` (§3.5). That is why capture concurrency is a per-component config field rather than a global mode.
**Note.** Adopt the LeRobot dataset format here and do not deviate. The adapter is roughly 80 lines and unlocks their training code, their pretrained checkpoints, and Hub hosting.

---

### Milestone B — Autonomous manipulation in simulation

This milestone delivers a robot that performs the task by itself, before any hardware is involved.

#### Stage 4 · Scripted autonomous expert
**Goal.** The first fully autonomous agent, and a way to debug the pipeline without a human in the loop.
**Builds.** `policies/scripted.py` (IK-based reach → grasp → lift → place), `envs/tasks/pick_place.py`, `success/sim_state.py`, `reset/sim_rerandomize.py`, `envs/randomization.py`.
**Exit.** 500 episodes generated **unattended** with >90% scripted success across randomized object poses; failure modes categorized.
**Note.** This is the stage teams skip and regret. It validates observation shapes, timing, recording, and success detection before demonstration time is spent — and it is the first proof that the autonomous path works end to end.

#### Stage 5 · Learned policy
**Goal.** Replace the scripted expert with a learned one.
**Builds.** `policies/act.py` (wrapping LeRobot's ACT), `scripts/train.py` (emits `PolicySpec` into the checkpoint), single-episode `scripts/eval.py`.
**Exit.** ACT trained on scripted data completes the task from pixels on held-out object poses; a single evaluation episode is reproducible from a seed; **loading the checkpoint against a robot with a renamed camera fails at construction with a readable error.**
**Note.** `device: mps` trains small ACT configurations on the Mac well enough to validate the pipeline. Push the dataset to the Hub so a GPU box reproduces the run exactly.

#### Stage 6 · Autonomous runtime
**Goal.** Unattended operation — the system runs, judges itself, resets itself, and stops itself.
**Builds.** `control/safety/sim_watchdog.py` (timeout, stall detection, workspace violation), `reset/` wired for between-episode auto-reset, `eval/harness.py` (batch N-trial evaluation on fixed seeds), `eval/failures.py` (failure taxonomy), `eval/report.py`, `scripts/run_auto.py`.
**Exit.** `run_auto.py` executes 100 consecutive episodes with **zero human input**, self-resetting between them, and emits a report with success rate, per-failure-mode counts, confidence intervals, and staleness/underrun distributions. No episode ends by a human noticing it is stuck.
**Note.** This is the stage that converts "the policy sometimes works" into a measurable number, and it is a prerequisite for every comparison made later (Stages 10, 12, 13). Retry-on-failure behavior is in scope; recovery *policies* are not.

#### Stage 7 · Shared autonomy and intervention
**Goal.** The human becomes a supervisor rather than a driver, and corrections become training data.
**Builds.** `control/sources/shared_source.py` (policy acts by default, teleop pre-empts on a takeover button), intervention logging (`ActionOrigin.HUMAN_CORRECTION`, `EpisodeResult.intervention_frac`), a DAgger-style retraining loop.
**Exit.** A human can take over mid-episode and hand back; interventions are recorded with correct provenance and distinguishable from base human demonstrations; a policy retrained on base data + interventions measurably improves on the Stage 6 harness.
**Note.** This is the highest-leverage data-collection mode in the whole project — corrections are collected exactly where the policy is weak, rather than uniformly. It is also the safety bridge to Stage 11.

---

### Milestone C — Real hardware

#### Stage 8 · Leader-arm teleoperation
**Goal.** The intended input device replaces the keyboard.
**Builds.** `teleop/leader_arm.py`, `scripts/calibrate.py`, leader/follower calibration file format.
**Exit.** Leader arm drives the simulated follower with acceptable latency; calibration procedure documented and repeatable from cold start; 50 human demonstrations collected.
**Note.** Decide here whether record/reset/discard/takeover are triggered from leader-arm buttons or the keyboard. Buttons are materially better past ~50 episodes, and Stage 7 adds a fourth trigger.

#### Stage 9 · Real robot driver and remote execution
**Goal.** The same loop drives physical hardware.
**Builds.** `robots/real/<driver>.py` for the actual arm, `robots/remote/{zmq_client,zmq_server}.py` (with frame compression), `cameras/async_capture.py`, hand-eye camera calibration.
**Exit.** `robot.type` swapped in config drives the real arm with no other change; **the real driver passes the conformance suite on the bench**; a policy running on a remote GPU controls the local arm through `AsyncPolicySource`; network round-trip latency appears as `staleness_ms` in the logged stream; **threaded capture holds 30 Hz on real cameras**, with per-stream capture timestamps recorded and dropped-frame counts reported.
**Risk.** This is where the `RobotSpec` abstraction gets its real test. Expect to discover spec fields that were missing. The spec hash (§4.5) makes the resulting dataset split visible instead of silent.

#### Stage 10 · Sim2real transfer
**Goal.** A sim-trained policy works on hardware.
**Builds.** Domain randomization sweep (lighting, textures, distractors, camera pose ±2 cm / ±3°), explicit actuation-latency model in sim **calibrated against Stage 9's measured `staleness_ms`**, system identification against real trajectories, sim+real co-training recipe.
**Exit.** A policy trained predominantly in sim, co-trained with a small set of real demonstrations, succeeds on the real robot at a documented rate — measured with the Stage 6 harness, human-supervised.
**Ranked gap list.** visual appearance > actuation latency > contact/friction and gripper compliance > servo dynamics > camera extrinsics > object mass/geometry.

#### Stage 11a · Real-hardware success detection and reset *(new)*
**Goal.** Make hardware autonomy *measurable*, which is a precondition for running it unattended.
**Builds.** `success/{learned,fixture,human}.py` — whichever the task admits; `reset/{scripted_replace,human_batch}.py`; agreement measurement against human labels.
**Exit.** An automated real-hardware success detector agrees with human labelling at a documented rate (target >95%) over at least 100 labelled episodes, and the chosen reset strategy is demonstrated across a batch.
**Note.** Split out of v2's Stage 11, which assumed this existed. Its cost is the real answer to open question 4: it bounds how unattended "unattended" can be.

#### Stage 11b · Autonomous operation on real hardware
**Goal.** The robot performs the task unattended on physical hardware.
**Builds.** `control/safety/hardware_safety.py` — force/current limits (against `RobotSpec.sensing`), workspace fencing from `RobotSpec.workspace_bounds`, collision and stall detection, hardware e-stop integration, safe-stop-on-trip.
**Exit.** A documented number of consecutive autonomous episodes on hardware with no human control input, scored by the Stage 11a detector, with every abnormal termination ending in a safe stop rather than a collision.
**Risk.** The highest-risk stage in the project. In sim a watchdog trip costs a wasted episode; on hardware it can cost the robot. Gate entry on Stage 7's intervention mode working reliably, and ramp: supervised-with-takeover → supervised-hands-off → unattended.

---

### Milestone D — Hierarchy and research

#### Stage 12 · Two-level policy and task specification
**Goal.** Slow planner plus fast controller — and a way to *tell* the autonomous system what to do now that no human is driving.
**Builds.** High level: a VLM emitting `{subtask: str, target_point_3d, phase}` at 1–5 Hz behind an `AsyncPolicySource`, which also supplies object grounding. Low level: the Stage 5/7 controller, conditioned on that. Explicit, typed, human-readable interface between the levels. A `TaskRequest` entry point (natural-language goal in, autonomous execution out).
**Exit.** Multi-step and language-conditioned tasks succeed on the Stage 6 harness; each level can be tested in isolation by injecting ground truth from the other; the fast loop holds rate while the slow level runs.
**Note.** Because Stage 2 made asynchrony a property of the source, the hierarchy is two sources at two rates rather than a new execution model. A latent-only interface between the levels is more elegant and much harder to debug — start typed; go latent only if typed proves limiting.

#### Stage 13 · World-model experiment
**Goal.** Evaluate a JEPA-style latent world model as the low level.
**Builds.** Action-conditioned video world model (V-JEPA 2-AC direction) with latent-space planning, benchmarked against the Stage 5 baseline on the same evaluation harness.
**Exit.** A like-for-like comparison against ACT on identical seeds, tasks, and `PolicySpec`-validated observation contracts.
**Note.** Sequenced last deliberately. Its value depends on having a data engine and an evaluation harness that already work; without a baseline the result is uninterpretable.

---

## 8. Data, perception, and training strategy

- **One dataset schema** (LeRobot v3) for sim and real, teleoperated and autonomous, with `ActionOrigin` distinguishing human, correction, scripted, and policy steps.
- **Per episode:** all camera streams, joint positions and velocities, force/current where sensed, commanded (clipped) actions, gripper state, per-sample timestamps, staleness, task string, `EpisodeResult`, and the provenance hashes of §4.5.
- **No separate perception stack.** Policies consume pixels end to end; there is deliberately no 6-DoF pose-estimation module. Object grounding for multi-step tasks arrives with the VLM at Stage 12. Revisit only if end-to-end grounding proves to be the bottleneck.
- **Co-training** sim-heavy with a real-data minority is the primary sim2real lever, ahead of any individual randomization trick.
- **Intervention data** (Stage 7) is weighted separately from base demonstrations — it is targeted at failure regions and is worth more per frame.
- **Portability:** datasets live on the HF Hub so a Mac and a cloud GPU train on byte-identical data.

## 9. Risks

| Risk | Mitigation |
|---|---|
| Unknown hardware invalidates spec assumptions | Spec is data; expect one revision at Stage 9 and budget for it; conformance suite catches driver-level drift |
| Visual sim2real gap is the dominant failure | Stage 10 randomization + real co-training; escalate to Isaac rendering on a Linux box only if RGB plateaus |
| Autonomous run damages hardware | Stage 11b safety layer; staged ramp from supervised to unattended; workspace bounds in the spec from Stage 0; force limits refuse to arm without backing sensing |
| Autonomous failures are unmeasurable / anecdotal | Stage 6 failure taxonomy and batch harness before any autonomy claim; Stage 11a extends this to hardware |
| Hidden latency (control loop or network) degrades real performance | Timestamp everything from Stage 0; `staleness_ms` logged per step from Stage 2; model actuation delay in sim from Stage 10 |
| Sequential loop cannot hold rate on hardware | Capture, inference and disk writes run off the loop thread (§3.5); per-stage latency measured at Stage 2 before anything is threaded |
| Threaded capture misaligns images and joint state | Per-stream capture timestamps in `Observation`; sensor age logged per step and bounded by `cameras.max_age_ms` |
| Threading destroys reproducible replay | Capture concurrency is a per-component config field; sim defaults to synchronous |
| Slow inference silently degrades control | `AsyncPolicySource` with counted underruns and an explicit fallback; slow-policy test in Stage 2's exit criterion |
| Checkpoint/robot mismatch degrades quietly | `PolicySpec` + `check_compatibility` at construction; camera names are a hard contract |
| Spec revision silently poisons old datasets | Spec hash in every episode; `train.py` refuses mixed specs without an explicit flag |
| Over-abstraction slows development | Eight named seams, each with its second implementation already identified; concrete elsewhere; a ninth requires naming two implementations |
| LeRobot API churn breaks the repo | Own the core interfaces; confine the dependency to one adapter module |
| Bimanual/dexterous arrives sooner than planned | Accepted as a breaking core-type revision; spec hashing makes the dataset split explicit |

## 10. Open questions

1. Placeholder arm for Stages 0–7 — proposal: **SO-101** (MuJoCo Menagerie support, native leader arm, plausible eventual hardware).
2. Record/reset/discard/takeover triggers: leader-arm buttons or keyboard (decided at Stage 8).
3. GPU availability and timing, which determines whether Stage 10 randomization runs locally or in the cloud.
4. **Real-hardware success detection strategy** — learned classifier, instrumented fixture, or human labelling. Now scheduled as Stage 11a rather than assumed. Its answer bounds how unattended Stage 11b can be, and it should be chosen before hardware is ordered, since a fixture is a purchasing decision.
5. Physical auto-reset strategy for Stage 11b — fixture, scripted re-place, or human reset between batches.
6. Task list beyond pick-and-place, which shapes when the `SuccessDetector` taxonomy needs generalizing.
7. Does the target arm report current or torque? This determines whether Stage 11b's force limits are real protection or decorative, and it is a hardware-selection criterion, not a discovery.

## 11. Changelog

**v3.1 (this revision)** — concurrency made explicit:

1. **§3.5 added.** Which thread owns what, with the 30 Hz budget arithmetic that forces it. Capture, inference and recording move off the loop thread; the joint read, `send_action` and the `SafetyMonitor` check stay on it. Threads over processes (the hot calls are native and release the GIL); processes reserved for Stage 6 batch eval, which is parallel across *episodes*. asyncio rejected. The 30–100 Hz upper bound is dodged by `JOINT_POSITION` — the servos close their own loop, so Python generates setpoints rather than servoing.
2. **Per-stream capture timestamps** added to `Observation` (§4.1) and to the step record (§4.4). A single scalar timestamp is only honest while capture is synchronous.
3. **Capture concurrency is per-component and config-driven** (§6), because threaded capture costs the bit-exact replay that Stage 3's exit criterion depends on. Sim defaults synchronous, hardware threaded.

Also: principle 4 now says *one code path, not one thread*; Stage 2 logs a per-stage latency breakdown and explicitly threads nothing until it is read; Stage 9 adds threaded capture; three risk rows.

**v3** — addresses seven findings from design review:

1. **Morphology claim corrected.** v2 asserted no non-goal was precluded by the architecture; the flat `joint_pos` + scalar `gripper` types do preclude bimanual and dexterous hands. Now a stated, priced decision (§1, §4.1).
2. **Asynchronous inference made first-class.** v2's blocking `get_action` was incompatible with its own objective 5, plus remote policies and Stage 13. Buffering and threading moved from `Policy` into `PolicySource`/`AsyncPolicySource`; `staleness_ms` and underruns logged from Stage 2 (§3.4, principle 7).
3. **`Task` decomposed.** Success detection and reset split out as seams, because they differ fundamentally between sim and hardware. Surfaced a missing work item — real-hardware success detection — now Stage 11a (§3.3, §2.1).
4. **`PolicySpec` and capability negotiation added.** Camera names, control mode, rate, and observation keys are now a checked contract that fails at construction (§4.3, principle 11).
5. **Seam count made honest.** Six → eight, each earned by a named second implementation; `Watchdog`/`SafetyMonitor` was already a de facto seam in v2 (§2.1).
6. **Dataset provenance added.** `RobotSpec` + hash, `PolicySpec` hash, and code version recorded per episode; `train.py` refuses silently mixed specs (§4.5).
7. **Conformance suite added.** Principle 3's unit conventions were stated but unenforced; now a shared suite every `Robot` must pass, including the real driver on a bench (§5.1).

Smaller: `Action.source` string → `ActionOrigin` enum covering Stage 7 and Stage 12 provenance; force/current/wrench added to `Observation` and declared in `RobotSpec.sensing`; `Retargeter` constructed with the target `RobotSpec`; ZMQ frame compression noted; `success/` and `reset/` lifted out of `envs/`.
