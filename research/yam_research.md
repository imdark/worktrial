# YAM — robot model and open-data research

**Question asked.** Can we adopt the I2RT **YAM** arm: is there a MuJoCo model we can
import, how much open-source data exists in LeRobot format, and what does it take to get
a simulation running?

**Researched** 2026-09-24. Findings marked **[verified]** were executed locally against
this repo's `.venv`; findings marked **[reported]** come from vendor docs or search results
and have not been run.

---

## Bottom line

1. **The MuJoCo model is a solved problem.** YAM ships in MuJoCo Menagerie as `i2rt_yam`,
   MIT-licensed, and loads and renders in our MuJoCo 3.14 with no modification. **[verified]**
2. **The data is enormous — ~8,450 hours across 225 unique LeRobot datasets** — but it is
   almost entirely *real-robot* recording. It will not give us a working simulation. **[verified]**
3. **LeRobot has no YAM driver.** The driver and the MJCF both come from I2RT's own repo.
   That is a third-party seam we do not control.

The realistic plan is: **Menagerie MJCF for the sim, LeRobot for dataset format and
policies, I2RT's repo for the driver and gripper variants.** Three upstreams, not one.

---

## 1. MuJoCo model

### Option A — MuJoCo Menagerie (recommended)

`google-deepmind/mujoco_menagerie` → `i2rt_yam/`. 6.4 MB sparse, 17 STL meshes,
`yam.xml` + `scene.xml`. Requires MuJoCo ≥ 3.1.3.

Loaded against our installed MuJoCo 3.14.0: **[verified]**

```
LOADED  nq=8  nv=8  nu=7  nbody=14  ngeom=55
joints:
   joint1        range=[-2.618, +3.054] rad
   joint2        range=[+0.000, +3.665] rad
   joint3        range=[+0.000, +3.665] rad
   joint4        range=[-1.571, +1.571] rad
   joint5        range=[-1.571, +1.571] rad
   joint6        range=[-2.094, +2.094] rad
   left_finger   range=[-0.002, +0.038]
   right_finger  range=[-0.038, +0.002]
actuators: joint1..joint6, gripper      (7 total)
cameras:   []          keyframes: ['home']
stepped 200x OK · offscreen render (240,320,3) OK
```

**License: MIT, © 2025 i2rt robotics.** No obstacle to vendoring into `assets/`.

### Option B — I2RT's own repo

`i2rt-robotics/i2rt` → `i2rt/robot_models/`. Richer, use when the variant matters: **[verified — file tree]**

| Path | Contents |
|---|---|
| `arm/yam/v1/` | `yam.urdf`, `yam.xml` |
| `arm/yam_pro/v1/`, `arm/yam_ultra/v1/`, `arm/yam_ultra/v2/`, `arm/big_yam/v1/` | hardware variants |
| `gripper/` | `linear_4310`, `crank_4310`, `flexible_4310`, `linear_3507`, `no_gripper`, `yam_teaching_handle` |
| `station/` | full arm + gripper + RealSense D405 assemblies |

The same repo carries a MuJoCo control path: `i2rt/robots/sim_robot.py`,
`i2rt/utils/mujoco_control_interface.py`, `examples/control_with_mujoco/`. Worth reading
before we write our own `MujocoRobot` — it may already solve gravity comp for this arm.

---

## 2. Open-source data inventory

Method: queried the HF API for datasets tagged `yam`, then fetched `meta/info.json` from
each and collapsed mirrors by `(total_episodes, total_frames)` signature. **[verified]**

| Measure | Value |
|---|---|
| Datasets tagged `yam` | 244 |
| Metadata successfully fetched | 234 |
| **Unique after collapsing mirrors** | **225** |
| **Episodes** | **369,184** |
| **Frames** | **911,570,187** |
| **Wall-clock** | **~8,450 hours** |
| LeRobot format split | 177 × v2.1, 57 × v3.0 |

`robot_type` values in the wild — note there is **no single canonical string**, which
matters for any filtering we write:

```
yam 201 · yam_bimanual 15 · bi_yam_follower 3 · yam_follower_bimanual 2
bi_yam 2 · bi_yam_sim 2 · YAM_ST_bimanual 2 · yam_ultra_bimanual 1
yam_right_linear_4310 1 · molmoact_dual_arm 1 · (null) 4
```

### ABC-130k dominates

One release is ~42% of all episodes: `lerobot/abc_130k_v3_train`
(mirrored as `typoverflow/abc_130k_train`, `huzheyuan/abc130k_v3_train`).

- 129,225 episodes · 382,468,339 frames · **~3,540 hours** · 201 distinct tasks
- `robot_type: yam_bimanual`, 30 fps, AV1
- **Apache-2.0** · **3.43 TB**
- Splits: `abc_130k_v3_train` / `_val` (1,592 eps) / `_smoke` (85 eps, 2.97 GB)

Schema: **[verified]**

```
observation.state               float32   [14]
action                          float32   [14]
observation.images.top          video     [224, 224, 3]
observation.images.left_wrist   video     [224, 224, 3]
observation.images.right_wrist  video     [224, 224, 3]
language_persistent             language  [1]
language_events                 language  [1]
```

### Largest datasets

| Dataset | Episodes | Frames | robot_type |
|---|---:|---:|---|
| `lerobot/abc_130k_v3_train` | 129,225 | 382.5 M | `yam_bimanual` |
| `kkipngenokoech/abc-molmoact2-yam-mix` | 21,767 | 61.1 M | `yam_bimanual` |
| `nuffnuff/robocoin-yam-v3-269x500` | 46,173 | 36.9 M | `YAM_ST_bimanual` |
| `nuffnuff/yam-master-v1-strong-aug` | 5,995 | 12.0 M | `yam_follower_bimanual` |
| `angkul07/EgoDex-PickPlace-YAM-14dof-multiview` | 8,842 | 1.07 M | `yam_bimanual` |

---

## 3. The catch — this data does not "run" in MuJoCo

Nearly all 8,450 hours is teleoperated **real-robot** recording: joint trajectories plus
camera video. There is **no object pose, no scene description, no physics state**.

- ✅ We **can** replay the 14-dim action stream open-loop into MuJoCo. That is precisely
  our own Stage 3 exit criterion (`DESIGN.md:403`).
- ❌ We **cannot** derive a task simulation from it. Nothing records where the vial, plate
  or legos were. The arm will move correctly through an empty scene.
- ❌ Recorded frames will not match rendered frames. That is the Stage 10 sim2real problem;
  this data does not close it.

Genuinely sim-native YAM data is negligible — two datasets, one author:
`Dimios45/sim_cube_stack_50` (50 eps, 20 fps, 0.33 GB) and
`Dimios45/yam-sim-cube-stacking` (3 eps). Useful as a format reference, not a corpus.

### Three concrete mismatches to budget for

1. **14-dim data vs. a 7-actuator model.** The Menagerie MJCF is *one* arm (6 joints +
   gripper). ABC-130k is bimanual, 2 × 7. Two YAM instances must be composed into one scene.
2. **`scene.xml` defines zero cameras.** To produce observations matching the dataset we
   must add `top`, `left_wrist`, `right_wrist` ourselves.
3. **Gripper units.** Model fingers are `[-0.002, 0.038]` — prismatic metres, not our
   normalized `[0, 1]`. Exactly the conversion `DESIGN.md:52` says must live inside a driver,
   and the conformance suite must test it at both extremes.

Also note a **frame-rate split**: real YAM data is 30 fps, the two sim datasets are 20 fps.

---

## 4. Integration gap — LeRobot has no YAM driver

Checked the `huggingface/lerobot` `main` git tree. It has **13** robot drivers: **[verified]**

```
bi_openarm_follower · bi_rebot_b601_follower · bi_so_follower · earthrover_mini_plus
hope_jr · koch_follower · lekiwi · omx_follower · openarm_follower · reachy2
rebot_b601_follower · so_follower · unitree_g1
```

**None is YAM or I2RT.** (Every apparent "yam" path match was the substring inside "yaml".)
Our installed **lerobot 0.3.2** has an older, even shorter list.

Consequence: data and policies come from LeRobot; the *driver* comes from
`i2rt-robotics/i2rt`. This is the "LeRobot API churn" risk in `DESIGN.md:510`, except the
exposure is larger than the single adapter module that risk assumed — we would depend on
two independently-versioned third parties.

---

## 5. Practical starting points

| Dataset | Size | Why |
|---|---:|---|
| `ttotmoon/yam_pick_up_grey_cube` | 0.04 GB | smallest valid v3.0 sample |
| `Dimios45/sim_cube_stack_50` | 0.33 GB | only real sim-native YAM data |
| `lerobot/abc_130k_v3_smoke` | 2.97 GB | official smoke split, schema-identical to the 3.43 TB train split |
| `flex-pi/sort_utensils` | 14.0 GB | mid-size single-task |
| `andlyu/Public-YAM-runs` | 34.2 GB | most-downloaded YAM set (11.9k/30d) |
| `lerobot/abc_130k_v3_train` | **3.43 TB** | the full corpus |

Start with the 2.97 GB smoke split — anything that loads it loads the full train split.

---

## 6. Bearing on open question 1

`DESIGN.md:515` proposes **SO-101** as the placeholder arm for Stages 0–7, on three
criteria: Menagerie support, a native leader arm, and plausible eventual hardware.

YAM now scores better on two of the three:

| Criterion | SO-101 | YAM |
|---|---|---|
| Menagerie support | ✅ `robotstudio_so101`, `trs_so_arm100` | ✅ `i2rt_yam`, MIT |
| Public LeRobot data | thin | **~8,450 h, 225 datasets** |
| Native leader arm | ✅ | ❌ — GELLO or I2RT's own stack |
| LeRobot driver upstream | ✅ `so_follower` | ❌ none |
| Hardware cost | ~$150 | ~$1,500–3,000 **[reported]** |

The leader arm and the missing upstream driver are the real arguments for staying on
SO-101 through Milestone A. The data argument for YAM only starts paying at Stage 5.

**Not a decision — input to one.** Worth noting the choice is cheaper to revisit than it
looks: `RobotSpec` YAML plus a driver is the whole cost, by construction.

---

## 7. Reproducing this

```bash
# MJCF — sparse clone, 6.4 MB
git clone --filter=blob:none --sparse --depth 1 \
  https://github.com/google-deepmind/mujoco_menagerie.git
cd mujoco_menagerie && git sparse-checkout set i2rt_yam
python -c "import mujoco; m=mujoco.MjModel.from_xml_path('i2rt_yam/scene.xml'); print(m.nq, m.nu)"

# Dataset inventory
curl -s "https://huggingface.co/api/datasets?filter=yam&limit=1000"
# then per dataset:
curl -s "https://huggingface.co/datasets/<id>/resolve/main/meta/info.json"
```

Caveat on the inventory: `filter=yam` relies on dataset owners tagging correctly, and the
`robot_type` spread above shows tagging is inconsistent. Treat 225 / 8,450 h as a **lower
bound** — some YAM data is certainly untagged, and a little non-YAM data may be swept in.

## 8. Not checked

- Whether ABC-130k's 201 tasks are usable as-is or need relabelling.
- Data quality — no episode was opened or viewed; all figures are from metadata.
- Whether I2RT's `sim_robot.py` gravity comp matches the Menagerie MJCF's inertials.
- GELLO-based YAM teleoperation, which is the gap if we adopt YAM before Stage 8.
- Licenses of the 224 non-ABC datasets (only ABC-130k's Apache-2.0 was confirmed).

## Sources

- MuJoCo Menagerie — https://github.com/google-deepmind/mujoco_menagerie
- I2RT stack + models — https://github.com/i2rt-robotics/i2rt
- YAM product docs — https://doc.i2rt.com/products/yam
- ABC-130k — https://huggingface.co/datasets/lerobot/abc_130k_v3_train
- LeRobot — https://github.com/huggingface/lerobot
