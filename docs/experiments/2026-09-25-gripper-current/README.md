# Gripper current tells held from not held (gem13, 2026-09-25)

**Question.** Can the Kronos gripper's motor current show whether the object
is held, without a vision model?

**Answer.** Yes, on these two runs. A held cup reads **~300** (the servos'
hold-current cap), and jaws closed on nothing read **0–1**. The rule "held iff
current >= 150" matches the jaw-position labels on **99.4 %** and **99.6 %** of
labelled steps, and caught every held step. The few disagreements are
transients while the jaws move.

## Setup

- Rig: gem13, right YAM arm, Kronos gripper (two Dynamixel XC330 servos behind
  a CAN bridge, driven in opposition). Task: pick up the frosted tapered cup,
  lift it, put it back (`scripts/collect_real_picks.py --episodes 1`).
- Signal: `gripper_eff` on the rig's `yam_right/joint_state` bus topic. On
  this gripper the driver (`robots_realtime/robots/can_dxl_gripper.py`) fills
  it with the **sum of |present current| of both servos**, in raw servo units
  (about mA for the XC330). `teleop_sim/robots/real/rr_bridge.py` now passes it
  through (`obs.extra["motors"]`), and `teleop_sim/telemetry.py` records it at
  every control step (~25 Hz), with `gripper_vel`, `joint_eff` and `joint_vel`.
- Labels come from the jaw position, not from current: jaws commanded closed
  and stopped part-way means held; jaws open, or commanded closed and shut
  (>= 0.9), means not held; jaws moving means uncertain
  (`telemetry.label_frame`). Current is an independent check of them.

| Run | Time | Grasp config | What happened |
|---|---|---|---|
| 1 | 15:26 | as the 10-pick batch: centre, 80 mm up, 10°, closer view when views agree | attempt 1 stopped ~2–3 cm short and closed on nothing; attempt 2 held |
| 2 | 15:31 | 10 mm past centre, always the mean of views | held on the first attempt, but the operator saw it go **too close**; reverted |

## Results

| Label | Run 1 current (median, p10–p90) | Run 2 current |
|---|---|---|
| held | 300 (299–399), n=547 | 299 (299–301), n=523 |
| closed on nothing | **0 (0–1)**, n=537 | none (no miss) |
| not held, overall | 21 (0–36), n=2463 | 30 (18–35), n=1161 |
| uncertain (jaws moving) | 411 (56–526), n=151 | 363 (133–570), n=74 |

Shape of a grasp, from the time series:

- **Onto the cup:** ~400–500 while closing, a grab spike of **~780–840** for
  ~0.5 s when the fingers meet the cup, then flat at **~300** with the jaw
  stopped (0.73 in run 1, 0.68 in run 2).
- **Onto nothing:** ~400–500 while closing, then down to **0–1** within
  ~1.7 s once the jaws reach their target, with the jaw at 0.99.

So after about 1–2 s, current and jaw position agree: ~300 with the jaw part-way
means held, ~0 with the jaw shut means empty. The "closed on nothing" case is
the hard negative the laya-vision `gem13_held` dataset lacked; current now
labels it for free.

## Caveats

- Two runs, one object, one miss. Collect deliberate misses and other objects
  before relying on it.
- The ~300 plateau is the driver's configured hold current, not a force
  measurement. A different current setting, a softer object, or a slipping
  grasp will read differently.
- Raw units. They have not been checked against a current meter.

## Files

- `run1_miss_then_hold.{csv,json,svg}` and `run2_hold.{csv,json,svg}`: per-step
  time series (time, phase, label, jaw measured / commanded, current,
  velocity), statistics, and the plot (current on top, jaw below, shaded by
  label).
- Regenerate from the raw run directories (`runs/` is not tracked):

```bash
python scripts/plot_gripper_current.py runs/current_sample_20260925-152657/ep01 \
    --out run1_miss_then_hold.svg --csv run1_miss_then_hold.csv --json run1_miss_then_hold.json
```

## Next

- Add current to the jaw check in `PickLiftPlacePolicy._grasp_held`: ~300 with
  the jaw part-way means held, with no vision call. Ask Claude only when the two
  disagree.
- Relabel training frames with position and current together, and record
  deliberate misses as hard negatives.
