# Collision detection on gem13: motor torque + on-board laya-vision (2026-09-25)

**Question.** Can the arm stop itself when something gets in its way, and go on once the
way is clear, using only its own motors and a small vision model on the laptop?

**Answer.** Yes, in this test. In a 254 s run of 10 sweep cycles an operator disturbed
the arm 12 times. All 12 were caught: 3 pushes felt by the motors and 9 hands seen by
laya-vision. Every one stopped the arm, and every one ended with Claude resuming the
sweep. There were no false alarms. Detection took about 0.1 s for a push and about 0.5 s
for a hand in view. The arm was moving away from a push about 0.1 s after that. The
wait before resuming was 7.5–10 s, almost all of it Claude's review.

## Setup

- **Rig:** gem13, right YAM arm with the Kronos gripper.
- **Motion:** `teleop_sim/tasks/linear_sweep.py`, with no target. The gripper tip runs
  front/back along a 16 cm line and down/up along a 12 cm line, at 0.08 m/s average,
  with a minimum-jerk profile. Config: `teleop_sim/configs/yam_kronos_sweep_real.yaml`.
- **Command:**
  `scripts/run_sweep.py --contact-monitor enforce --hazard-monitor --max-alarms 12`
- **Motor torque monitor** (`teleop_sim/control/safety/contact.py`):
  - It predicts each joint's torque from the calibrated gem13 model (MuJoCo inverse
    dynamics at the measured joints), with a per-joint correction fitted on normal
    runs (`scripts/fit_torque_model.py`).
  - It trips when measured minus predicted stays high for 3 readings (about 100 ms at
    28 Hz), either as a sudden jump or a sustained offset.
  - The arm then backs off 3 cm along the path it just travelled.
- **Camera monitor** (`teleop_sim/control/safety/vision_hazard.py`): laya-vision runs
  locally, a few times a second, on the wrist and overhead cameras. It asks whether a
  person's hand or arm is in view. Three confident answers in a row pause the arm.
- **Claude's review** (`teleop_sim/control/safety/review.py`), after either monitor
  stops the arm:
  - What Claude sees: frames from the start of the run, the moment of the alarm, and
    now, plus what the detector saw and what the gripper's sensors report.
  - Decisions: resume, wait (look again in 2 s), or abort.
  - Models: Sonnet 5 decides. An unsure resume or abort (below 0.6) goes to Opus 5.5.
  - On resume after a back-off, the arm glides back to where it stopped before the
    sweep continues.
- **Fail-safe:** the arm stays stopped if Claude doesn't answer, if a hazard stays for
  60 s, or after more than the allowed number of alarms in an episode.

## Results (combined run, `runs/sweep_both_20260925-171928`)

| | Motor torque (3 pushes) | laya-vision (9 hands) |
|---|---|---|
| Detection | 108–125 ms from the torque rising to the trip | 458–502 ms from the first confident frame to the pause |
| Reaction | 5 mm back 103–112 ms after detection; backed off 20–46 mm | holds in place |
| Claude | wait, then resume, every time (Sonnet) | wait, then resume, every time (Sonnet) |
| Stopped for | 9.0–10.0 s | 7.5–8.4 s |
| Moving again after resume | 1.0–2.5 s (the glide back is included) | about 40 ms |
| False alarms | 0 | 0 |

Every alarm was checked by hand against the video. Each camera pause had a hand in view,
one holding a water bottle. Each contact was a hand pushing the forearm. The alarms one by
one are in `alarms_combined_run.csv`; the full report is `reaction_report_combined_run.json`.

## What earlier runs of the same day fixed

| Run | Problem seen | Fix |
|---|---|---|
| Replay of 15:26 pick | Claude called the empty gripper pressing a cup's rim a false alarm | The review now includes the gripper's sensors: jaw position and current ([gripper current](../2026-09-25-gripper-current/README.md)) |
| 16:22 live pick | After a resume, the empty put-down pressed the cup again | Reported; the policy should not put down when the gripper is empty (open) |
| 16:39 sweep | Claude aborted because of a bottle that was there all along | Frames from the start of the run are shown as the reference scene |
| 16:44 sweep | 4 hand pauses stopped the run, because the limit counted Claude reviews | The limit counts alarms (`max_alarms`) |
| 16:56 fast sweep | 22 false contacts: the torque model was fitted on slow picks only | Refit on fast motion too; the monitor resets after a pause |
| 17:02 fast touch test | Backed off 12–22 cm (it went back to the command of 1 s earlier), then snapped back at full speed on resume | Back off a fixed 3 cm; glide back smoothly before resuming; minimum-jerk motion |
| 17:14 smooth sweep | Sonnet aborted at 0.60 confidence after a real, cleared push | Unsure aborts go to Opus, like unsure resumes |

## Limits

- This is one 4-minute run with one operator. Detection rate and false-alarm rate need
  more runs, other people and other objects.
- **Pushes on the base read slower and weaker.** The base joint's limit is loose
  (0.5 Nm at 6 times the normal spread, calibrated at speed). In the earlier fast run,
  pushes felt at the base took 430–585 ms. Light pushes, of 2–5 N, may go undetected.
- **laya-vision only asks about people.** An object in the path is caught only by
  touch.
- **The camera frames reach the laptop at 5 fps over Tailscale,** so a hand can be in
  view for up to about 0.2 s before the first frame showing it.
- **Claude's review is the slow step.** It takes 7–10 s to resume.
- **This does not replace the e-stop or the rig's own limits.** The 28 Hz loop runs on
  the laptop, not the rig. A stop lands about 0.1–0.3 s after contact.

## Files and how to reproduce

- `alarms_combined_run.csv`, `reaction_report_combined_run.json`: the numbers above.
- **Videos stay local** because they show the operator. They're in
  `runs/sweep_both_20260925-171928/`: `video/cameras.mp4` (the whole run with events
  overlaid), `clips/contact_*.mp4`, `clips/camera_*.mp4` and `clips/highlights.mp4`.
- To record and render video during a run:

```bash
python scripts/record_cameras.py record --host <rig> --out runs/<run>/video
python scripts/record_cameras.py render runs/<run>/video --events runs/<run> \
    --start 70 --end 86 --title "Contact 2" --out clip.mp4
```

- To refit the torque model and replay recorded runs:
  `python scripts/fit_torque_model.py <runs>`, then
  `python scripts/eval_contact_detector.py <runs>`.
