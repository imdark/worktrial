"""Replay recorded real runs through the motor-torque contact detector.

    python scripts/eval_contact_detector.py runs/current_sample_*/ep01 [--out report.json]

Two measurements, both offline (the arm is not touched):

1. False stops. Leave-one-run-out: fit the torque model on the other runs,
   replay the held-out run step by step through ContactDetector, count trips.
   These runs had no collisions, so every trip is a false stop (trips inside
   phases where contact is expected are reported separately).
2. Synthetic pushes. On the held-out run, add a known force at the gripper
   (+-x, +-y, +-z in the arm base frame, as a 1 s step or a 2 s ramp), turned
   into joint torques through the arm's Jacobian (tau = J^T F), at random
   free-motion moments. Reports the fraction detected and the delay. These are
   SIMULATED pushes on real noise, a stand-in until staged contacts are
   recorded on the rig -- they assume the motors report external torque 1:1.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from teleop_sim.control.safety.contact import (  # noqa: E402
    CONTACT_PHASES,
    ContactDetector,
    RunArrays,
    fit_params,
    run_features,
)
from teleop_sim.core.spec import RobotSpec  # noqa: E402

SPEC = ROOT / "teleop_sim/robots/specs/yam_kronos_gem13.yaml"
DIRECTIONS = {
    "+x": [1, 0, 0],
    "-x": [-1, 0, 0],
    "+y": [0, 1, 0],
    "-y": [0, -1, 0],
    "+z": [0, 0, 1],
    "-z": [0, 0, -1],
}


def replay(params, residual, run, extra=None, stop_at_first=False):
    det = ContactDetector(params)
    trips = []
    for i in range(len(run.t)):
        r = residual[i] if extra is None else residual[i] + extra[i]
        trip = det.update(r, run.phase[i], run.cmd_q[i], run.q[i])
        if trip is not None:
            trips.append((i, trip))
            if stop_at_first:
                break
            det.reset()
    return trips


def ee_jacobians(model, run, spec) -> np.ndarray:
    """(N, 3, dof) translational Jacobian of the end-effector site."""
    import mujoco

    kin = model.kin
    m, d = kin.model, kin.data
    site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, spec.ee_site)
    out = np.zeros((len(run.t), 3, len(kin._dadr)))
    jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
    for i, q in enumerate(run.q):
        d.qpos[:] = m.qpos0
        d.qpos[kin._qadr] = q
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        mujoco.mj_jacSite(m, d, jp, jr, site)
        out[i] = jp[:, kin._dadr]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("episodes", nargs="+", type=Path)
    ap.add_argument("--spec", type=Path, default=SPEC)
    ap.add_argument("--forces", type=float, nargs="+", default=[2, 5, 10, 20], help="N")
    ap.add_argument("--trials", type=int, default=24, help="pushes per force and shape")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    spec = RobotSpec.from_yaml(args.spec)
    runs = [RunArrays.load(e) for e in args.episodes]
    if len(runs) < 2:
        raise SystemExit("need at least two runs: fit on the others, test on each")
    rng = np.random.default_rng(args.seed)
    report: dict = {"runs": [r.name for r in runs], "held_out": []}
    push = {}  # (shape, force) -> [detected, delays]

    for k, test in enumerate(runs):
        train = [r for i, r in enumerate(runs) if i != k]
        model, params, _ = fit_params(train, spec)
        feats = run_features(model, test)
        residual = test.eff - np.einsum("njf,jf->nj", feats, model.coef)
        trips = replay(params, residual, test)
        free_trips = [(i, t) for i, t in trips if test.phase[i] not in CONTACT_PHASES]
        row = {
            "run": test.name,
            "seconds": round(float(test.t[-1]), 1),
            "residual_std_nm": residual[test.free()].std(axis=0).round(3).tolist(),
            "false_stops": len(free_trips),
            "trips_in_contact_phases": len(trips) - len(free_trips),
            "trips": [
                {"t": round(float(test.t[i]), 1), "phase": test.phase[i], "detail": t.detail}
                for i, t in trips
            ],
        }
        report["held_out"].append(row)
        print(f"\nheld out {test.name} ({row['seconds']} s), fit on {len(train)} other run(s)")
        print(f"  residual std, Nm {row['residual_std_nm']}")
        print(
            f"  false stops (free motion): {row['false_stops']}; "
            f"trips in contact phases: {row['trips_in_contact_phases']}"
        )
        for tr in row["trips"]:
            print(f"    {tr['t']:6.1f}s {tr['phase']:10s} {tr['detail']}")

        # synthetic pushes at the gripper
        jac = ee_jacobians(model, test, spec)
        hz = 1.0 / np.median(np.diff(test.t))
        free_idx = np.flatnonzero(test.free())
        for shape, dur_s in (("step", 1.0), ("ramp", 2.0)):
            n = int(dur_s * hz)
            for force in args.forces:
                det_n, ran, delays = 0, 0, []
                for _ in range(args.trials):
                    start = int(rng.choice(free_idx[free_idx < len(test.t) - n - 1]))
                    window = range(start, start + n)
                    if any(test.phase[i] in CONTACT_PHASES for i in window):
                        continue
                    ran += 1
                    f = np.array(DIRECTIONS[rng.choice(list(DIRECTIONS))], dtype=float) * force
                    extra = np.zeros_like(residual)
                    for i in range(start, start + n):
                        a = 1.0 if shape == "step" else (i - start + 1) / n
                        extra[i] = jac[i].T @ (a * f)
                    hit = [
                        i
                        for i, _ in replay(params, residual, test, extra)
                        if start <= i < start + n
                    ]
                    if hit:
                        det_n += 1
                        delays.append((hit[0] - start) / hz)
                p = push.setdefault((shape, force), [0, 0, []])
                p[0] += det_n
                p[1] += ran
                p[2] += delays

    print("\nSIMULATED pushes at the gripper (real noise, J^T F added; not real contacts)")
    report["synthetic_pushes"] = []
    for (shape, force), (hit, n, delays) in sorted(push.items()):
        line = {
            "shape": shape,
            "force_n": force,
            "detected": hit,
            "trials": n,
            "median_delay_ms": round(1000 * float(np.median(delays))) if delays else None,
        }
        report["synthetic_pushes"].append(line)
        print(
            f"  {shape:4s} {force:5.1f} N: detected {hit}/{n}"
            + (f", median delay {line['median_delay_ms']} ms" if delays else "")
        )
    rules = Counter(t["detail"].split(" ")[0] for r in report["held_out"] for t in r["trips"])
    report["trip_rules"] = dict(rules)
    if args.out:
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        print(f"report: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
