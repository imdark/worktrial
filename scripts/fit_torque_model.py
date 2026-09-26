"""Fit the contact detector's torque model on free-motion runs of the real rig.

    python scripts/fit_torque_model.py runs/current_sample_*/ep01 \
        --out data/torque_models/gem13.json

For every recorded step with motor readings (teleop_sim/telemetry.py, real
rig, from 2026-09-25): MuJoCo inverse dynamics on the robot description at the
measured joints, then a per-joint linear fit of the measured torque on
(gravity, inertia, velocity, friction sign, offset). Steps in phases where
contact is expected (closing, putting down) are left out of the fit. Writes the
coefficients plus the residual spreads the detector normalises by, and the
worst command lag seen, as DetectorParams JSON. The output lives under data/
(not tracked): it is fitted to one rig.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from teleop_sim.control.safety.contact import RunArrays, fit_params  # noqa: E402
from teleop_sim.core.spec import RobotSpec  # noqa: E402

SPEC = ROOT / "teleop_sim/robots/specs/yam_kronos_gem13.yaml"


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("episodes", nargs="+", type=Path)
    ap.add_argument("--spec", type=Path, default=SPEC)
    ap.add_argument("--out", type=Path, default=ROOT / "data/torque_models/gem13.json")
    args = ap.parse_args()

    runs = [RunArrays.load(e) for e in args.episodes]
    _, params, _ = fit_params(runs, RobotSpec.from_yaml(args.spec))
    params.save(args.out)
    m = params.meta
    print(f"fitted on {m['free_steps']} free-motion steps from {len(runs)} runs")
    print(f"  raw torque std, Nm   {m['raw_torque_std']}")
    print(f"  residual std, Nm     {m['residual_std']}")
    print(f"  sigma used, Nm       {params.sigma}")
    print(f"  jump sigma, Nm       {params.sigma_jump}")
    print(f"  worst command lag, deg {params.track_max_deg}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
