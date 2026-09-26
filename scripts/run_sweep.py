"""Sweep the arm along straight lines with the safety monitors on; time their reactions.

    # real rig (session launched, camera relay running, laya server for --hazard-monitor)
    python scripts/run_sweep.py teleop_sim/configs/yam_kronos_sweep_real.yaml \
        --host pantheon-gem13 --camera-port 5557 --contact-monitor enforce --hazard-monitor
    # the same path in MuJoCo first
    python scripts/run_sweep.py teleop_sim/configs/yam_kronos_sweep_sim.yaml --fast
    # report again on a finished run
    python scripts/run_sweep.py --analyze runs/sweep_<ts>

The arm runs teleop_sim/tasks/linear_sweep.py: front/back and down/up lines
at a set tip speed, nothing targeted. Put a hand or an object in the way. For
every stop, reaction_report.json (and the printout) gives:

  contact (motor torque)   onset -> detected -> backing off -> 5 mm back ->
                           Claude's decisions -> resumed -> moving again
  hazard (camera)          first confident frame -> paused -> Claude's
                           decisions -> resumed -> moving again

"onset" is when the torque leftover of the tripping joint first rose past 2x
its normal spread before the trip, from telemetry (~28 Hz, so +-36 ms).
Everything is in one run directory: telemetry.jsonl, contact_checks.jsonl,
hazard_checks.jsonl, events.jsonl, Claude's calls and frames.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from teleop_sim.control.safety.contact import (  # noqa: E402
    DEFAULT_PARAMS,
    DetectorParams,
    RunArrays,
    TorqueModel,
    accelerations,
    with_contact_monitor,
)
from teleop_sim.control.safety.vision_hazard import with_hazard_monitor  # noqa: E402
from teleop_sim.core.clock import ManualClock, WallClock  # noqa: E402
from teleop_sim.core.config import RunConfig, build_system  # noqa: E402
from teleop_sim.core.spec import RobotSpec  # noqa: E402
from teleop_sim.runlog import RunLog  # noqa: E402
from teleop_sim.telemetry import TelemetryRecorder  # noqa: E402

SPEC = ROOT / "teleop_sim/robots/specs/yam_kronos_gem13.yaml"
MOVING_RAD_S = 0.03  # a joint faster than this counts as moving again
SWEEP_CONTEXT = (
    "The robot has no object to pick up: it is moving its gripper back and forth and up "
    "and down along straight lines in free space above the table, and is not meant to "
    "touch anything. The cups on the table, the robot's own gripper, its cables and "
    "mounts are expected; anything touching the arm or in its path is not."
)


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open()] if path.exists() else []


def _ms(a: float | None, b: float | None) -> int | None:
    return None if a is None or b is None else round(1000 * (b - a))


def analyze(run: Path, params_path: str = DEFAULT_PARAMS) -> dict:
    tele = _jsonl(run / "telemetry.jsonl")
    wall = np.array([r["wall"] for r in tele])
    qd = np.array([r.get("motors", {}).get("joint_vel", [0.0] * 6) for r in tele])
    contact = _jsonl(run / "contact_checks.jsonl")
    hazard = _jsonl(run / "hazard_checks.jsonl")
    report: dict = {"run": str(run), "contacts": [], "hazards": []}

    def moving_after(t: float | None) -> float | None:
        """First moment the arm moves after t, once it has been seen standing still.

        The first rows logged after a pause can carry the observation taken before
        the pause (the loop blocks inside the monitor), so a still row comes first.
        """
        if t is None or not len(wall):
            return None
        moving = np.abs(qd).max(axis=1) > MOVING_RAD_S
        after = np.flatnonzero(wall > t)
        still = [i for i in after if not moving[i]]
        if not still:
            return None
        idx = [i for i in after if i > still[0] and moving[i]]
        return float(wall[idx[0]]) if idx else None

    # residual torque, for the contact onset
    residual, sigma = None, None
    spec_ok = Path(params_path).exists() and SPEC.exists()
    if contact and spec_ok and any("joint_eff" in r.get("motors", {}) for r in tele):
        params = DetectorParams.load(params_path)
        model = TorqueModel(RobotSpec.from_yaml(SPEC), params.coef)
        arr = RunArrays.load(run)
        qdd = accelerations(arr.qd, arr.t)
        pred = np.array([model.predict(arr.q[i], arr.qd[i], qdd[i]) for i in range(len(arr.t))])
        residual, sigma = arr.eff - pred, np.maximum(np.asarray(params.sigma), 1e-3)
        res_wall = np.array([r["wall"] for r in tele if "joint_eff" in r.get("motors", {})])
        kin, ee_site = model.kin, RobotSpec.from_yaml(SPEC).ee_site

    i = 0
    while i < len(contact):
        c = contact[i]
        if c["event"] != "contact":
            i += 1
            continue
        ev = {
            "detected": c["t"],
            "phase": c.get("phase"),
            "detail": c.get("detail"),
            "joint": c.get("joint", 0) + 1,
            "decisions": [],
        }
        j = i + 1
        while j < len(contact) and contact[j]["event"] not in ("contact",):
            e = contact[j]
            if e["event"] == "retreat":
                ev["backing_off"] = e["t"]
            elif e["event"] == "retreat_trace" and e.get("trace") and residual is not None:
                before = np.flatnonzero(res_wall < c["t"])
                q0 = np.array(tele[before[-1]]["joint_pos"]) if before.size else None
                if q0 is not None:
                    p0 = kin.frame(q0, "site", ee_site)[0]
                    for s in e["trace"]:
                        p = kin.frame(np.array(s["q"]), "site", ee_site)[0]
                        d = float(np.linalg.norm(p - p0))
                        ev["backed_off_mm"] = round(1000 * d)
                        if d >= 0.005 and "back_5mm" not in ev:
                            ev["back_5mm"] = s["t"]
            elif e["event"] == "review":
                ev["decisions"].append(f"{e['decision']} ({e['confidence']:.2f}, {e['model']})")
            elif e["event"] == "resume":
                ev["resumed"] = e["t"]
            elif e["event"] == "stop":
                ev["stopped_for_good"] = e["t"]
            j += 1
        if residual is not None:
            k = int(np.searchsorted(res_wall, c["t"])) - 1
            jn = ev["joint"] - 1
            onset = None
            while k >= 0 and abs(residual[k, jn]) / sigma[jn] > 2.0:
                onset = float(res_wall[k])
                k -= 1
            ev["onset"] = onset
        ev["moving_again"] = moving_after(ev.get("resumed"))
        report["contacts"].append(ev)
        i = j

    pauses = [h for h in hazard if h["event"] in ("pause", "review", "resume", "trip")]
    answers = [h for h in hazard if h["event"] == "answer"]
    for n, h in enumerate(pauses):
        if h["event"] != "pause":
            continue
        ev = {"paused": h["t"], "check": h.get("check"), "detail": h.get("detail"), "decisions": []}
        check = h.get("check")
        highs = [a["t"] for a in answers if a["t"] <= h["t"] and a["p"].get(check, 0) >= 0.6]
        # the first of the run of confident answers that led to this pause
        first = None
        for a in reversed([a for a in answers if a["t"] <= h["t"] and check in a["p"]]):
            if a["p"][check] >= 0.6:
                first = a["t"]
            else:
                break
        ev["first_confident_frame"] = first or (highs[-1] if highs else None)
        for later in pauses[n + 1 :]:
            if later["event"] == "pause":
                break
            if later["event"] == "review":
                ev["decisions"].append(
                    f"{later['decision']} ({later['confidence']:.2f}, {later['model']})"
                )
            if later["event"] == "resume":
                ev["resumed"] = later["t"]
                break
        ev["moving_again"] = moving_after(ev.get("resumed"))
        report["hazards"].append(ev)

    for ev in report["contacts"]:
        ev["ms"] = {
            "onset_to_detected": _ms(ev.get("onset"), ev["detected"]),
            "detected_to_backing_off": _ms(ev["detected"], ev.get("backing_off")),
            "detected_to_5mm_back": _ms(ev["detected"], ev.get("back_5mm")),
            "detected_to_resumed": _ms(ev["detected"], ev.get("resumed")),
            "resumed_to_moving": _ms(ev.get("resumed"), ev.get("moving_again")),
        }
    for ev in report["hazards"]:
        ev["ms"] = {
            "first_frame_to_paused": _ms(ev.get("first_confident_frame"), ev["paused"]),
            "paused_to_resumed": _ms(ev["paused"], ev.get("resumed")),
            "resumed_to_moving": _ms(ev.get("resumed"), ev.get("moving_again")),
        }
    return report


def show(report: dict) -> None:
    t0 = None
    tele = _jsonl(Path(report["run"]) / "telemetry.jsonl")
    if tele:
        t0 = tele[0]["wall"]

    def at(t):
        return "   -  " if t is None or t0 is None else f"{t - t0:6.1f}s"

    print(f"\n{len(report['contacts'])} contact stop(s), {len(report['hazards'])} camera pause(s)")
    for n, ev in enumerate(report["contacts"], 1):
        m = ev["ms"]
        print(f"\ncontact {n} at {at(ev['detected'])} ({ev['phase']}): {ev['detail']}")
        print(f"  onset -> detected        {m['onset_to_detected']} ms")
        print(f"  detected -> backing off  {m['detected_to_backing_off']} ms")
        print(
            f"  detected -> 5 mm back    {m['detected_to_5mm_back']} ms "
            f"(backed off {ev.get('backed_off_mm', '-')} mm)"
        )
        print(f"  Claude                   {', '.join(ev['decisions']) or '-'}")
        print(
            f"  detected -> resumed      {m['detected_to_resumed']} ms"
            + ("  (stopped for good)" if ev.get("stopped_for_good") else "")
        )
        print(f"  resumed -> moving        {m['resumed_to_moving']} ms")
    for n, ev in enumerate(report["hazards"], 1):
        m = ev["ms"]
        print(f"\ncamera pause {n} at {at(ev['paused'])}: {ev['detail']}")
        print(f"  first confident frame -> paused  {m['first_frame_to_paused']} ms")
        print(f"  Claude                           {', '.join(ev['decisions']) or '-'}")
        print(f"  paused -> resumed                {m['paused_to_resumed']} ms")
        print(f"  resumed -> moving                {m['resumed_to_moving']} ms")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("config", nargs="?")
    ap.add_argument("--analyze", type=Path, help="only report on this finished run")
    ap.add_argument("--host")
    ap.add_argument("--camera-port", type=int)
    ap.add_argument("--contact-monitor", nargs="?", const="log", choices=["log", "enforce"])
    ap.add_argument("--hazard-monitor", action="store_true")
    ap.add_argument("--cycles", type=int)
    ap.add_argument("--tip-speed", type=float, help="m/s along the lines")
    ap.add_argument("--fast", action="store_true", help="sim only: run faster than real time")
    ap.add_argument(
        "--max-alarms", type=int, help="alarms per episode before the arm stays stopped"
    )
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    if args.analyze:
        report = analyze(args.analyze)
        (args.analyze / "reaction_report.json").write_text(json.dumps(report, indent=2) + "\n")
        show(report)
        return 0
    if not args.config:
        ap.error("give a config, or --analyze RUN")

    config = RunConfig.from_yaml(args.config)
    policy_cfg = config.source["policy"]
    if args.cycles:
        policy_cfg["cycles"] = args.cycles
    if args.tip_speed:
        policy_cfg["tip_speed"] = args.tip_speed
    if args.host:
        config.robot["host"] = args.host
    if args.camera_port:
        config.robot["camera_port"] = args.camera_port
    if args.hazard_monitor:
        config.safety = with_hazard_monitor(config.safety)
    if args.contact_monitor:
        config.safety = with_contact_monitor(config.safety, mode=args.contact_monitor)
    if args.max_alarms:
        cfg = config.safety
        while cfg is not None:
            if "review" in cfg:
                cfg["review"]["max_alarms"] = args.max_alarms
            cfg = cfg.get("inner")
    is_sim = config.robot.get("type") == "mujoco"
    clock = ManualClock() if (args.fast and is_sim) else WallClock()
    system = build_system(config, clock=clock)
    policy = system.source.policy
    out = args.out or ROOT / "runs" / f"sweep_{time.strftime('%Y%m%d-%H%M%S')}"
    log = RunLog(out)
    policy.bind(robot=system.robot, run_log=log, gate=None)
    if hasattr(system.safety, "bind"):
        system.safety.bind(
            run_log=log,
            robot=system.robot,
            phase_of=lambda: policy.phase,
            context=SWEEP_CONTEXT,
        )
    system.loop.recorder = TelemetryRecorder(
        out,
        phase_of=lambda: policy.phase,
        control_hz=system.spec.control_hz,
        meta={
            "config": str(args.config),
            "contact_monitor": args.contact_monitor,
            "hazard_monitor": args.hazard_monitor,
        },
    )
    print(f"run: {out}")
    try:
        result = system.loop.run_episode(seed=config.seed)
    finally:
        system.robot.disconnect()
        if hasattr(system.safety, "close"):
            system.safety.close()
    print(
        f"\n{result.outcome.value}"
        + (f" ({result.failure_tag})" if result.failure_tag else "")
        + f" in {result.duration:.0f}s"
    )
    report = analyze(out)
    (out / "reaction_report.json").write_text(json.dumps(report, indent=2) + "\n")
    show(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
