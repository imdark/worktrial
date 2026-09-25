"""A/B: the pick task with Claude-only vision vs laya-vision for verify.

    third_party/laya-vision/.venv/bin/python scripts/laya_server.py   # in another shell
    python scripts/vision_ab.py --scenes 12                            # all arms
    python scripts/vision_ab.py --scenes 1 --arms oracle               # smoke test, no API

Arms, each run on the same randomized scenes (paired):

* ``claude`` -- Claude Opus 5.5 plans, Sonnet 5 locates and verifies.
* ``laya``   -- laya-vision confirms grasps; Claude does everything else,
  including every grasp laya-vision does not confirm.
* ``oracle`` -- privileged sim state. A reference, not a contender: its
  failures are physics or reach, which tells the other arms' failures apart.

Ground truth never steers a decision. A probe wraps the arm's vision, passes
every question through unchanged, and records the true answer alongside:
where the target really is, whether it really was lifted. laya-vision is also
asked, silently, every verify question in every arm, so both models are
scored on identical frames.

Each scene places both glasses at random and asks for the left or the right
one, so `plan` (which glass?) is tested, not just `locate`. Success is judged
from the final sim state, not from what the task reports: the target lifted,
put back upright, the other glass left alone.

Writes outputs/vision_ab/<stamp>/: episodes.jsonl, summary.md, and one run
log (events, calls, images) per episode.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from teleop_sim.core.clock import ManualClock  # noqa: E402
from teleop_sim.core.config import RunConfig, build_system  # noqa: E402
from teleop_sim.perception.laya_vision import VERIFY_QUESTION, LayaVision  # noqa: E402
from teleop_sim.perception.oracle import OracleVision  # noqa: E402
from teleop_sim.perception.vision import Vision, VisionError  # noqa: E402
from teleop_sim.runlog import RunLog  # noqa: E402

CONFIG = ROOT / "teleop_sim/configs/yam_kronos_pick_sim.yaml"
CLAUDE = {"type": "claude", "fast_model": "claude-sonnet-5",
          "smart_model": "claude-opus-5-5", "escalate_below": 0.5}
ARMS = {
    "oracle": None,  # built per scene: it needs the target body
    "claude": CLAUDE,
    "laya": {"type": "laya", "fallback": CLAUDE},
}
#: $ per million tokens (input, output), Anthropic pricing page, 2026-09-24.
PRICES = {"claude-sonnet-5": (2.0, 10.0), "claude-opus-5-5": (4.0, 20.0)}
LIFTED_MM = 30.0     # the oracle's own threshold for "held and lifted"
UPRIGHT_DEG = 10.0


def make_scene(k: int) -> dict:
    """Glass positions and the target side for scene k. Deterministic in k."""
    rng = np.random.default_rng(1000 + k)
    return {
        "glass_left": (float(rng.uniform(0.35, 0.45)), float(rng.uniform(0.045, 0.10))),
        "glass_right": (float(rng.uniform(0.35, 0.45)), float(rng.uniform(-0.10, -0.045))),
        "side": "right" if k % 2 == 0 else "left",
    }


class Probe(Vision):
    """Pass-through vision that records ground truth next to every answer."""

    def __init__(self, inner: Vision, truth: OracleVision, other: OracleVision,
                 shadow: LayaVision | None) -> None:
        self.inner, self.truth, self.other, self.shadow = inner, truth, other, shadow
        self.records: list[dict] = []

    def plan(self, views, instruction):
        answer = self.inner.plan(views, instruction)
        self.records.append({"q": "plan", "description": answer.description,
                             "feasible": answer.feasible})
        return answer

    def locate(self, image, camera, target):
        det = self.inner.locate(image, camera, target)
        rec = {"q": "locate", "found": det.found, "model": det.model}
        want = self.truth.locate(image, camera, target)
        other = self.other.locate(image, camera, target)
        if det.found and det.base is not None and want.found:
            err = float(np.hypot(det.base.u - want.base.u, det.base.v - want.base.v))
            rec["px_err"] = round(err, 1)
            if other.found:
                err_other = float(np.hypot(det.base.u - other.base.u, det.base.v - other.base.v))
                rec["wrong_glass"] = err_other < err
        rec["target_visible"] = want.found
        self.records.append(rec)
        return det

    def verify(self, views, question):
        truth = self.truth.verify(views, question).ok
        p_laya = None
        if self.shadow is not None and "wrist" in views.images:
            try:
                q = {"held": {"type": "noul", "instructions": VERIFY_QUESTION}}
                p_laya = float(self.shadow._predict(views.images["wrist"], q, "verify_shadow")
                               ["answers"]["held"]["noul"])
            except VisionError:
                pass
        verdict = self.inner.verify(views, question)
        self.records.append({"q": "verify", "truth": truth, "ok": verdict.ok,
                             "confidence": verdict.confidence, "model": verdict.model,
                             "p_laya": p_laya})
        return verdict


def run_episode(arm: str, k: int, scene: dict, out: Path, shadow_endpoint: str | None) -> dict:
    cfg = RunConfig.from_yaml(CONFIG)
    target = f"glass_{scene['side']}"
    other_name = "glass_left" if scene["side"] == "right" else "glass_right"
    cfg.source["policy"]["instruction"] = f"pick up the glass on the {scene['side']}"
    system = build_system(cfg, clock=ManualClock())
    robot, policy = system.robot, system.source.policy
    model, data = robot.model, robot.data
    bodies = {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)
              for n in ("glass_left", "glass_right")}
    qadr = {n: model.jnt_qposadr[model.body_jntadr[b]] for n, b in bodies.items()}

    base_reset = robot.reset

    def reset(seed=None):
        base_reset(seed)
        for name in bodies:
            data.qpos[qadr[name]: qadr[name] + 2] = scene[name]
        mujoco.mj_forward(model, data)
        return robot.get_observation()

    robot.reset = reset
    reset()
    rest = {n: data.xpos[b].copy() for n, b in bodies.items()}

    log = RunLog(out / f"scene{k:02d}_{arm}")
    policy.bind(robot=robot, run_log=log)
    height = policy.glass_height
    truth = OracleVision(robot, target, height=height)
    other = OracleVision(robot, other_name, height=height)
    inner = truth if arm == "oracle" else policy._build_vision(ARMS[arm])
    shadow = LayaVision(endpoint=shadow_endpoint, run_log=log) if shadow_endpoint else None
    probe = Probe(inner, truth, other, shadow)
    policy.vision = probe

    def tilt(name: str) -> float:
        z = data.xmat[bodies[name]].reshape(3, 3)[2, 2]
        return float(np.degrees(np.arccos(np.clip(z, -1, 1))))

    peak = {n: 0.0 for n in bodies}
    base_event = policy.log.event

    def event(kind, **fields):
        for n, b in bodies.items():
            peak[n] = max(peak[n], 1000 * float(data.xpos[b][2] - rest[n][2]))
        return base_event(kind, **fields)

    policy.log.event = event
    started = time.monotonic()
    result = system.loop.run_episode(seed=0)
    wall = time.monotonic() - started

    moved = {n: 1000 * float(np.linalg.norm(data.xpos[b][:2] - rest[n][:2]))
             for n, b in bodies.items()}
    true_success = bool(
        peak[target] >= LIFTED_MM and tilt(target) < UPRIGHT_DEG
        and peak[other_name] < 10.0 and tilt(other_name) < UPRIGHT_DEG
    )
    calls = [json.loads(line) for line in (log.root / "calls.jsonl").read_text().splitlines()] \
        if (log.root / "calls.jsonl").exists() else []
    decision_calls = [c for c in calls if c.get("purpose") != "verify_shadow"]
    cost = sum(
        c["usage"]["input_tokens"] * PRICES[c["model"]][0] / 1e6
        + c["usage"]["output_tokens"] * PRICES[c["model"]][1] / 1e6
        for c in decision_calls if c.get("model") in PRICES and "usage" in c
    )
    return {
        "arm": arm, "scene": k, **scene, "target": target,
        "reported": result.outcome.value, "failure_tag": result.failure_tag,
        "true_success": true_success,
        "target_lift_mm": round(peak[target]), "other_lift_mm": round(peak[other_name]),
        "target_tilt_deg": round(tilt(target), 1), "other_moved_mm": round(moved[other_name]),
        "vision_s": round(sum(c.get("latency_s", 0.0) for c in decision_calls), 2),
        "motion_s": round(result.duration, 1),
        "wall_s": round(wall, 1),
        "calls": {m: sum(c.get("model") == m for c in decision_calls)
                  for m in ("claude-opus-5-5", "claude-sonnet-5", "laya-vision")},
        "errors": [c["error"] for c in decision_calls if "error" in c],
        "cost_usd": round(cost, 4),
        "probe": probe.records,
    }


# ------------------------------------------------------------------ summary


def _pct(n: int, d: int) -> str:
    return f"{n}/{d} ({100 * n / d:.0f}%)" if d else "–"


def _episode_table(eps: list[dict], arms: list[str]) -> list[str]:
    lines = ["| Arm | True success | Reported success | False success | Vision time / ep | "
             "Motion time / ep | Est. real time / ep | API cost / ep | "
             "Opus / Sonnet / Laya calls |",
             "|---|---|---|---|---|---|---|---|---|"]
    for a in arms:
        es = [e for e in eps if e["arm"] == a]
        n = len(es)
        ok = sum(e["true_success"] for e in es)
        rep = sum(e["reported"] == "success" for e in es)
        false = sum(e["reported"] == "success" and not e["true_success"] for e in es)
        v, m = np.mean([e["vision_s"] for e in es]), np.mean([e["motion_s"] for e in es])
        c = np.mean([e["cost_usd"] for e in es])
        calls = [sum(e["calls"][k] for e in es) for k in ("claude-opus-5-5", "claude-sonnet-5",
                                                          "laya-vision")]
        lines.append(f"| {a} | {_pct(ok, n)} | {_pct(rep, n)} | {false} | {v:.1f} s | {m:.1f} s | "
                     f"{v + m:.1f} s | ${c:.3f} | {calls[0]} / {calls[1]} / {calls[2]} |")
    return lines


def summarize(eps: list[dict]) -> str:
    arms = [a for a in ARMS if any(e["arm"] == a for e in eps)]
    lines = ["## Episodes, all scenes", "", *_episode_table(eps, arms)]
    solved = {e["scene"] for e in eps if e["arm"] == "oracle" and e["true_success"]}
    if any(e["arm"] == "oracle" for e in eps):
        lines += ["", f"## Episodes, only the {len(solved)} scenes the oracle solves", "",
                  "Failures here are vision's: perfect vision succeeds on these scenes.", "",
                  *_episode_table([e for e in eps if e["scene"] in solved], arms)]

    lines += ["", "## Per-question accuracy (vs ground truth)", "",
              "| Arm | locate: found | locate: wrong glass | locate: median px error | "
              "verify: decided right | verify: decided by laya |", "|---|---|---|---|---|---|"]
    for a in arms:
        rs = [r for e in eps if e["arm"] == a for r in e["probe"]]
        loc = [r for r in rs if r["q"] == "locate" and r["target_visible"]]
        errs = [r["px_err"] for r in loc if "px_err" in r]
        ver = [r for r in rs if r["q"] == "verify"]
        by_laya = [r for r in ver if r["model"] == "laya-vision"]
        median = f"{np.median(errs):.0f} px" if errs else "–"
        lines.append(
            f"| {a} | {_pct(sum(r['found'] for r in loc), len(loc))} | "
            f"{sum(r.get('wrong_glass', False) for r in loc)} | {median} | "
            f"{_pct(sum(r['ok'] == r['truth'] for r in ver), len(ver))} | "
            f"{_pct(sum(r['ok'] == r['truth'] for r in by_laya), len(by_laya))} right |"
        )

    shadow = [r for e in eps for r in e["probe"] if r["q"] == "verify" and r["p_laya"] is not None]
    if shadow:
        pos = sorted(r["p_laya"] for r in shadow if r["truth"])
        neg = sorted(r["p_laya"] for r in shadow if not r["truth"])
        lines += ["", "## laya-vision on every verify frame (shadow, all arms)", "",
                  f"- frames: {len(shadow)} ({len(pos)} truly held, {len(neg)} not)",
                  f"- P(held) on held frames: {[round(p, 2) for p in pos]}",
                  f"- P(held) on not-held frames: {[round(p, 2) for p in neg]}"]
        for th in (0.5, 0.7):
            right = sum((r["p_laya"] >= th) == r["truth"] for r in shadow)
            fa = sum(r["p_laya"] >= th and not r["truth"] for r in shadow)
            lines.append(f"- alone at threshold {th}: {_pct(right, len(shadow))} right, "
                         f"{fa} false 'held'")
        claude = [r for e in eps if e["arm"] == "claude" for r in e["probe"] if r["q"] == "verify"]
        if claude:
            lines.append(f"- Claude (arm `claude`) on its verify frames: "
                         f"{_pct(sum(r['ok'] == r['truth'] for r in claude), len(claude))} right")

    lines += ["", "## Failures", ""]
    for e in eps:
        if not e["true_success"]:
            lines.append(f"- scene {e['scene']} `{e['arm']}` ({e['side']}): reported "
                         f"{e['reported']} {e['failure_tag'] or ''}; target lifted "
                         f"{e['target_lift_mm']} mm, other glass lifted {e['other_lift_mm']} mm "
                         f"/ moved {e['other_moved_mm']} mm" + (f"; errors: {e['errors']}"
                                                                 if e["errors"] else ""))
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--scenes", type=int, default=12)
    ap.add_argument("--first-scene", type=int, default=0)
    ap.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    ap.add_argument("--laya", default="http://127.0.0.1:8765", help="laya server ('' = none)")
    ap.add_argument("--out", default=str(ROOT / "outputs/vision_ab"))
    args = ap.parse_args()

    out = Path(args.out) / time.strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True)
    endpoint = args.laya or None
    if endpoint:
        import urllib.request
        try:
            urllib.request.urlopen(f"{endpoint}/health", timeout=2).read()
        except OSError as exc:
            sys.exit(f"laya server not reachable at {endpoint} ({exc}); start it or pass --laya ''")

    episodes = []
    with (out / "episodes.jsonl").open("w") as fh:
        for k in range(args.first_scene, args.first_scene + args.scenes):
            scene = make_scene(k)
            for arm in args.arms:  # interleaved, so API conditions spread over arms
                ep = run_episode(arm, k, scene, out, endpoint)
                episodes.append(ep)
                fh.write(json.dumps(ep) + "\n")
                fh.flush()
                print(f"[ab] scene {k:2d} {arm:6s} {scene['side']:5s} true_success="
                      f"{ep['true_success']!s:5s} reported={ep['reported']:7s} "
                      f"vision={ep['vision_s']:5.1f}s motion={ep['motion_s']:5.1f}s "
                      f"${ep['cost_usd']:.3f}", flush=True)
    summary = summarize(episodes)
    (out / "summary.md").write_text(summary)
    print("\n" + summary + f"\nwritten to {out}")


if __name__ == "__main__":
    main()
