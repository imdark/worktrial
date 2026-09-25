"""Plot gripper current against time, shaded by the held / not-held label.

    python scripts/plot_gripper_current.py runs/current_sample_<ts>/ep01 [--out plot.svg]

Reads telemetry.jsonl from a real-rig episode (teleop_sim/telemetry.py records
``motors.gripper_eff`` from rr_bridge: on gem13's Kronos, the summed |present
current| of its two XC330 servos, raw units ~mA) and writes a standalone SVG:

    top     gripper current, background shaded by each step's label
            (held / not_held / uncertain, from the jaws -- see label_frame)
    bottom  jaw opening, measured and commanded (0 = open, 1 = closed)

Phase names mark where each phase starts. Also prints current by label and how
well ``current >= --threshold`` alone matches the labels; --csv and --json
save the time series and those numbers. No plotting library needed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

COLORS = {"held": "#2e9d57", "not_held": "#9aa3ad", "uncertain": "#e3b341"}
W, H = 1200, 560
L, R = 70, 20
TOP = (40, 290)  # current panel y range
BOT = (340, 510)  # jaw panel y range


def load(ep: Path) -> list[dict]:
    rows = [json.loads(line) for line in (ep / "telemetry.jsonl").read_text().splitlines()]
    rows = [r for r in rows if r.get("motors", {}).get("gripper_eff") is not None]
    if not rows:
        raise SystemExit(f"{ep}: no motors.gripper_eff in telemetry (recorded before 2026-09-25?)")
    return rows


def svg(rows: list[dict], title: str) -> str:
    t = np.array([r["wall"] for r in rows])
    t = t - t[0]
    cur = np.array([r["motors"]["gripper_eff"][0] for r in rows])
    meas = np.array([r["gripper"] for r in rows])
    cmd = np.array([r["cmd_gripper"] for r in rows])
    labels = [r.get("label", "uncertain") for r in rows]
    tmax = float(t[-1]) or 1.0
    cmax = max(50.0, float(np.percentile(cur, 99.5)) * 1.1)

    def x(v):
        return L + (W - L - R) * v / tmax

    def yc(v):
        return TOP[1] - (TOP[1] - TOP[0]) * min(v, cmax) / cmax

    def yj(v):
        return BOT[1] - (BOT[1] - BOT[0]) * v

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
        f'font-family="Helvetica,Arial,sans-serif" font-size="11">',
        f'<rect width="{W}" height="{H}" fill="white"/>',
        f'<text x="{L}" y="22" font-size="14" font-weight="bold">{title}</text>',
    ]
    # label shading, as runs of equal labels
    start = 0
    for i in range(1, len(rows) + 1):
        if i == len(rows) or labels[i] != labels[start]:
            x0, x1 = x(t[start]), x(t[i - 1] if i == len(rows) else t[i])
            for y0, y1 in (TOP, BOT):
                out.append(
                    f'<rect x="{x0:.1f}" y="{y0}" width="{max(x1 - x0, 0.5):.1f}" '
                    f'height="{y1 - y0}" fill="{COLORS[labels[start]]}" opacity="0.22"/>'
                )
            start = i
    # phase starts
    prev = None
    for ti, r in zip(t, rows, strict=True):
        if r["phase"] != prev:
            xi = x(ti)
            out.append(
                f'<line x1="{xi:.1f}" y1="{TOP[0]}" x2="{xi:.1f}" y2="{BOT[1]}" '
                'stroke="#555" stroke-width="0.5" stroke-dasharray="2,3"/>'
            )
            out.append(
                f'<text x="{xi + 2:.1f}" y="{TOP[0] + 10}" font-size="9" fill="#333" '
                f'transform="rotate(90 {xi + 2:.1f} {TOP[0] + 10})">{r["phase"]}</text>'
            )
            prev = r["phase"]

    def line(ys, color, width=1.4, dash=""):
        pts = " ".join(f"{x(a):.1f},{b:.1f}" for a, b in zip(t, ys, strict=True))
        extra = f' stroke-dasharray="{dash}"' if dash else ""
        return (
            f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="{width}"{extra}/>'
        )

    out.append(line([yc(v) for v in cur], "#c0392b", 1.5))
    out.append(line([yj(v) for v in cmd], "#2c3e50", 1.2, "4,3"))
    out.append(line([yj(v) for v in meas], "#2471a3", 1.6))
    # axes
    for (y0, y1), name in ((TOP, "gripper current (raw, ~mA)"), (BOT, "jaw (0 open, 1 closed)")):
        out.append(
            f'<rect x="{L}" y="{y0}" width="{W - L - R}" height="{y1 - y0}" '
            'fill="none" stroke="#333"/>'
        )
        out.append(
            f'<text x="14" y="{(y0 + y1) / 2}" transform="rotate(-90 14 '
            f'{(y0 + y1) / 2})" text-anchor="middle">{name}</text>'
        )
    for v in np.linspace(0, cmax, 5):
        out.append(f'<text x="{L - 6}" y="{yc(v) + 4:.1f}" text-anchor="end">{v:.0f}</text>')
    for v in (0, 0.5, 1):
        out.append(f'<text x="{L - 6}" y="{yj(v) + 4:.1f}" text-anchor="end">{v}</text>')
    for s in range(0, int(tmax) + 1, 10):
        out.append(f'<text x="{x(s):.1f}" y="{BOT[1] + 16}" text-anchor="middle">{s}s</text>')
    # legend
    lx = W - R - 520
    items = [
        ("held", COLORS["held"]),
        ("not held", COLORS["not_held"]),
        ("uncertain", COLORS["uncertain"]),
    ]
    for k, (name, col) in enumerate(items):
        out.append(
            f'<rect x="{lx + 80 * k}" y="{H - 22}" width="12" height="12" '
            f'fill="{col}" opacity="0.5"/>'
        )
        out.append(f'<text x="{lx + 80 * k + 16}" y="{H - 12}">{name}</text>')
    for k, (name, col, dash) in enumerate(
        [
            ("current", "#c0392b", ""),
            ("jaw measured", "#2471a3", ""),
            ("jaw commanded", "#2c3e50", "4,3"),
        ]
    ):
        xx = lx + 250 + 95 * k
        out.append(
            f'<line x1="{xx}" y1="{H - 16}" x2="{xx + 18}" y2="{H - 16}" '
            f'stroke="{col}" stroke-width="2"'
            + (f' stroke-dasharray="{dash}"' if dash else "")
            + "/>"
        )
        out.append(f'<text x="{xx + 22}" y="{H - 12}">{name}</text>')
    out.append("</svg>")
    return "\n".join(out)


def stats(rows: list[dict], threshold: float) -> dict:
    """Current by label, and how well ``current >= threshold`` alone matches the labels."""
    by: dict[str, list[float]] = {}
    for r in rows:
        by.setdefault(r.get("label", "uncertain"), []).append(r["motors"]["gripper_eff"][0])
    # closed on nothing: commanded closed, jaws shut -- the hard negative
    empty = [
        r["motors"]["gripper_eff"][0]
        for r in rows
        if r["cmd_gripper"] > 0.95 and r["gripper"] >= 0.9
    ]
    if empty:
        by["closed_on_nothing"] = empty
    out: dict = {"steps": len(rows), "seconds": round(rows[-1]["wall"] - rows[0]["wall"], 1)}
    out["current_by_label"] = {
        k: {
            "n": len(v),
            "median": float(np.median(v)),
            "p10": float(np.percentile(v, 10)),
            "p90": float(np.percentile(v, 90)),
        }
        for k, v in sorted(by.items())
    }
    y = np.array([r["label"] == "held" for r in rows if r["label"] != "uncertain"])
    c = np.array([r["motors"]["gripper_eff"][0] for r in rows if r["label"] != "uncertain"])
    pred = c >= threshold
    out["threshold_rule"] = {
        "rule": f"held iff gripper current >= {threshold:g}",
        "n": int(y.size),
        "agreement": round(float((pred == y).mean()), 4),
        "held_recall": round(float(pred[y].mean()), 4) if y.any() else None,
        "not_held_recall": round(float((~pred[~y]).mean()), 4) if (~y).any() else None,
    }
    return out


def write_csv(rows: list[dict], path: Path) -> None:
    t0 = rows[0]["wall"]
    lines = ["t_s,phase,label,jaw_measured,jaw_commanded,gripper_current,gripper_vel"]
    for r in rows:
        m = r["motors"]
        vel = m.get("gripper_vel", [float("nan")])[0]
        lines.append(
            f"{r['wall'] - t0:.3f},{r['phase']},{r.get('label', '')},{r['gripper']},"
            f"{r['cmd_gripper']},{m['gripper_eff'][0]},{vel}"
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("episode", type=Path, help="directory with telemetry.jsonl")
    ap.add_argument("--out", type=Path, default=None, help="SVG path")
    ap.add_argument("--csv", type=Path, help="also write the time series as CSV")
    ap.add_argument("--json", type=Path, help="also write the statistics as JSON")
    ap.add_argument("--threshold", type=float, default=150.0)
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    rows = load(args.episode)
    out = args.out or args.episode / "gripper_current.svg"
    out.write_text(svg(rows, args.title or f"Gripper current vs label: {args.episode}"))
    print(f"wrote {out}")
    st = stats(rows, args.threshold)
    for k, v in st["current_by_label"].items():
        print(
            f"  {k:20s} n={v['n']:4d}  current median {v['median']:6.1f}  "
            f"p10 {v['p10']:6.1f}  p90 {v['p90']:6.1f}"
        )
    tr = st["threshold_rule"]
    print(f"  {tr['rule']}: agreement {tr['agreement']:.2%} over {tr['n']} labelled steps")
    if args.csv:
        write_csv(rows, args.csv)
        print(f"wrote {args.csv}")
    if args.json:
        args.json.write_text(json.dumps({"episode": str(args.episode), **st}, indent=2) + "\n")
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
