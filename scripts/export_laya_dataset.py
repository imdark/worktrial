"""Turn collected pick batches into a laya-vision fine-tuning dataset.

    python scripts/export_laya_dataset.py runs/collect_20260925-133615 \
        --out data/laya --val 8 9 --test 10

Reads each batch's labels.jsonl (written by teleop_sim/telemetry.py through
scripts/collect_real_picks.py) and writes, in laya-vision's JSONL format
(laya/vlm_train.py: jsonl_example):

    <out>/gem13_held/{train,val,test}.jsonl
    <out>/gem13_held/images/<batch>/<episode>/<frame>.jpg   (copied)
    <out>/gem13_held/README.md                               counts and caveats

Each record asks laya-vision's own verify question of one wrist frame:

    {"id": "collect_..._ep03-000645", "image": "images/...jpg",
     "question": {"type": "noul", "instructions": "<VERIFY_QUESTION>"},
     "label": 1, "phase": "close", "gripper": 0.64, "source": "jaws stopped ..."}

label 1 = held, 0 = not held (noul: index 1 is true). What is left out:

    uncertain       jaws moving; the gripper cannot vouch for these
    first 'held' of each grasp   the jaws may still be closing the last few
                    hundredths when the command reaches closed (0.65 -> 0.72)

Splits are by episode, never by frame: neighbouring frames of one episode are
near-duplicates, and splitting them would make validation meaningless.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from teleop_sim.perception.laya_vision import VERIFY_QUESTION  # noqa: E402

DATASET = "gem13_held"
#: Phases where the cup is at or between the fingers: the negatives that matter.
NEAR_PHASES = {
    "standoff",
    "reach_in",
    "close",
    "open",
    "back_out",
    "put_down",
    "release",
    "back_off",
}


def records_from_batch(batch: Path) -> list[dict]:
    rows = [json.loads(line) for line in (batch / "labels.jsonl").read_text().splitlines()]
    rows.sort(key=lambda r: (r["episode"], r["step"]))
    out, prev_label = [], {}
    for r in rows:
        ep, label = r["episode"], r["label"]
        first_hold = label == "held" and prev_label.get(ep) != "held"
        prev_label[ep] = label
        if label == "uncertain" or first_hold:
            continue
        out.append({**r, "batch": batch.name})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("batches", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "laya")
    ap.add_argument("--val", type=int, nargs="*", default=[8, 9], help="episode numbers")
    ap.add_argument("--test", type=int, nargs="*", default=[10], help="episode numbers")
    ap.add_argument(
        "--far-negative-keep",
        type=float,
        default=1.0,
        help="fraction of open-jaw frames away from the cup to keep (0-1)",
    )
    args = ap.parse_args()

    base = args.out / DATASET
    if base.exists():
        shutil.rmtree(base)
    (base / "images").mkdir(parents=True)
    splits: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    kept_far = Counter()
    for batch in args.batches:
        for r in records_from_batch(batch):
            near = r["phase"] in NEAR_PHASES or r["label"] == "held"
            if not near:
                kept_far["seen"] += 1
                if (kept_far["seen"] * args.far_negative_keep) % 1 >= args.far_negative_keep:
                    continue
                kept_far["kept"] += 1
            split = (
                "test"
                if r["episode"] in args.test
                else "val"
                if r["episode"] in args.val
                else "train"
            )
            src = batch / r["image"]
            rel = Path("images") / batch.name / r["image"].replace("/frames/", "/")
            (base / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, base / rel)
            splits[split].append(
                {
                    "id": f"{batch.name}_ep{r['episode']:02d}-{r['step']:06d}",
                    "image": str(rel),
                    "question": {"type": "noul", "instructions": VERIFY_QUESTION},
                    "label": 1 if r["label"] == "held" else 0,
                    "phase": r["phase"],
                    "gripper": r["gripper"],
                    "cmd_gripper": r["cmd_gripper"],
                    "near_cup": near,
                    "source": r["reason"],
                }
            )
    lines = [
        f"# {DATASET}: is the gripper holding the cup? (gem13 wrist camera)",
        "",
        f'Question: "{VERIFY_QUESTION}" (noul; label 1 = held, 0 = not held).',
        "",
        "Labels come from the gripper, not a model: jaws commanded closed and stopped "
        "part-way -> held; jaws open -> not held. See teleop_sim/telemetry.py.",
        "",
        "| split | episodes | held | not held | of which near the cup |",
        "|---|---|---|---|---|",
    ]
    for name, recs in splits.items():
        with open(base / f"{name}.jsonl", "w") as fh:
            for rec in recs:
                fh.write(json.dumps(rec) + "\n")
        eps = sorted({rec["id"].rsplit("-", 1)[0].rsplit("_ep", 1)[1] for rec in recs})
        held = sum(rec["label"] for rec in recs)
        near_neg = sum(1 for rec in recs if rec["label"] == 0 and rec["near_cup"])
        lines.append(f"| {name} | {', '.join(eps)} | {held} | {len(recs) - held} | {near_neg} |")
        print(
            f"{name}: {len(recs)} frames ({held} held, {len(recs) - held} not held), episodes {eps}"
        )
    hard = sum(
        1
        for recs in splits.values()
        for rec in recs
        if rec["label"] == 0 and rec["cmd_gripper"] > 0.95
    )
    lines += [
        "",
        f"Hard negatives (jaws closed on nothing): {hard}.",
        "Without them a model can learn 'closed jaws = held'. Collect deliberate misses "
        "before trusting a fine-tuned 'no'.",
        "",
        f"Sources: {', '.join(b.name for b in args.batches)}. Weights of any model trained "
        "on this inherit laya-vision's CC BY-NC-SA 4.0 (non-commercial).",
    ]
    (base / "README.md").write_text("\n".join(lines) + "\n")
    print(f"hard negatives: {hard}\nwrote {base}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
