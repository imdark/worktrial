"""Fine-tune laya-vision's "is the cup held?" answer on gem13's own wrist frames.

Runs in laya-vision's venv (it needs transformers >= 5.3; see
scripts/setup_laya_vision.sh), on a dataset from scripts/export_laya_dataset.py:

    PY=third_party/laya-vision/.venv/bin/python
    $PY scripts/train_laya_verify.py --eval-only                      # zero-shot baseline
    $PY scripts/train_laya_verify.py --steps 300 --freeze last_n      # fine-tune

What it does:

1. Loads the pinned checkpoint (the one scripts/laya_server.py serves).
2. Scores it on val and test exactly as the rig asks: agent.predict on the
   wrist image with laya_vision.VERIFY_QUESTION, one image at a time.
3. Unless --eval-only: trains on train (laya.vlm_train.train), fits the
   confidence temperature on val (fit_temperatures), scores val and test
   again, and saves the checkpoint to --out (outside git: the weights are
   CC BY-NC-SA 4.0, non-commercial only).

Everything it measured goes to <out>/report.json: accuracy at 0.5, balanced
accuracy, AUROC, Brier, ECE, and P(held) per phase -- the phases that matter
are reach_in / close / open, where the cup is between the fingers.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL_ID = "thaitea/laya-vision"
REVISION = "8b318c99d7ad3ce19c24369263463882eada9d1e"  # keep in sync with laya_server.py


def load_records(data: Path, split: str) -> list[dict]:
    path = data / "gem13_held" / f"{split}.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def p_held(agent, records: list[dict], data: Path) -> list[float]:
    from PIL import Image

    out = []
    for rec in records:
        image = Image.open(data / "gem13_held" / rec["image"]).convert("RGB")
        answer = agent.predict(
            {"image": image},
            {"held": {"type": "noul", "instructions": rec["question"]["instructions"]}},
        )
        out.append(float(answer["answers"]["held"]["noul"]))
    return out


def auroc(scores: list[float], labels: list[int]) -> float:
    pos = [s for s, y in zip(scores, labels, strict=True) if y == 1]
    neg = [s for s, y in zip(scores, labels, strict=True) if y == 0]
    if not pos or not neg:
        return float("nan")
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def metrics(records: list[dict], probs: list[float]) -> dict:
    labels = [r["label"] for r in records]
    pred = [int(p >= 0.5) for p in probs]
    tp = sum(p == y == 1 for p, y in zip(pred, labels, strict=True))
    tn = sum(p == y == 0 for p, y in zip(pred, labels, strict=True))
    n_pos, n_neg = sum(labels), len(labels) - sum(labels)
    bins = defaultdict(list)
    for p, y in zip(probs, labels, strict=True):
        conf = max(p, 1 - p)
        bins[min(int(conf * 10), 9)].append((conf, int((p >= 0.5) == y)))
    ece = sum(
        len(b) / len(labels) * abs(sum(c for c, _ in b) / len(b) - sum(a for _, a in b) / len(b))
        for b in bins.values()
    )
    by_phase = defaultdict(list)
    for r, p in zip(records, probs, strict=True):
        by_phase[(r["phase"], "held" if r["label"] else "not_held")].append(p)
    return {
        "n": len(labels),
        "held": n_pos,
        "not_held": n_neg,
        "accuracy": round((tp + tn) / len(labels), 4),
        "balanced_accuracy": round(0.5 * (tp / max(n_pos, 1) + tn / max(n_neg, 1)), 4),
        "recall_held": round(tp / max(n_pos, 1), 4),
        "recall_not_held": round(tn / max(n_neg, 1), 4),
        "auroc": round(auroc(probs, labels), 4),
        "brier": round(
            sum((p - y) ** 2 for p, y in zip(probs, labels, strict=True)) / len(labels), 4
        ),
        "ece": round(ece, 4),
        "mean_p_held_by_phase": {
            f"{ph}/{lab}": round(sum(v) / len(v), 3) for (ph, lab), v in sorted(by_phase.items())
        },
    }


def show(name: str, m: dict) -> None:
    print(
        f"  {name}: acc {m['accuracy']:.3f}  bal-acc {m['balanced_accuracy']:.3f}  "
        f"recall held {m['recall_held']:.3f} / not held {m['recall_not_held']:.3f}  "
        f"AUROC {m['auroc']:.3f}  Brier {m['brier']:.3f}  ECE {m['ece']:.3f}",
        flush=True,
    )


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "laya")
    ap.add_argument("--init", default=MODEL_ID)
    ap.add_argument("--revision", default=REVISION)
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--freeze", choices=["head", "last_n", "full"], default="last_n")
    ap.add_argument("--n-last", type=int, default=4)
    ap.add_argument("--lr-head", type=float, default=1e-4)
    ap.add_argument("--lr-backbone", type=float, default=2e-5)
    ap.add_argument("--max-minutes", type=float, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT / "third_party" / "laya-vision-ft" / time.strftime("gem13_held_%Y%m%d-%H%M%S"),
    )
    args = ap.parse_args()

    import laya
    from laya.vlm_train import fit_temperatures, load_jsonl_examples, train

    args.out.mkdir(parents=True, exist_ok=True)
    revision = args.revision if args.init == MODEL_ID else None
    agent = laya.load_vlm(args.init, revision=revision, device=args.device)
    print(f"[laya] {args.init}@{(revision or 'local')[:8]} on {agent.device}", flush=True)
    splits = {s: load_records(args.data, s) for s in ("train", "val", "test")}
    report: dict = {
        "init": args.init,
        "revision": revision,
        "data": str(args.data),
        "sizes": {s: len(r) for s, r in splits.items()},
    }

    print("zero-shot:", flush=True)
    report["zero_shot"] = {}
    for s in ("val", "test"):
        m = metrics(splits[s], p_held(agent, splits[s], args.data))
        report["zero_shot"][s] = m
        show(s, m)

    if not args.eval_only:
        examples = load_jsonl_examples(str(args.data), "gem13_held", "train")
        val_examples = load_jsonl_examples(str(args.data), "gem13_held", "val")
        t0 = time.time()
        losses = train(
            agent.model,
            agent.processor,
            examples,
            steps=args.steps,
            batch_size=args.batch_size,
            freeze=args.freeze,
            n_last=args.n_last,
            lr_head=args.lr_head,
            lr_backbone=args.lr_backbone,
            device=str(agent.device),
            max_minutes=args.max_minutes,
            log_every=25,
        )
        agent.model.eval()
        report["train"] = {
            "steps": len(losses),
            "final_loss": float(losses[-1]),
            "finite": math.isfinite(losses[-1]),
            "minutes": round((time.time() - t0) / 60, 1),
            "freeze": args.freeze,
            "n_last": args.n_last,
            "batch_size": args.batch_size,
        }
        agent.temperature = fit_temperatures(agent.model, agent.processor, val_examples)
        report["temperatures"] = [float(t) for t in agent.temperature]
        print(
            f"trained {len(losses)} steps in {report['train']['minutes']} min, final loss "
            f"{losses[-1]:.4f}; temperatures {report['temperatures']}",
            flush=True,
        )
        print("fine-tuned:", flush=True)
        report["fine_tuned"] = {}
        for s in ("val", "test"):
            m = metrics(splits[s], p_held(agent, splits[s], args.data))
            report["fine_tuned"][s] = m
            show(s, m)
        agent.save(str(args.out / "checkpoint"), include_backbone=args.freeze != "head")
        report["checkpoint"] = str(args.out / "checkpoint")
        print(f"saved {report['checkpoint']}", flush=True)
    (args.out / "report.json").write_text(json.dumps(report, indent=2))
    print(f"report: {args.out / 'report.json'}")
    return 0


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    raise SystemExit(main())
