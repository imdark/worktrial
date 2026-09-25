"""Per-step telemetry and labelled camera frames, for training vision models.

A Recorder the control loop calls every step. It writes, into a run directory:

    telemetry.jsonl   one line per control step: time, phase, joints, gripper
                      (measured), commanded joints and gripper, action staleness,
                      the step's label, and on the real rig raw motor readings
                      (gripper current and velocity, arm torques)
    frames/*.jpg      camera frames at ``image_hz`` (wrist and overview)
    labels.jsonl      one line per saved wrist frame: is the object held?

Labels come from the gripper's physical state, never from a vision model, so
they can train and score one:

    not_held    jaws commanded open and measured open
    held        jaws commanded closed and stopped part-way (something between
                them), in a phase where the object is off or leaving the table
    not_held    jaws commanded closed and measured fully closed: closed on
                nothing -- the hard negative
    uncertain   jaws moving, or any other state; kept, but not for training

laya-vision's verify question is "Is the gripper holding a cup between its
closed fingers?", asked of the wrist image; ``held`` / ``not_held`` answer it.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from teleop_sim.core.protocols import Recorder

#: Phases in which a stopped, closed gripper means the object is held.
HELD_PHASES = ("close", "lift", "hold", "carry", "lower")


def label_frame(
    phase: str, commanded: float, measured: float, empty_above: float = 0.9
) -> tuple[str, str]:
    """(label, reason) for one wrist frame from the gripper's state."""
    if commanded < 0.05 and measured < 0.1:
        return "not_held", "jaws open"
    if commanded > 0.95:
        if measured >= empty_above:
            return "not_held", f"jaws closed fully ({measured:.2f}): nothing between them"
        if 0.05 < measured < empty_above and phase in HELD_PHASES:
            return "held", f"jaws stopped at {measured:.2f} while closed"
    return "uncertain", f"jaws moving or ambiguous (cmd {commanded:.2f}, meas {measured:.2f})"


class TelemetryRecorder(Recorder):
    def __init__(
        self,
        root: str | Path,
        phase_of,
        cameras=("wrist", "overview"),
        image_hz: float = 2.0,
        control_hz: float = 30.0,
        empty_above: float = 0.9,
        meta: dict[str, Any] | None = None,
    ) -> None:
        self.root = Path(root)
        self.phase_of = phase_of
        self.cameras = tuple(cameras)
        self.every = max(1, round(control_hz / image_hz))
        self.empty_above = float(empty_above)
        self.meta = dict(meta or {})
        self._step = 0
        self._tele = None
        self._labels = None
        self.counts: dict[str, int] = {}

    def start_episode(self, seed: int | None = None) -> None:
        (self.root / "frames").mkdir(parents=True, exist_ok=True)
        self._tele = open(self.root / "telemetry.jsonl", "a")
        self._labels = open(self.root / "labels.jsonl", "a")
        self._step = 0
        (self.root / "meta.json").write_text(
            json.dumps({"started": time.time(), "seed": seed, **self.meta}, indent=2)
        )

    def record(self, obs, action) -> None:
        phase = str(self.phase_of())
        line = {
            "step": self._step,
            "t": round(float(obs.timestamp), 4),
            "wall": round(time.time(), 3),
            "phase": phase,
            "joint_pos": np.round(obs.joint_pos, 5).tolist(),
            "gripper": round(float(obs.gripper), 4),
            "cmd_joint_pos": np.round(action.values, 5).tolist(),
            "cmd_gripper": round(float(action.gripper), 4),
            "staleness_ms": None
            if action.obs_timestamp is None
            else round((float(action.timestamp) - float(action.obs_timestamp)) * 1000, 1),
        }
        if obs.ee_pose is not None:
            line["ee_pose"] = np.round(obs.ee_pose, 5).tolist()
        motors = obs.extra.get("motors") if obs.extra else None
        if motors:  # real rig: gripper current / velocity, arm torques (rr_bridge)
            line["motors"] = {k: [round(v, 4) for v in vals] for k, vals in motors.items()}
        line["label"] = label_frame(
            phase, float(action.gripper), float(obs.gripper), self.empty_above
        )[0]
        if self._step % self.every == 0:
            import cv2

            saved = {}
            for name in self.cameras:
                img = obs.images.get(name)
                if img is None:
                    continue
                fname = f"frames/{self._step:06d}_{name}.jpg"
                cv2.imwrite(
                    str(self.root / fname),
                    np.ascontiguousarray(img[:, :, ::-1]),
                    [cv2.IMWRITE_JPEG_QUALITY, 90],
                )
                saved[name] = fname
            line["frames"] = saved
            if "wrist" in saved:
                label, reason = label_frame(
                    phase, float(action.gripper), float(obs.gripper), self.empty_above
                )
                self.counts[label] = self.counts.get(label, 0) + 1
                self._labels.write(
                    json.dumps(
                        {
                            "image": saved["wrist"],
                            "label": label,
                            "reason": reason,
                            "phase": phase,
                            "gripper": round(float(obs.gripper), 4),
                            "cmd_gripper": round(float(action.gripper), 4),
                            "step": self._step,
                        }
                    )
                    + "\n"
                )
        self._tele.write(json.dumps(line) + "\n")
        self._step += 1

    def end_episode(self, result) -> None:
        for fh in (self._tele, self._labels):
            if fh is not None:
                fh.close()
        summary = {"label_counts": self.counts}
        if result is not None:
            summary["result"] = result.to_record()
        (self.root / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    def discard_episode(self) -> None:
        self.end_episode(None)
