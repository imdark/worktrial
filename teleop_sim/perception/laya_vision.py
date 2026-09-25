"""Vision answered by laya-vision where a typed yes/no is enough, and by a
fallback backend everywhere else.

laya-vision (thaitea/laya-vision, SmolVLM-256M cut to 201M) returns calibrated
choice / score / yes-no probabilities in one forward pass -- no text is
generated. It cannot answer ``locate``: that needs pixel coordinates, and its
outputs are typed choices. So the split is:

* ``verify`` -> laya-vision, on the wrist camera. ~150 ms on an M5 (MPS),
  where Claude Sonnet 5 takes 2.5-4.8 s. It is only asked after the jaw
  check (pick_lift_place.py) found something between the fingers.
* An answer between ``reject_below`` and ``accept_above``, or a server that is
  down, is asked again of the fallback.
* ``plan`` and ``locate`` -> the fallback (Claude on hardware, the oracle in
  sim).

By default laya-vision may only *confirm* a grasp (``reject_below=0``): every
"no" goes to the fallback too. Zero-shot, measured 2026-09-25:

* Real wrist frames, through this client (JPEG): held 0.75 / 0.78, not held
  0.11-0.34.
* A sim wrist frame looking straight down into a held glass: **0.17** -- a
  confident, wrong "no" that cost a pointless retry.

Raise ``reject_below`` only after fine-tuning and calibrating on this rig's
own frames, sent the same way (JPEG shifted one held frame from 0.53 to 0.75).
Every call is logged to calls.jsonl with its image for exactly that.

The model runs in its own venv behind scripts/laya_server.py; this client uses
only the standard library. Weights are CC BY-NC-SA 4.0: non-commercial only.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any

import numpy as np

from teleop_sim.perception.camera import CameraModel
from teleop_sim.perception.claude_vision import encode_image
from teleop_sim.perception.vision import (
    Detection,
    TargetPlan,
    Verdict,
    Views,
    Vision,
    VisionError,
)
from teleop_sim.runlog import RunLog

MODEL_NAME = "laya-vision"
VERIFY_QUESTION = "Is the gripper holding a cup between its closed fingers?"


class LayaVision(Vision):
    def __init__(
        self,
        fallback: Vision | None = None,
        endpoint: str = "http://127.0.0.1:8765",
        camera: str = "wrist",
        verify_question: str = VERIFY_QUESTION,
        accept_above: float = 0.7,
        reject_below: float = 0.0,
        timeout_s: float = 2.0,
        run_log: RunLog | None = None,
    ) -> None:
        if not 0.0 <= reject_below <= accept_above <= 1.0:
            raise ValueError(
                f"need 0 <= reject_below <= accept_above <= 1, got {reject_below}, {accept_above}"
            )
        self.fallback = fallback
        self.endpoint = endpoint.rstrip("/")
        self.camera = camera
        self.verify_question = verify_question
        self.accept_above, self.reject_below = float(accept_above), float(reject_below)
        self.timeout_s = float(timeout_s)
        self.run_log = run_log or RunLog(None)

    # ------------------------------------------------------------ questions

    def plan(self, views: Views, instruction: str) -> TargetPlan:
        return self._need_fallback("plan").plan(views, instruction)

    def locate(self, image: np.ndarray, camera: CameraModel, target: str) -> Detection:
        return self._need_fallback("locate").locate(image, camera, target)

    def verify(self, views: Views, question: str) -> Verdict:
        image = views.images.get(self.camera)
        if image is None:
            return self._escalate(views, question, f"no {self.camera!r} image")
        questions = {"held": {"type": "noul", "instructions": self.verify_question}}
        try:
            answer = self._predict(image, questions, purpose="verify")
        except VisionError as exc:
            return self._escalate(views, question, str(exc))
        p = float(answer["answers"]["held"]["noul"])
        if p >= self.accept_above:
            return Verdict(ok=True, confidence=p, reason=f"P(held)={p:.2f}", model=MODEL_NAME)
        if p < self.reject_below:  # strict: the default 0 never rejects
            return Verdict(ok=False, confidence=1.0 - p, reason=f"P(held)={p:.2f}",
                           model=MODEL_NAME)
        return self._escalate(views, question, f"P(held)={p:.2f} is not a confident yes")

    # ------------------------------------------------------------ internals

    def _escalate(self, views: Views, question: str, why: str) -> Verdict:
        if self.fallback is None:
            raise VisionError(f"laya-vision could not answer verify ({why}) and has no fallback")
        self.run_log.event("escalate", question="verify", reason=why, frm=MODEL_NAME)
        return self.fallback.verify(views, question)

    def _need_fallback(self, question: str) -> Vision:
        if self.fallback is None:
            raise VisionError(
                f"laya-vision cannot answer {question!r} (it returns typed choices, not "
                "descriptions or pixels); configure a fallback vision backend"
            )
        return self.fallback

    def _predict(self, image: np.ndarray, questions: dict[str, Any], purpose: str) -> dict:
        data, _ = encode_image(image)
        record: dict[str, Any] = {
            "purpose": purpose,
            "model": MODEL_NAME,
            "images": [self.run_log.image(f"{purpose}_{self.camera}", image)],
            "prompt": json.dumps(questions),
        }
        started = time.monotonic()
        body = json.dumps({"image": data, "questions": questions}).encode()
        request = urllib.request.Request(
            f"{self.endpoint}/predict", data=body, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                answer = json.loads(response.read())
        except urllib.error.HTTPError as exc:
            self._fail(record, started, f"server error {exc.code}: {exc.read()[:200]!r}")
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            self._fail(record, started, f"laya server unreachable at {self.endpoint}: {exc}")
        record["latency_s"] = round(time.monotonic() - started, 3)
        record["server_latency_s"] = answer.get("server_latency_s")
        record["revision"] = answer.get("provenance", {}).get("checkpoint", {}).get("revision")
        record["answer"] = answer.get("answers")
        self.run_log.call(record)
        print(f"[vision] {purpose} via {MODEL_NAME}: {record['answer']} ({record['latency_s']}s)")
        return answer

    def _fail(self, record: dict[str, Any], started: float, why: str) -> None:
        record["latency_s"] = round(time.monotonic() - started, 3)
        record["error"] = why
        self.run_log.call(record)
        raise VisionError(why)
