"""Vision answered by Claude: a fast model for the frequent questions, a smart
one for the rare hard ones.

Routing, and why:

* ``locate`` and ``verify`` go to the fast model (Claude Sonnet 5) at low
  effort. They are narrow, asked every attempt, and the arm waits on them.
* ``plan`` goes to the smart model (Claude Opus 5.5) at high effort: which
  glass the instruction means is scene reasoning, asked once per episode.
* A fast answer below ``escalate_below`` confidence, or a miss, is asked
  again of the smart model before the task gives up.

Answers are forced into a JSON schema with structured outputs rather than
forced tool use, which Claude Opus 5.5 rejects. Thinking is left adaptive
(Opus 5.5 cannot disable it); depth is controlled with ``effort`` only, set
explicitly because Opus 5.5 defaults to ``medium``.

Every call is logged with its image, latency and token usage (runlog.py).
"""

from __future__ import annotations

import base64
import json
import time
from typing import Any

import numpy as np

from teleop_sim.perception.camera import CameraModel
from teleop_sim.perception.vision import (
    Detection,
    HazardReview,
    Pixel,
    TargetPlan,
    Verdict,
    Views,
    Vision,
    VisionError,
)
from teleop_sim.runlog import RunLog

FAST_MODEL = "claude-sonnet-5"
SMART_MODEL = "claude-opus-5-5"


def _schema(properties: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


LOCATE_SCHEMA = _schema(
    {
        "found": {"type": "boolean"},
        "base_x": {"type": "number"},
        "base_y": {"type": "number"},
        "rim_x": {"type": "number"},
        "rim_y": {"type": "number"},
        "confidence": {"type": "number"},
        "note": {"type": "string"},
    }
)
PLAN_SCHEMA = _schema(
    {
        "feasible": {"type": "boolean"},
        "target_description": {"type": "string"},
        "reason": {"type": "string"},
    }
)
VERIFY_SCHEMA = _schema(
    {
        "ok": {"type": "boolean"},
        "confidence": {"type": "number"},
        "reason": {"type": "string"},
    }
)
HAZARD_SCHEMA = _schema(
    {
        "hazard": {"type": "boolean"},
        "decision": {"type": "string", "enum": ["resume", "wait", "abort"]},
        "confidence": {"type": "number"},
        "reason": {"type": "string"},
    }
)

LOCATE_SYSTEM = (
    "You are the perception module of a robot arm that picks up drinking glasses from a "
    "table. You are given one image from the camera on the robot's wrist; the gripper's "
    "fingers may be visible at the bottom corners. Report pixel coordinates in that image: "
    "x from the left edge, y from the top edge, in pixels of the image exactly as given. "
    "Be precise -- the pixels are converted into the 3D point the gripper moves to."
)
PLAN_SYSTEM = (
    "You direct a robot arm working at a table. You see the scene from one or more cameras "
    "and receive an instruction naming an object to pick up. Decide which object is meant "
    "and describe it so that a second model, shown only a single image from the WRIST "
    "camera, can pick it out without ambiguity: say where it is relative to the other "
    "objects as seen in the wrist image (left/right/nearer/further), and what it looks "
    "like. If the instruction cannot be carried out with what is visible, say so."
)
VERIFY_SYSTEM = (
    "You check the progress of a robot arm doing a manipulation task, from its camera "
    "images. Answer the question strictly from what is visible. Be conservative: report "
    "ok=true only when the images clearly show it."
)


def encode_image(rgb: np.ndarray, quality: int = 90) -> tuple[str, str]:
    """Base64 image data and media type for an RGB uint8 array."""
    try:
        import cv2

        ok, buf = cv2.imencode(
            ".jpg", np.ascontiguousarray(rgb[:, :, ::-1]), [cv2.IMWRITE_JPEG_QUALITY, int(quality)]
        )
        if not ok:
            raise VisionError("JPEG encoding failed")
        return base64.standard_b64encode(buf.tobytes()).decode("ascii"), "image/jpeg"
    except ImportError:
        return base64.standard_b64encode(_png(rgb)).decode("ascii"), "image/png"


def _png(rgb: np.ndarray) -> bytes:
    """Minimal PNG encoder, so the SDK path works without OpenCV."""
    import struct
    import zlib

    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 6))
        + chunk(b"IEND", b"")
    )


HAZARD_SYSTEM = (
    "You are the safety reviewer of a robot arm working at a table. An on-board detector "
    "has stopped the arm because of a possible hazard: either a camera detector thinks "
    "something unexpected is in or near the arm's path, or the arm's own motors felt an "
    "unexpected force, as if it hit or pressed on something. The context says which. The "
    "arm is holding still. Look at the images and decide:\n"
    "- resume: nothing is in the arm's path or within reach of it except what the task "
    "expects (the object it is handling, other objects already on the table, the robot's "
    "own gripper, cables and mounts), and the arm has not hit anything it should not: the "
    "alarm was false, or the hazard has gone.\n"
    "- wait: a person, a hand, or an object that should not be there is in or near the "
    "arm's path, but it is likely to move away. The arm keeps holding and you will be shown "
    "fresh images again.\n"
    "- abort: the hazard will not clear by itself, or the scene has changed so the task "
    "cannot safely continue (the arm is touching or has knocked over something, something "
    "placed in the path, the object knocked over, something wrong with the arm).\n"
    "Be conservative: a person's body part anywhere near the arm is a hazard. Choose "
    "resume only when the images clearly show the way is clear; confidence is how sure "
    "you are of your decision, 0 to 1."
)


class ClaudeVision(Vision):
    def __init__(
        self,
        fast_model: str = FAST_MODEL,
        smart_model: str = SMART_MODEL,
        fast_effort: str = "low",
        smart_effort: str = "high",
        escalate_below: float = 0.5,
        fast_timeout_s: float = 60.0,
        smart_timeout_s: float = 180.0,
        run_log: RunLog | None = None,
        client: Any = None,
    ) -> None:
        if client is None:
            import os

            import anthropic

            # Credentials resolve from the environment: ANTHROPIC_API_KEY, or
            # an `ant auth login` profile. Never from a file in this repo. A key
            # not scoped to a workspace also needs ANTHROPIC_WORKSPACE_ID.
            workspace = os.environ.get("ANTHROPIC_WORKSPACE_ID")
            headers = {"anthropic-workspace-id": workspace} if workspace else None
            client = anthropic.Anthropic(default_headers=headers)
        self.client = client
        self.fast_model, self.smart_model = fast_model, smart_model
        self.fast_effort, self.smart_effort = fast_effort, smart_effort
        self.escalate_below = float(escalate_below)
        self.timeouts = {fast_model: float(fast_timeout_s), smart_model: float(smart_timeout_s)}
        self.run_log = run_log or RunLog(None)

    # ------------------------------------------------------------ questions

    def plan(self, views: Views, instruction: str) -> TargetPlan:
        content = self._images(views.images, "plan")
        content.append({"type": "text", "text": f"Instruction: {instruction}"})
        data = self._call(
            self.smart_model, self.smart_effort, PLAN_SYSTEM, content, PLAN_SCHEMA, purpose="plan"
        )
        return TargetPlan(
            description=data["target_description"],
            feasible=bool(data["feasible"]),
            reason=data["reason"],
        )

    def locate(self, image: np.ndarray, camera: CameraModel, target: str) -> Detection:
        det = self._locate(self.fast_model, self.fast_effort, image, target)
        if not det.found or det.confidence < self.escalate_below:
            self.run_log.event(
                "escalate",
                question="locate",
                reason=det.note or "low confidence",
                confidence=det.confidence,
                to=self.smart_model,
            )
            smart = self._locate(self.smart_model, self.smart_effort, image, target)
            if smart.found or not det.found:
                return smart
        return det

    def verify(self, views: Views, question: str) -> Verdict:
        verdict = self._verify(self.fast_model, self.fast_effort, views, question)
        if verdict.confidence < self.escalate_below:
            self.run_log.event(
                "escalate", question="verify", confidence=verdict.confidence, to=self.smart_model
            )
            verdict = self._verify(self.smart_model, self.smart_effort, views, question)
        return verdict

    def review_hazard(self, images: dict[str, np.ndarray], context: str) -> HazardReview:
        """Decide what a paused arm should do. The fast model answers; a "resume" or an
        "abort" it is not sure of (below escalate_below) goes to the smart model, whose
        word stands. (An unsure abort ends a run for nothing: 2026-09-25, Sonnet at
        0.60 missed the hand that had pushed the arm.)"""
        review = self._review_hazard(self.fast_model, self.fast_effort, images, context)
        unsure = review.confidence < self.escalate_below
        if unsure and review.decision in ("resume", "abort"):
            self.run_log.event(
                "escalate", question="hazard", confidence=review.confidence, to=self.smart_model
            )
            review = self._review_hazard(self.smart_model, self.smart_effort, images, context)
        return review

    # ------------------------------------------------------------ internals

    def _review_hazard(
        self, model: str, effort: str, images: dict[str, np.ndarray], context: str
    ) -> HazardReview:
        content = self._images(images, "hazard")
        content.append({"type": "text", "text": context})
        data = self._call(model, effort, HAZARD_SYSTEM, content, HAZARD_SCHEMA, purpose="hazard")
        return HazardReview(
            decision=str(data["decision"]),
            hazard=bool(data["hazard"]),
            confidence=float(data["confidence"]),
            reason=data["reason"],
            model=model,
        )

    def _locate(self, model: str, effort: str, image: np.ndarray, target: str) -> Detection:
        h, w = image.shape[:2]
        content = self._images({"wrist": image}, "locate")
        content.append(
            {
                "type": "text",
                "text": (
                    f"Target: {target}\n\n"
                    f"The image is {w}x{h} pixels. If the target is visible, report:\n"
                    "- base_x, base_y: the centre of the glass's circular bottom, where it rests "
                    "on the table. For a transparent glass, the centre of the ellipse traced by "
                    "its bottom edge.\n"
                    "- rim_x, rim_y: the centre of the glass's circular top opening.\n"
                    "- confidence: 0 to 1, how sure you are that these points are right.\n"
                    "If the target is not visible, or you cannot tell which object is meant, set "
                    "found=false and report 0 for the coordinates. Use note for anything unusual."
                ),
            }
        )
        data = self._call(model, effort, LOCATE_SYSTEM, content, LOCATE_SCHEMA, purpose="locate")
        found = bool(data["found"])
        base = Pixel(float(data["base_x"]), float(data["base_y"]))
        rim = Pixel(float(data["rim_x"]), float(data["rim_y"]))
        if found and not (0 <= base.u < w and 0 <= base.v < h):
            found, data["note"] = False, f"base pixel {base} outside the image. {data['note']}"
        rim_ok = found and 0 <= rim.u < w and 0 <= rim.v < h and (rim.u, rim.v) != (0.0, 0.0)
        return Detection(
            found=found,
            base=base if found else None,
            rim=rim if rim_ok else None,
            confidence=float(data["confidence"]),
            note=data["note"],
            model=model,
        )

    def _verify(self, model: str, effort: str, views: Views, question: str) -> Verdict:
        content = self._images(views.images, "verify")
        content.append({"type": "text", "text": f"Question: {question}"})
        data = self._call(model, effort, VERIFY_SYSTEM, content, VERIFY_SCHEMA, purpose="verify")
        return Verdict(
            ok=bool(data["ok"]),
            confidence=float(data["confidence"]),
            reason=data["reason"],
            model=model,
        )

    def _images(self, images: dict[str, np.ndarray], purpose: str) -> list[dict[str, Any]]:
        content: list[dict[str, Any]] = []
        self._last_images = []
        for name, rgb in images.items():
            data, media_type = encode_image(rgb)
            self._last_images.append(self.run_log.image(f"{purpose}_{name}", rgb))
            h, w = rgb.shape[:2]
            content.append({"type": "text", "text": f"Camera '{name}' ({w}x{h}):"})
            content.append(
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": media_type, "data": data},
                }
            )
        return content

    def _call(
        self,
        model: str,
        effort: str,
        system: str,
        content: list[dict[str, Any]],
        schema: dict[str, Any],
        purpose: str,
    ) -> dict[str, Any]:
        import anthropic

        started = time.monotonic()
        record: dict[str, Any] = {
            "purpose": purpose,
            "model": model,
            "effort": effort,
            "images": list(getattr(self, "_last_images", [])),
            "prompt": next((c["text"] for c in reversed(content) if c["type"] == "text"), ""),
        }
        try:
            response = self.client.with_options(
                timeout=self.timeouts.get(model, 120.0)
            ).messages.create(
                model=model,
                max_tokens=16000,
                system=system,
                messages=[{"role": "user", "content": content}],
                output_config={
                    "effort": effort,
                    "format": {"type": "json_schema", "schema": schema},
                },
            )
        except anthropic.RateLimitError as exc:
            self._fail(record, started, f"rate limited: {exc}")
        except anthropic.APIStatusError as exc:
            self._fail(record, started, f"API error {exc.status_code}: {exc.message}")
        except anthropic.APIConnectionError as exc:
            self._fail(record, started, f"connection error: {exc}")

        record["latency_s"] = round(time.monotonic() - started, 3)
        record["request_id"] = getattr(response, "_request_id", None)
        record["stop_reason"] = response.stop_reason
        record["usage"] = {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        }
        if response.stop_reason == "refusal":
            self._fail(record, started, f"refused: {getattr(response, 'stop_details', None)}")
        if response.stop_reason == "max_tokens":
            self._fail(record, started, "ran out of tokens before answering")
        text = next((b.text for b in response.content if b.type == "text"), None)
        if text is None:
            self._fail(record, started, "no text block in the response")
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            self._fail(record, started, f"unparseable answer: {exc}: {text[:200]}")
        record["answer"] = data
        self.run_log.call(record)
        print(f"[vision] {purpose} via {model}: {data} ({record['latency_s']}s)")
        return data

    def _fail(self, record: dict[str, Any], started: float, why: str) -> None:
        record["latency_s"] = round(time.monotonic() - started, 3)
        record["error"] = why
        self.run_log.call(record)
        raise VisionError(f"{record['purpose']} via {record['model']}: {why}")
