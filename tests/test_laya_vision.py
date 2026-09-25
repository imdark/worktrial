"""laya-vision as the verify backend, tested against a stand-in server.

No model is loaded: a stdlib HTTP server answers /predict with a canned
P(held), so these run anywhere. What they pin down is the routing -- which
answers are trusted, which go to the fallback, and what gets logged.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import pytest

from teleop_sim.perception.laya_vision import LayaVision
from teleop_sim.perception.vision import Verdict, Views, Vision, VisionError
from teleop_sim.runlog import RunLog

WRIST = np.zeros((48, 64, 3), dtype=np.uint8)


class FakeLayaServer:
    """Answers every /predict with ``p`` as the noul probability."""

    def __init__(self, p: float = 0.5, status: int = 200) -> None:
        self.p, self.status, self.requests = p, status, []
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802
                fake.requests.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
                if fake.status != 200:
                    body = json.dumps({"error": "boom"}).encode()
                else:
                    body = json.dumps({
                        "answers": {"held": {"type": "noul", "noul": fake.p}},
                        "provenance": {"checkpoint": {"revision": "abc123"}},
                        "server_latency_s": 0.15,
                    }).encode()
                self.send_response(fake.status)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args) -> None:
                pass

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.endpoint = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


class RecordingFallback(Vision):
    def __init__(self) -> None:
        self.asked: list[str] = []

    def plan(self, views, instruction):
        self.asked.append("plan")

    def locate(self, image, camera, target):
        self.asked.append("locate")

    def verify(self, views, question):
        self.asked.append("verify")
        return Verdict(ok=True, confidence=0.95, reason="fallback", model="fallback")


@pytest.fixture
def server():
    fake = FakeLayaServer()
    yield fake
    fake.close()


def _views() -> Views:
    return Views(images={"wrist": WRIST, "overview": WRIST})


def test_a_confident_yes_is_used_without_the_fallback(server):
    server.p = 0.9
    fallback = RecordingFallback()
    verdict = LayaVision(fallback, endpoint=server.endpoint).verify(_views(), "held?")
    assert (verdict.ok, verdict.model) == (True, "laya-vision")
    assert verdict.confidence == pytest.approx(0.9)
    assert fallback.asked == []


@pytest.mark.parametrize("p", [0.53, 0.17, 0.0])
def test_by_default_every_answer_short_of_a_confident_yes_goes_to_the_fallback(server, p):
    # 0.17 is what it said, zero-shot, about a glass held in sim.
    server.p = p
    fallback = RecordingFallback()
    verdict = LayaVision(fallback, endpoint=server.endpoint).verify(_views(), "held?")
    assert verdict.model == "fallback" and fallback.asked == ["verify"]


def test_a_confident_no_is_used_once_rejection_is_enabled(server):
    server.p = 0.1
    fallback = RecordingFallback()
    vision = LayaVision(fallback, endpoint=server.endpoint, reject_below=0.3)
    verdict = vision.verify(_views(), "held?")
    assert (verdict.ok, verdict.model) == (False, "laya-vision")
    assert verdict.confidence == pytest.approx(0.9)
    assert fallback.asked == []


def test_only_the_wrist_image_and_the_fixed_question_are_sent(server):
    server.p = 0.9
    LayaVision(endpoint=server.endpoint, verify_question="Held?").verify(_views(), "long prose")
    (request,) = server.requests
    assert set(request) == {"image", "questions"}
    assert request["questions"] == {"held": {"type": "noul", "instructions": "Held?"}}


def test_a_server_that_is_down_falls_back_rather_than_failing():
    fake = FakeLayaServer()
    endpoint = fake.endpoint
    fake.close()
    fallback = RecordingFallback()
    verdict = LayaVision(fallback, endpoint=endpoint, timeout_s=0.5).verify(_views(), "held?")
    assert verdict.model == "fallback"


def test_a_server_error_without_a_fallback_is_a_vision_error():
    fake = FakeLayaServer(status=400)
    try:
        with pytest.raises(VisionError, match="server error 400"):
            LayaVision(endpoint=fake.endpoint).verify(_views(), "held?")
    finally:
        fake.close()


def test_plan_and_locate_need_a_fallback():
    fallback = RecordingFallback()
    vision = LayaVision(fallback)
    vision.plan(_views(), "pick the cup")
    vision.locate(WRIST, None, "the cup")
    assert fallback.asked == ["plan", "locate"]
    with pytest.raises(VisionError, match="configure a fallback"):
        LayaVision().locate(WRIST, None, "the cup")


def test_every_call_is_logged_with_its_image_for_later_training(server, tmp_path):
    server.p = 0.9
    log = RunLog(tmp_path)
    LayaVision(endpoint=server.endpoint, run_log=log).verify(_views(), "held?")
    (record,) = [json.loads(line) for line in (tmp_path / "calls.jsonl").read_text().splitlines()]
    assert record["purpose"] == "verify" and record["model"] == "laya-vision"
    assert record["revision"] == "abc123" and record["answer"]["held"]["noul"] == 0.9
    assert record["images"] and (tmp_path / record["images"][0]).exists()


def test_the_band_must_be_ordered():
    with pytest.raises(ValueError, match="reject_below <= accept_above"):
        LayaVision(accept_above=0.3, reject_below=0.7)


def test_the_pick_task_builds_laya_with_a_nested_fallback():
    from teleop_sim.perception.claude_vision import ClaudeVision
    from teleop_sim.tasks.pick_lift_place import PickLiftPlacePolicy

    policy = PickLiftPlacePolicy.__new__(PickLiftPlacePolicy)
    policy.vision_config = {"type": "laya", "accept_above": 0.8,
                            "fallback": {"type": "claude", "client": object()}}
    policy.log = RunLog(None)
    vision = policy._build_vision()
    assert isinstance(vision, LayaVision) and vision.accept_above == 0.8
    assert isinstance(vision.fallback, ClaudeVision)
    policy.vision_config = {"type": "laya", "fallback": {"type": "laya"}}
    with pytest.raises(ValueError, match="cannot fall back to itself"):
        policy._build_vision()
