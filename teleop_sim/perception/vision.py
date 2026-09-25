"""What the task asks of a vision system, independent of who answers.

Three questions, each answerable by a VLM, by privileged simulator state, or
later by a trained detector:

* ``plan``   -- given the scene and an instruction, which object is meant?
* ``locate`` -- where, in pixels, does that object stand on the table?
* ``verify`` -- did the step just taken work?

Answers are in pixels, never metres. Turning a pixel into a 3D point is
geometry (perception/camera.py), and keeping it out of the model is what lets
the same answer be checked against ground truth in sim and trusted on
hardware.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from teleop_sim.perception.camera import CameraModel


class VisionError(RuntimeError):
    """The vision backend could not produce an answer (API failure, refusal, ...)."""


@dataclass(frozen=True)
class Pixel:
    u: float
    v: float


@dataclass(frozen=True)
class TargetPlan:
    description: str  # an unambiguous visual description for ``locate``
    feasible: bool = True
    reason: str = ""


@dataclass(frozen=True)
class Detection:
    found: bool
    base: Pixel | None = None  # centre of the glass's footprint on the table
    rim: Pixel | None = None  # centre of the top rim
    confidence: float = 0.0
    note: str = ""
    model: str = ""


@dataclass(frozen=True)
class Verdict:
    ok: bool
    confidence: float
    reason: str = ""
    model: str = ""


@dataclass
class Views:
    """The images a question is asked about, keyed by camera name."""

    images: dict[str, np.ndarray]
    cameras: dict[str, CameraModel] = field(default_factory=dict)


class Vision(ABC):
    @abstractmethod
    def plan(self, views: Views, instruction: str) -> TargetPlan: ...

    @abstractmethod
    def locate(self, image: np.ndarray, camera: CameraModel, target: str) -> Detection: ...

    @abstractmethod
    def verify(self, views: Views, question: str) -> Verdict: ...
