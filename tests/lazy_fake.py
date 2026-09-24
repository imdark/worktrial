"""Exists only so a test can prove lazy registry resolution actually imports."""

from __future__ import annotations

from teleop_sim.core.registry import SUCCESS, register
from tests.fakes import FakeSuccessDetector


@register(SUCCESS, "lazy_fake")
class LazySuccessDetector(FakeSuccessDetector):
    pass
