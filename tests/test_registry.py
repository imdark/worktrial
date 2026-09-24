"""Regression cover for the Stage 0 defect found by running test files alone.

Registration used to be a side effect of whichever module a caller happened to
import first, so `retarget: {type: identity}` resolved in one entry point and
raised in another. The suite hid it: one test file imported the retargeters and
every other test silently benefited.
"""

from __future__ import annotations

import sys

import pytest

from teleop_sim.core.registry import (
    POLICIES,
    RETARGETERS,
    SAFETY,
    SOURCES,
    SUCCESS,
    Registry,
    RegistryError,
)


def test_register_and_lookup():
    registry = Registry("widget")

    @registry.register("a")
    class Widget:
        pass

    assert registry.lookup("a") is Widget
    assert "a" in registry
    assert len(registry) == 1


def test_duplicate_name_rejected():
    registry = Registry("widget")
    registry.register("a")(type("A", (), {}))
    with pytest.raises(RegistryError, match="already registered"):
        registry.register("a")(type("B", (), {}))


def test_registering_the_same_class_twice_is_harmless():
    registry = Registry("widget")
    widget = type("A", (), {})
    registry.register("a")(widget)
    registry.register("a")(widget)
    assert registry.lookup("a") is widget


def test_unknown_name_lists_alternatives():
    registry = Registry("widget")
    registry.register("alpha")(type("A", (), {}))
    with pytest.raises(RegistryError, match=r"available: \['alpha'\]"):
        registry.lookup("beta")


def test_build_requires_a_type_key():
    with pytest.raises(RegistryError, match="'type' key"):
        Registry("widget").build({"colour": "red"})


def test_build_merges_config_and_injected_kwargs():
    registry = Registry("widget")

    @registry.register("w")
    class Widget:
        def __init__(self, size, spec):
            self.size, self.spec = size, spec

    widget = registry.build({"type": "w", "size": 3}, spec="injected")
    assert (widget.size, widget.spec) == (3, "injected")


def test_declared_module_is_imported_on_first_use():
    """The mechanism Stage 1 needs: `robot.type: mujoco` must resolve without
    `import teleop_sim.core.config` dragging mujoco in."""
    assert "tests.lazy_fake" not in sys.modules
    SUCCESS.declare("lazy_fake", "tests.lazy_fake")
    assert "lazy_fake" in SUCCESS.names()

    assert SUCCESS.lookup("lazy_fake").__name__ == "LazySuccessDetector"
    assert "tests.lazy_fake" in sys.modules


def test_declared_module_that_cannot_be_imported():
    registry = Registry("robot")
    registry.declare("ghost", "teleop_sim.module_that_does_not_exist")
    with pytest.raises(RegistryError, match="failed to import"):
        registry.lookup("ghost")


def test_declared_module_that_forgets_to_register():
    registry = Registry("robot")
    registry.declare("silent", "teleop_sim.core.clock")
    with pytest.raises(RegistryError, match="did not register it"):
        registry.lookup("silent")


def test_builtins_populate_the_registries():
    import teleop_sim.builtins  # noqa: F401

    assert {"identity", "ik"} <= set(RETARGETERS.names())
    assert "constant" in POLICIES.names()
    assert {"teleop", "policy", "async_policy"} <= set(SOURCES.names())
    assert {"sim_watchdog", "hardware_safety"} <= set(SAFETY.names())
