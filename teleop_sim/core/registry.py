"""Name -> class lookup, so swapping an implementation is a config edit.

Two ways in. Implementations that need only the base dependencies are
*registered* when ``teleop_sim.builtins`` imports them. Backends that pull a
simulator, a learning framework or a hardware SDK are *declared* against a
module path and imported only when a config actually names one -- which is how
`pip install -e ".[dev]"` stays sufficient for the core while `robot.type:
mujoco` still works.

Deliberately not an entry-point or plugin system: a single repo does not need
one, and a dict is trivial to debug.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Any, TypeVar

T = TypeVar("T")


class RegistryError(KeyError):
    """An implementation name is unknown, or its module failed to import."""


class Registry:
    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._items: dict[str, type] = {}
        self._lazy: dict[str, str] = {}

    def register(self, name: str) -> Callable[[type[T]], type[T]]:
        def decorator(cls: type[T]) -> type[T]:
            existing = self._items.get(name)
            if existing is not None and existing is not cls:
                raise RegistryError(
                    f"{self.kind} {name!r} already registered to {existing.__name__}"
                )
            self._items[name] = cls
            return cls

        return decorator

    def declare(self, name: str, module: str) -> None:
        """Bind a name to a module to import on first use."""
        self._lazy[name] = module

    def names(self) -> list[str]:
        return sorted(set(self._items) | set(self._lazy))

    def lookup(self, name: str) -> type:
        if name in self._items:
            return self._items[name]

        module = self._lazy.get(name)
        if module is None:
            raise RegistryError(
                f"unknown {self.kind} type {name!r}; "
                f"available: {self.names() or '(none registered)'}"
            )

        try:
            importlib.import_module(module)
        except ImportError as exc:
            raise RegistryError(
                f"{self.kind} {name!r} needs module {module!r}, which failed to import "
                f"({exc}). Its optional extra is probably not installed."
            ) from exc

        if name not in self._items:
            raise RegistryError(
                f"{self.kind} {name!r} is declared against {module!r}, but importing "
                "that module did not register it"
            )
        return self._items[name]

    def build(self, config: dict[str, Any], **extra: Any) -> Any:
        """Instantiate from a ``{'type': ..., **kwargs}`` mapping."""
        if "type" not in config:
            raise RegistryError(
                f"{self.kind} config must contain a 'type' key; got keys {sorted(config)}"
            )
        config = dict(config)
        cls = self.lookup(config.pop("type"))
        return cls(**config, **extra)

    def __contains__(self, name: str) -> bool:
        return name in self._items or name in self._lazy

    def __len__(self) -> int:
        return len(self.names())


ROBOTS = Registry("robot")
TELEOPS = Registry("teleoperator")
RETARGETERS = Registry("retargeter")
CAMERAS = Registry("camera")
POLICIES = Registry("policy")
TASKS = Registry("task")
SOURCES = Registry("action source")
SUCCESS = Registry("success detector")
RESETS = Registry("reset strategy")
SAFETY = Registry("safety monitor")


# Free functions so call sites read as `@register(ROBOTS, "fake")`.
def register(registry: Registry, name: str) -> Callable[[type[T]], type[T]]:
    return registry.register(name)


def lookup(registry: Registry, name: str) -> type:
    return registry.lookup(name)


def build(registry: Registry, config: dict[str, Any], **extra: Any) -> Any:
    return registry.build(config, **extra)
