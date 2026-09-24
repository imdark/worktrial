"""MuJoCo offscreen cameras.

One renderer per distinct resolution, shared by every camera that wants it --
a mujoco.Renderer owns a GL context and framebuffer, and one per camera is
wasteful for no benefit.
"""

from __future__ import annotations

import numpy as np

from teleop_sim.core.protocols import Camera
from teleop_sim.core.spec import CameraSpec


class RendererPool:
    """Lazily-created renderers, keyed by (height, width)."""

    def __init__(self, model) -> None:
        self._model = model
        self._renderers: dict[tuple[int, int], object] = {}

    def get(self, height: int, width: int):
        import mujoco

        key = (int(height), int(width))
        if key not in self._renderers:
            self._renderers[key] = mujoco.Renderer(self._model, height=key[0], width=key[1])
        return self._renderers[key]

    def close(self) -> None:
        for renderer in self._renderers.values():
            renderer.close()
        self._renderers.clear()


class MujocoCamera(Camera):
    def __init__(
        self,
        spec: CameraSpec,
        data,
        pool: RendererPool,
        size: tuple[int, int] | None = None,
    ) -> None:
        self.spec = spec
        self._data = data
        self._pool = pool
        # An explicit override reports its own resolution rather than the
        # spec's, so the startup compatibility check validates what is actually
        # served instead of what the spec wishes were served.
        self._size = (int(size[0]), int(size[1])) if size else spec.resolution

    @property
    def resolution(self) -> tuple[int, int]:
        return self._size

    def read(self) -> np.ndarray:
        width, height = self._size
        renderer = self._pool.get(height, width)
        renderer.update_scene(self._data, camera=self.spec.name)
        return renderer.render()
