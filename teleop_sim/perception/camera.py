"""Pinhole geometry: pixels to rays to points on a plane, and back.

Frames follow MuJoCo's camera convention -- the camera looks along its -z
axis with +y up -- because camera poses come from the kinematic model. Pixel
(u, v) has its origin at the top-left, v increasing downward, as in the
images a renderer or a RealSense returns.

Why a plane rather than the depth image: a water glass is transparent, and a
stereo depth camera reports holes or the table behind it. The table itself
reads fine, and a glass stands on it, so the ray through the pixel where the
glass meets the table, intersected with the table plane, locates the glass
without trusting depth on the glass at all.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class CameraModel:
    K: np.ndarray  # 3x3 intrinsics for an image of (width, height)
    position: np.ndarray  # camera origin, world frame
    rotation: np.ndarray  # world_from_camera; columns are the camera axes
    width: int
    height: int

    @classmethod
    def from_intrinsics_dict(
        cls, intr: dict[str, Any], position, rotation, width: int, height: int
    ) -> CameraModel:
        """From a RealSense-style dict (fx, fy, cx, cy, width, height), rescaled
        to the size of the image actually being used."""
        sx = width / float(intr.get("width", width))
        sy = height / float(intr.get("height", height))
        K = np.array(
            [
                [intr["fx"] * sx, 0.0, intr["cx"] * sx],
                [0.0, intr["fy"] * sy, intr["cy"] * sy],
                [0.0, 0.0, 1.0],
            ]
        )
        return cls(
            K=K,
            position=np.asarray(position, float),
            rotation=np.asarray(rotation, float),
            width=width,
            height=height,
        )

    def project(self, point_world: np.ndarray) -> tuple[float, float] | None:
        """Pixel of a world point, or None if it is behind the camera."""
        p = self.rotation.T @ (np.asarray(point_world, float) - self.position)
        depth = -p[2]
        if depth <= 1e-6:
            return None
        u = self.K[0, 0] * p[0] / depth + self.K[0, 2]
        v = -self.K[1, 1] * p[1] / depth + self.K[1, 2]
        return float(u), float(v)

    def ray(self, u: float, v: float) -> tuple[np.ndarray, np.ndarray]:
        """World-frame origin and unit direction of the ray through pixel (u, v)."""
        x = (u - self.K[0, 2]) / self.K[0, 0]
        y = -(v - self.K[1, 2]) / self.K[1, 1]
        direction = self.rotation @ np.array([x, y, -1.0])
        return self.position.copy(), direction / np.linalg.norm(direction)

    def intersect_z(self, u: float, v: float, z: float) -> np.ndarray | None:
        """Where the ray through (u, v) meets the horizontal plane at height z."""
        origin, direction = self.ray(u, v)
        if abs(direction[2]) < 1e-6:
            return None
        t = (z - origin[2]) / direction[2]
        if t <= 0:
            return None
        return origin + t * direction

    def in_image(self, u: float, v: float, margin: float = 0.0) -> bool:
        return margin <= u < self.width - margin and margin <= v < self.height - margin
