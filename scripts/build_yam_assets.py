"""Regenerate the MJCF built from the vendored YAM model.

    python scripts/build_yam_assets.py

Splits assets/i2rt_yam/upstream/yam.xml into an arm and a stock end effector,
and writes the glasses and gem13's tapered cup scene. Never touches assets/i2rt_yam/upstream/.
"""

from __future__ import annotations

from pathlib import Path

from teleop_sim.envs.yam_assets import (
    build_arm,
    build_cup,
    build_cup_scene,
    build_glasses,
    build_linear_gripper,
    split_fused_mesh,
)

ASSETS = Path(__file__).resolve().parent.parent / "assets"
UPSTREAM = ASSETS / "i2rt_yam" / "upstream" / "yam.xml"


def outputs() -> dict[Path, str | bytes]:
    upstream = UPSTREAM.read_text()
    fused = (UPSTREAM.parent / "assets" / "model2__13.stl").read_bytes()
    motor, frame = split_fused_mesh(upstream, fused)
    return {
        ASSETS / "i2rt_yam" / "yam_arm.xml": build_arm(upstream),
        ASSETS / "i2rt_yam" / "meshes" / "yam_wrist_motor.stl": motor,
        ASSETS / "end_effectors" / "yam_linear" / "yam_linear.xml": build_linear_gripper(upstream),
        ASSETS / "end_effectors" / "yam_linear" / "meshes" / "stock_frame.stl": frame,
        ASSETS / "scenes" / "glasses_table" / "glasses.xml": build_glasses(),
        ASSETS / "scenes" / "cup_table" / "cup.xml": build_cup(),
        ASSETS / "scenes" / "cup_table" / "scene.xml": build_cup_scene(
            (ASSETS / "scenes" / "glasses_table" / "scene.xml").read_text()
        ),
    }


def main() -> None:
    for path, text in outputs().items():
        path.parent.mkdir(parents=True, exist_ok=True)
        data = text.encode() if isinstance(text, str) else text
        changed = not path.exists() or path.read_bytes() != data
        path.write_bytes(data)
        print(f"{'wrote   ' if changed else 'unchanged'} {path.relative_to(ASSETS.parent)}")


if __name__ == "__main__":
    main()
