"""Regenerate the MJCF built from the vendored YAM model.

    python scripts/build_yam_assets.py

Splits assets/i2rt_yam/upstream/yam.xml into an arm and a stock end effector,
and writes the glasses. Never touches assets/i2rt_yam/upstream/.
"""

from __future__ import annotations

from pathlib import Path

from teleop_sim.envs.yam_assets import build_arm, build_glasses, build_linear_gripper

ASSETS = Path(__file__).resolve().parent.parent / "assets"
UPSTREAM = ASSETS / "i2rt_yam" / "upstream" / "yam.xml"


def outputs() -> dict[Path, str]:
    upstream = UPSTREAM.read_text()
    return {
        ASSETS / "i2rt_yam" / "yam_arm.xml": build_arm(upstream),
        ASSETS / "end_effectors" / "yam_linear" / "yam_linear.xml": build_linear_gripper(upstream),
        ASSETS / "scenes" / "glasses_table" / "glasses.xml": build_glasses(),
    }


def main() -> None:
    for path, text in outputs().items():
        path.parent.mkdir(parents=True, exist_ok=True)
        changed = not path.exists() or path.read_text() != text
        path.write_text(text)
        print(f"{'wrote   ' if changed else 'unchanged'} {path.relative_to(ASSETS.parent)}")


if __name__ == "__main__":
    main()
