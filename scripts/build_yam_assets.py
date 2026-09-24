"""Regenerate the MJCF layered on top of the vendored YAM model.

    python scripts/build_yam_assets.py

Writes assets/i2rt_yam/yam_follower.xml and assets/i2rt_yam/glasses.xml.
Never touches assets/i2rt_yam/upstream/.
"""

from __future__ import annotations

from pathlib import Path

from teleop_sim.envs.yam_assets import build_follower, build_glasses

ROOT = Path(__file__).resolve().parent.parent / "assets" / "i2rt_yam"


def main() -> None:
    upstream = (ROOT / "upstream" / "yam.xml").read_text()
    outputs = {
        ROOT / "yam_follower.xml": build_follower(upstream),
        ROOT / "glasses.xml": build_glasses(),
    }
    for path, text in outputs.items():
        changed = not path.exists() or path.read_text() != text
        path.write_text(text)
        print(f"{'wrote   ' if changed else 'unchanged'} {path.relative_to(ROOT.parent.parent)}")


if __name__ == "__main__":
    main()
