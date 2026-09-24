"""Write a composed robot (and its scene) out as a single MJCF.

    python scripts/export_model.py --out build/yam_glasses.xml
    python -m mujoco.viewer --mjcf=build/yam_glasses.xml

A composed robot has no single MJCF on disk -- arm, end effector and scene are
assembled at load time. This writes one, for MuJoCo's viewer or anything else
that wants a file. Asset paths in it are absolute: a build artefact for this
machine, not something to commit (build/ is gitignored).
"""

from __future__ import annotations

import argparse

from teleop_sim.core.parts import SceneSpec
from teleop_sim.core.spec import RobotSpec
from teleop_sim.robots.sim.assembly import export_xml

DEFAULT_SPEC = "teleop_sim/robots/specs/yam.yaml"
DEFAULT_SCENE = "teleop_sim/scenes/glasses_table.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", default=DEFAULT_SPEC)
    parser.add_argument("--scene", default=DEFAULT_SCENE,
                        help="scene description, or 'none' for the bare robot")
    parser.add_argument("--out", default="build/model.xml")
    args = parser.parse_args()

    spec = RobotSpec.from_yaml(args.spec)
    if spec.assembly is not None:
        spec = spec.with_scene(None if args.scene == "none" else SceneSpec.from_yaml(args.scene))
    path = export_xml(spec, args.out)
    print(f"{spec.name} -> {path}")


if __name__ == "__main__":
    main()
