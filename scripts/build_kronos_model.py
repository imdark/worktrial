"""Write assets/end_effectors/kronos/kronos.xml from kronos_extract.json.

    python scripts/build_kronos_model.py

Plain Python: no CAD, no gmsh. The closing angle -- where the two pads meet --
is found by sweeping the fingers shut through their own pad collision shapes,
then written in as the joint's hard stop and the actuator range. Rerun after any change to the
extraction or to teleop_sim/envs/kronos_assets.py.
"""

from __future__ import annotations

from pathlib import Path

from teleop_sim.envs.kronos_assets import build_kronos, degrees, load_extract

OUT = Path(__file__).resolve().parent.parent / "assets" / "end_effectors" / "kronos"


def measure_closing_angle(extract: dict, step: float = 0.0005) -> float:
    """Sweep both fingers shut, kinematically, until the pad shapes touch."""
    import mujoco
    import numpy as np

    probe = build_kronos(extract, closed_rad=extract_travel(extract))
    path = OUT / "_probe.xml"
    path.write_text(probe)
    try:
        m = mujoco.MjModel.from_xml_path(str(path))
    finally:
        path.unlink()
    d = mujoco.MjData(m)
    name = lambda g: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""  # noqa: E731
    body = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)  # noqa: E731
    pads = {
        side: [
            g
            for g in range(m.ngeom)
            if "pad_hull" in name(g) and m.geom_bodyid[g] == body(f"kronos_finger_{side}")
        ]
        for side in ("left", "right")
    }
    joints = [
        m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"kronos_finger_{s}")]
        for s in ("left", "right")
    ]
    fromto = np.zeros(6)
    for angle in np.arange(0.0, extract_travel(extract), step):
        d.qpos[joints] = angle
        mujoco.mj_kinematics(m, d)
        gap = min(
            mujoco.mj_geomDistance(m, d, a, b, 1.0, fromto)
            for a in pads["left"]
            for b in pads["right"]
        )
        if gap <= 0.0:
            return float(angle)
    raise RuntimeError("the pads never meet within the search travel")


def extract_travel(extract: dict) -> float:
    from teleop_sim.envs.kronos_assets import KronosActuation

    return KronosActuation().travel_rad


def main() -> None:
    extract = load_extract(OUT / "kronos_extract.json")
    closed = measure_closing_angle(extract)
    (OUT / "kronos.xml").write_text(build_kronos(extract, closed_rad=closed))
    print(f"pads meet at {closed:.4f} rad ({degrees(closed):.1f} deg) -> {OUT.name}/kronos.xml")


if __name__ == "__main__":
    main()
