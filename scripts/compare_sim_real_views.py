"""Does the sim look like the rig? Render the sim at the joints of real wrist images.

    python scripts/compare_sim_real_views.py            # writes outputs/sim_real_views/

For each real wrist image whose joint angles are known, the sim wrist camera is
rendered at exactly those joints, with the cup where the rig found it, in three
set-ups:

    nominal     yam_kronos in cup_table (checker table, open sky)
    calibrated  yam_kronos_gem13 (2 deg tool pitch, D405 fov) in cup_table
    gem13 cell  yam_kronos_gem13 in gem13_cell (wood table at the measured
                depth, curtains, warm light, black cup)

Per pair:

    gripper IoU   overlap of the gripper's outline (dark body + white pads, lower
                  half of the image): camera pose on the hand and finger opening
    edge row      rows between sim and real table far edge (where visible):
                  camera pitch and table depth
    colour dE     mean CIELAB difference of 40x30 downsampled images: appearance
    table rgb     rendered table colour in a fixed patch, vs the real one

Texture-driven edge scores were tried first and dropped: wood grain and a
checkerboard produce more edges than the geometry does.

and a side-by-side plus a 50/50 blend of the gem13 cell over the real image.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import cv2
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from teleop_sim.core.clock import WallClock  # noqa: E402
from teleop_sim.core.parts import SceneSpec  # noqa: E402
from teleop_sim.core.spec import RobotSpec  # noqa: E402
from teleop_sim.robots.kinematics import Kinematics  # noqa: E402
from teleop_sim.robots.sim.mujoco_robot import MujocoRobot  # noqa: E402
from teleop_sim.tasks.pick_lift_place import PickLiftPlacePolicy  # noqa: E402

SPECS = ROOT / "teleop_sim/robots/specs"
SCENES = ROOT / "teleop_sim/scenes"
RUNS = ROOT / "runs"
OUT = ROOT / "outputs/sim_real_views"

SETUPS = {
    "nominal": ("yam_kronos", "cup_table"),
    "calibrated": ("yam_kronos_gem13", "cup_table"),
    "gem13 cell": ("yam_kronos_gem13", "gem13_cell"),
}
REST = [0.0044, 0.0071, 0.0067, 0.0021, -0.0002, -0.0013]  # probe at 17:38, measured
RUN3_CUP = (0.5586, -0.0389)  # run 3's closer-view estimate; the cup sat there 17:40-17:50


def planned_poses() -> dict[str, np.ndarray]:
    """Joints of the images the task took, recomputed exactly as the rig planned
    them (nominal model, same seeds). The rig tracked these to within ~1 deg."""
    nominal = RobotSpec.from_yaml(SPECS / "yam_kronos.yaml")
    policy = PickLiftPlacePolicy(nominal, WallClock(), vision={"type": "oracle"}, glass_height=0.11)
    policy._q_cmd = np.array(REST)
    survey = policy._view_pose(np.array([0.55, -0.03]), policy.survey)
    policy._q_cmd = survey
    refine = policy._view_pose(np.array([0.5632, -0.0447]), policy.refine)
    kin = Kinematics(nominal)
    home_rot = kin.frame(nominal.home_joints(), "site", nominal.ee_site)[1]
    level = kin.ik_multi_seed(
        "site",
        nominal.ee_site,
        np.array([0.40, 0.0, 0.25]),
        home_rot,
        [nominal.home_joints()],
        pos_tol=2e-3,
        rot_tol=math.radians(0.5),
    ).q
    return {"survey": survey, "refine": refine, "level": level}


def pairs() -> list[dict]:
    q = planned_poses()
    return [
        dict(
            name="rest",
            image=RUNS / "20260924-173808_probe/0001_probe_wrist.png",
            joints=REST,
            cup=(0.552, -0.03),
            note="probe at rest, joints measured",
        ),
        dict(
            name="level",
            image=RUNS / "level_check_wrist.png",
            joints=q["level"],
            cup=RUN3_CUP,
            note="level check, joints within 0.1 deg of these",
        ),
        dict(
            name="survey",
            image=RUNS / "20260924-174953_real/0003_locate_wrist.png",
            joints=q["survey"],
            cup=RUN3_CUP,
            note="run 3 survey view, as planned",
        ),
        dict(
            name="refine",
            image=RUNS / "20260924-174953_real/0005_locate_wrist.png",
            joints=q["refine"],
            cup=RUN3_CUP,
            note="run 3 closer view, as planned",
        ),
    ]


def render(spec_name: str, scene: str, joints, cup_xy) -> np.ndarray:
    spec = RobotSpec.from_yaml(SPECS / f"{spec_name}.yaml").with_scene(
        SceneSpec.from_yaml(SCENES / f"{scene}.yaml")
    )
    robot = MujocoRobot(spec, WallClock(), render_cameras=["wrist"])
    m, d = robot.model, robot.data
    d.qpos[robot._qpos_adr] = joints
    d.qpos[robot._gripper_qadr] = spec.gripper.open_pos
    robot._sync_coupled_joints()
    cup = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "cup")
    adr = m.jnt_qposadr[m.body_jntadr[cup]]
    d.qpos[adr : adr + 2] = cup_xy
    mujoco.mj_forward(m, d)
    rgb = robot.get_observation().images["wrist"]
    robot.disconnect()
    return cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR)


def gripper_mask(bgr: np.ndarray) -> np.ndarray:
    """Kronos in the lower half of a wrist image: black foam body or white pads."""
    hsv = cv2.cvtColor(cv2.GaussianBlur(bgr, (5, 5), 0), cv2.COLOR_BGR2HSV)
    sat, val = hsv[..., 1].astype(int), hsv[..., 2].astype(int)
    mask = ((val < 75) | ((sat < 40) & (val > 140))).astype(np.uint8)
    mask[: bgr.shape[0] // 2] = 0
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8)) > 0


def gripper_iou(sim: np.ndarray, real: np.ndarray) -> float:
    a, b = gripper_mask(sim), gripper_mask(real)
    return float((a & b).sum() / max((a | b).sum(), 1))


def far_edge_row(bgr: np.ndarray) -> int | None:
    """Row where the dark backdrop meets the table, in the central columns."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(float)
    profile = np.convolve(gray[:, 150:490].mean(1), np.ones(9) / 9, mode="same")
    top = profile[5:40].mean()
    if top > 60:  # no backdrop in view
        return None
    jump = np.diff(profile[20:320])
    row = int(np.argmax(jump)) + 20
    return row if jump[row - 20] > 1.5 else None


def table_rgb(bgr: np.ndarray) -> list[float]:
    patch = bgr[140:240, 380:600].reshape(-1, 3)[:, ::-1] / 255
    return [round(float(v), 3) for v in patch.mean(0)]


def colour_de(sim: np.ndarray, real: np.ndarray) -> float:
    def lab(b):
        small = cv2.resize(b, (40, 30), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
        return cv2.cvtColor(small, cv2.COLOR_BGR2LAB)

    return float(np.linalg.norm(lab(sim) - lab(real), axis=2).mean())


def label(img: np.ndarray, text: str) -> np.ndarray:
    img = img.copy()
    cv2.rectangle(img, (0, 0), (img.shape[1], 26), (20, 20, 20), -1)
    cv2.putText(img, text, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (235, 235, 235), 1, cv2.LINE_AA)
    return img


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    results = {}
    for pair in pairs():
        real = cv2.resize(cv2.imread(str(pair["image"])), (640, 480))
        row = {}
        sims = {}
        for setup, (spec, scene) in SETUPS.items():
            sim = render(spec, scene, pair["joints"], pair["cup"])
            sims[setup] = sim
            er, es = far_edge_row(real), far_edge_row(sim)
            row[setup] = {
                "gripper_iou": round(gripper_iou(sim, real), 3),
                "edge_row_err": None if er is None or es is None else es - er,
                "colour_dE": round(colour_de(sim, real), 1),
                "table_rgb": table_rgb(sim),
            }
        row["real_table_rgb"] = table_rgb(real)
        results[pair["name"]] = {"note": pair["note"], **row}
        tiles = [label(real, f"real: {pair['name']}")] + [
            label(sims[s], f"{s}: IoU {row[s]['gripper_iou']}  dE {row[s]['colour_dE']}")
            for s in SETUPS
        ]
        blend = label(
            cv2.addWeighted(sims["gem13 cell"], 0.5, real, 0.5, 0),
            "50/50 blend: gem13 cell over real",
        )
        grid = np.vstack(
            [np.hstack(tiles[:2] + [blend]), np.hstack(tiles[2:] + [np.zeros_like(real)])]
        )
        cv2.imwrite(str(OUT / f"{pair['name']}.jpg"), grid, [cv2.IMWRITE_JPEG_QUALITY, 85])
        print(pair["name"], json.dumps(row))
    (OUT / "metrics.json").write_text(json.dumps(results, indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
