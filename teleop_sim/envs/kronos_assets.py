"""Generate the Kronos end effector's MJCF from its CAD extraction.

scripts/build_kronos_assets.py reads the CAD (needs the hardware archive and
gmsh) and writes meshes plus kronos_extract.json. This module turns that JSON
into kronos.xml with plain Python, so the model can be regenerated, reviewed and
tested without the proprietary CAD or GPL tooling.

What comes from where:

    geometry, pivots, CoM, inertia   the CAD assembly            (measured)
    masses                           slicer projects + datasheet (measured; board assumed)
    travel, 1:1 coupling             video + STL + gear teeth    (measured)
    torque limit                     2 x XL430 stall, 11.1 V     (datasheet)
    servo gains, armature, damping   typical values              (ASSUMED)
    pad friction                     TPU + grip tape             (ASSUMED)
    camera                           RealSense D405              (ASSUMED model)
    mount on the arm                 read off the video          (ASSUMED)
"""

from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from teleop_sim.envs.yam_assets import GENERATED_HEADER, look_at_xyaxes

PREFIX = "kronos_"


@dataclass(frozen=True)
class KronosActuation:
    """Two XL430-W250-T servos, geared 1:1, act as one actuator on the pair."""

    stall_torque_nm: float = 1.4 * 2  # datasheet, 11.1 V, both servos
    #: Position gain chosen so the torque saturates within ~0.1 rad of error --
    #: a servo that pushes at full stall on anything it closes on. ASSUMED.
    kp: float = 28.0
    #: Reflected rotor inertia through the 258.5:1 gearbox, and back-EMF damping. ASSUMED.
    armature: float = 0.005
    damping: float = 0.05
    frictionloss: float = 0.02
    #: Search bound for the closing angle; the pads meet well inside it.
    travel_rad: float = 1.0


@dataclass(frozen=True)
class KronosCamera:
    """Eye-in-hand camera on the neck's mounting plate. D405 ASSUMED."""

    name: str = "wrist"  # the same contract as every end effector
    body: str = f"{PREFIX}camera"
    housing_half: tuple[float, float, float] = (0.021, 0.021, 0.0115)  # D405 42x42x23 mm
    mass: float = 0.060
    fovy: float = 58.0


@dataclass(frozen=True)
class KronosContact:
    """TPU pads with grip tape: grippier and softer than the printed shell. ASSUMED."""

    pad_friction: tuple[float, float, float] = (1.2, 0.02, 0.002)
    shell_friction: tuple[float, float, float] = (0.6, 0.01, 0.001)
    pad_solref: tuple[float, float] = (0.02, 1.0)
    #: 4 = sliding + torsional friction. With MuJoCo's default 3, the torsional
    #: coefficient above is ignored and a pinched object pivots freely about
    #: the line between the two pads: in sim the gem13 cup swung 40-60 deg in
    #: the grasp and fell when placed, while on the rig the soft, textured pads
    #: held it upright (2026-09-24).
    pad_condim: int = 4


#: Joint angle 0 is the STEP assembly's pose: fingers splayed ~39 deg, the widest
#: pose in the CAD, kept as the joint's mechanical lower limit. "Fully open" in
#: use is narrower: the widest opening in the teleop video matches the STL export,
#: whose fingers sit 18.5 / 19.2 deg closer to closed (PCA of the finger shells in
#: both files). That pose is the gripper's normalised 0.0, so a policy's "open"
#: means the same physical pose in sim and on the arm.
OPERATING_OPEN_RAD = math.radians(18.85)

#: Where ee_pose points: on the symmetry line, 20 mm back from where the
#: fingertips meet (CAD y = 131 mm), in the fingers' mid-plane. Root frame, m.
GRASP_POINT = (0.0, 0.0, 0.175)


def _vec(values, fmt="{:.6g}") -> str:
    return " ".join(fmt.format(float(v)) for v in values)


def _fullinertia(i: list[list[float]]) -> str:
    m = np.asarray(i)
    return _vec([m[0, 0], m[1, 1], m[2, 2], m[0, 1], m[0, 2], m[1, 2]])


def build_kronos(
    extract: dict,
    closed_rad: float,
    actuation: KronosActuation = KronosActuation(),  # noqa: B008
    camera: KronosCamera = KronosCamera(),  # noqa: B008
    contact: KronosContact = KronosContact(),  # noqa: B008
) -> str:
    """MJCF for the Kronos end effector.

    ``closed_rad`` is the finger angle at which the pads meet, measured from these
    same pad collision shapes (scripts/build_kronos_model.py). It becomes the
    joint's hard stop and the top of the actuator range, so "fully closed"
    means what it means on the real gripper: tips together.
    """
    root = ET.Element("mujoco", model="kronos")
    ET.SubElement(root, "compiler", angle="radian")

    default = ET.SubElement(root, "default")

    def cls(name, **kw):
        d = ET.SubElement(default, "default", {"class": PREFIX + name})
        return d, kw

    d, _ = cls("visual")
    ET.SubElement(d, "geom", type="mesh", contype="0", conaffinity="0", group="2", density="0")
    d, _ = cls("shell")
    ET.SubElement(
        d, "geom", type="mesh", group="3", density="0", friction=_vec(contact.shell_friction)
    )
    d, _ = cls("pad")
    ET.SubElement(
        d,
        "geom",
        type="mesh",
        group="3",
        density="0",
        friction=_vec(contact.pad_friction),
        solref=_vec(contact.pad_solref),
        condim=str(contact.pad_condim),
    )
    d, _ = cls("finger")
    ET.SubElement(
        d,
        "joint",
        type="hinge",
        range=f"0 {closed_rad:.6g}",
        armature=f"{actuation.armature:g}",
        damping=f"{actuation.damping:g}",
        frictionloss=f"{actuation.frictionloss:g}",
    )

    asset = ET.SubElement(root, "asset")
    bodies = extract["bodies"]
    for body in bodies.values():
        for g in body["visual"] + body["collision"]:
            ET.SubElement(asset, "mesh", name=PREFIX + Path(g["mesh"]).stem, file=g["mesh"])

    worldbody = ET.SubElement(root, "worldbody")
    base = bodies["base"]
    kronos = ET.SubElement(worldbody, "body", name="kronos")
    ET.SubElement(
        kronos,
        "inertial",
        pos=_vec(base["com"]),
        mass=f"{base['mass_kg']:.6g}",
        fullinertia=_fullinertia(base["inertia"]),
    )
    _geoms(kronos, base)
    ET.SubElement(
        kronos,
        "site",
        name=f"{PREFIX}grasp",
        pos=_vec(GRASP_POINT),
        size="0.004",
        rgba="0 1 0 1",
        group="4",
    )
    _camera(kronos, extract["camera_plate"], camera)

    for side in ("left", "right"):
        info = bodies[f"finger_{side}"]
        finger = ET.SubElement(kronos, "body", name=f"{PREFIX}finger_{side}", pos=_vec(info["pos"]))
        ET.SubElement(
            finger,
            "inertial",
            pos=_vec(info["com"]),
            mass=f"{info['mass_kg']:.6g}",
            fullinertia=_fullinertia(info["inertia"]),
        )
        ET.SubElement(
            finger,
            "joint",
            {"class": PREFIX + "finger"},
            name=f"{PREFIX}finger_{side}",
            axis=_vec(info["axis"]),
        )
        _geoms(finger, info)

    # The finger hubs turn inside the housing around the servo horns, so their
    # convex collision slices overlap the housing's by construction. Say so
    # explicitly: MuJoCo's parent-child filter would hide it on the arm, but not
    # when the gripper's root is welded to the world, and relying on it would
    # make the gripper's behaviour depend on what it is attached to.
    contact_block = ET.SubElement(root, "contact")
    for side in ("left", "right"):
        ET.SubElement(contact_block, "exclude", body1="kronos", body2=f"{PREFIX}finger_{side}")
    # Finger against finger is excluded too: the hubs are meshed gears whose
    # teeth interlock by design, so their slices always overlap. The coupling
    # below already is that mechanism, and "closed" is the joint limit at the
    # angle the pads meet -- measured from the pad geometry, not left to a
    # contact between shapes that are only approximations of the gear.
    ET.SubElement(
        contact_block, "exclude", body1=f"{PREFIX}finger_left", body2=f"{PREFIX}finger_right"
    )

    # The two finger hubs are meshed 1:1 gears: exact mirror motion. Axes are
    # already signed so that closing is positive on both.
    equality = ET.SubElement(root, "equality")
    ET.SubElement(
        equality,
        "joint",
        joint1=f"{PREFIX}finger_right",
        joint2=f"{PREFIX}finger_left",
        polycoef="0 1 0 0 0",
    )
    actuator = ET.SubElement(root, "actuator")
    torque = actuation.stall_torque_nm
    ET.SubElement(
        actuator,
        "position",
        name=f"{PREFIX}grip",
        joint=f"{PREFIX}finger_left",
        ctrlrange=f"0 {closed_rad:.6g}",
        kp=f"{actuation.kp:g}",
        dampratio="1",
        forcerange=f"{-torque:g} {torque:g}",
    )

    ET.indent(root, space="  ")
    return GENERATED_HEADER + ET.tostring(root, encoding="unicode") + "\n"


def _geoms(body: ET.Element, info: dict) -> None:
    for g in info["visual"]:
        ET.SubElement(
            body,
            "geom",
            {"class": PREFIX + "visual"},
            mesh=PREFIX + Path(g["mesh"]).stem,
            rgba=g["rgba"],
        )
    for g in info["collision"]:
        kind = "pad" if g["kind"] == "pad" else "shell"
        ET.SubElement(
            body,
            "geom",
            {"class": PREFIX + kind},
            name=PREFIX + Path(g["mesh"]).stem,
            mesh=PREFIX + Path(g["mesh"]).stem,
        )


def _camera(parent: ET.Element, plate: dict, camera: KronosCamera) -> None:
    """Camera housing on the plate's front face, looking along the plate normal."""
    normal = np.asarray(plate["normal_root"], dtype=float)
    normal /= np.linalg.norm(normal)
    centre = np.asarray(plate["centre_root_m"]) + normal * camera.housing_half[2]
    up = np.array([0.0, 1.0, 0.0]) - normal * normal[1]  # "up" = the neck, square to the view
    xyaxes = look_at_xyaxes(centre, centre + normal, up)
    body = ET.SubElement(parent, "body", name=camera.body, pos=_vec(centre), xyaxes=xyaxes)
    ET.SubElement(
        body,
        "inertial",
        pos="0 0 0",
        mass=f"{camera.mass:g}",
        diaginertia="1.2e-05 1.2e-05 1.8e-05",
    )
    ET.SubElement(
        body,
        "geom",
        name=f"{PREFIX}camera_housing",
        type="box",
        size=_vec(camera.housing_half),
        contype="0",
        conaffinity="0",
        group="2",
        rgba="0.72 0.73 0.75 1",
    )
    # The body's -z is the viewing direction, so the lens sits on its -z face.
    ET.SubElement(
        body,
        "camera",
        name=camera.name,
        pos=f"0 0 {-camera.housing_half[2]:.6g}",
        fovy=f"{camera.fovy:g}",
    )


def load_extract(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def degrees(rad: float) -> float:
    return math.degrees(rad)
