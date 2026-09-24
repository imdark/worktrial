"""Convert the Kronos gripper CAD into simulation assets.

    python scripts/build_kronos_assets.py

Reads Kronos's master assembly straight out of `assets/Hardware Archive.zip`
(gitignored: proprietary CAD) and writes, into assets/end_effectors/kronos/:

    meshes/*.stl          visual meshes, and convex pieces for collision, in metres,
                          each already expressed in the frame of the body it belongs to
    kronos_extract.json   everything the MJCF generator needs: body frames, pivot
                          axes, masses, centres of mass, inertia tensors, camera plate,
                          and where every number came from

Then run `python scripts/build_kronos_model.py` to turn that into kronos.xml. The
split keeps this step -- the only one needing the CAD and gmsh -- separate from the
model, which is plain Python and tested.

Requires gmsh (`uv pip install gmsh`). gmsh is GPL; it is used here as a
conversion tool only and is never imported by teleop_sim.

Frames. The CAD assembly frame has the fingers pointing +y, the camera neck +z,
the fingers opening along x. The end effector's root frame is placed so that it
lines up with YAM's flange (link_6): z along the tool (the fingers), y up (the
neck), x the closing direction:

    x_root = -x_cad    y_root = z_cad    z_root = y_cad

Its origin is where the gripper meets the wrist: the back of the housing
(y_cad = -62.8 mm), on the line through the finger mid-plane (x = 1.1, z = 22.9).
**That mount is inferred from the video, not read from the CAD** -- no bolt
circle facing the arm could be identified. Change MOUNT_CAD below if it is wrong.
"""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ARCHIVE = ROOT / "assets" / "Hardware Archive.zip"
MEMBER = "Hardware Archive/Part Files/Kronos/Master Assembly.step"
OUT = ROOT / "assets" / "end_effectors" / "kronos"

#: Root frame origin in CAD coordinates (mm). See the module docstring.
MOUNT_CAD = np.array([1.1, -62.8, 22.9])
#: CAD -> root rotation: rows are the root axes expressed in CAD coordinates.
ROT = np.array([[-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])

#: Real masses (g), and where each came from. CAD volume x material density
#: would overstate the printed parts badly: they are 15% infill.
MASSES_G = {
    "bottom_body": (42.7, "slicer: 512.8 g PETG-CF / 12 copies"),
    "mid_body": (45.0, "slicer: 675.7 g PETG-CF / 15 copies"),
    "camera_neck": (55.6, "slicer: 444.6 g PETG-CF / 8 copies"),
    "servo": (57.2, "ROBOTIS XL430-W250-T datasheet"),
    "board": (5.0, "ASSUMED: small UART-to-CAN PCB"),
    "finger_shell": (25.5, "slicer: 102.0 g PETG-CF / 4 fingers"),
    "finger_pad": (10.5, "slicer: 41.9 g TPU / 4 fingers (TPU pad + grip tape)"),
}


def _import(step: Path):
    import gmsh

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.option.setNumber("Geometry.OCCImportLabels", 1)
    gmsh.model.occ.importShapes(str(step))
    gmsh.model.occ.synchronize()
    return gmsh


def classify(gmsh) -> dict[str, list[int]]:
    """Every solid to one group, by the component names Fusion exported."""
    vols = {t: gmsh.model.getEntityName(3, t) for _, t in gmsh.model.getEntities(3)}
    unnamed = sorted(
        (t for t, n in vols.items() if not n),
        key=lambda t: gmsh.model.occ.getMass(3, t),
        reverse=True,
    )
    neck, mid = unnamed[0], unnamed[1]  # 114.5 and 33.3 cm3; checked below

    groups: dict[str, list[int]] = {
        k: []
        for k in (
            "bottom_body",
            "mid_body",
            "camera_neck",
            "servo_left",
            "servo_right",
            "board",
            "horn_left",
            "horn_right",
            "finger_left_shell",
            "finger_left_tpu",
            "finger_left_tape",
            "finger_left_dots",
            "finger_right_shell",
            "finger_right_tpu",
            "finger_right_tape",
            "finger_right_dots",
        )
    }
    for tag, name in vols.items():
        if tag == neck:
            groups["camera_neck"].append(tag)
        elif tag == mid:
            groups["mid_body"].append(tag)
        elif "Bottom (Whale)" in name:
            groups["bottom_body"].append(tag)
        elif "XL430_W250_T:" in name:
            # :2 sits under the left finger (x = -14.4 mm), :1 under the right.
            side = "left" if "XL430_W250_T:2" in name else "right"
            groups[f"horn_{side}" if "Part3^" in name else f"servo_{side}"].append(tag)
        elif "Finger Gripper 3.2:1" in name:
            side = "right" if "New Right" in name else "left"
            kind = (
                "tpu"
                if "TPU" in name
                else "tape"
                if "Griptape" in name
                else "dots"
                if ("Dot" in name or "Component1" in name)
                else "shell"
            )
            groups[f"finger_{side}_{kind}"].append(tag)
        else:
            groups["board"].append(tag)  # the bridge PCB and its components
    return groups


def mass_properties(gmsh, tags: list[int], mass_g: float):
    """Mass, CAD-frame centre of mass (mm) and inertia about it (g mm^2),
    distributing the real mass over the solids in proportion to volume."""
    vols = np.array([gmsh.model.occ.getMass(3, t) for t in tags])
    coms = np.array([gmsh.model.occ.getCenterOfMass(3, t) for t in tags])
    density = mass_g / vols.sum()
    com = (vols[:, None] * coms).sum(0) / vols.sum()
    inertia = np.zeros((3, 3))
    for t, c, v in zip(tags, coms, vols, strict=True):
        local = np.array(gmsh.model.occ.getMatrixOfInertia(3, t)).reshape(3, 3) * density
        d = c - com
        inertia += local + v * density * (d @ d * np.eye(3) - np.outer(d, d))
    return mass_g, com, inertia


def triangles(gmsh, tags: list[int]) -> np.ndarray:
    """Outward-oriented surface triangles of the given solids, CAD frame, mm."""
    out = []
    for t in tags:
        for _, s in gmsh.model.getBoundary([(3, t)], oriented=True):
            types, _, nodes = gmsh.model.mesh.getElements(2, abs(s))
            for typ, conn in zip(types, nodes, strict=True):
                if typ != 2:
                    continue
                conn = np.asarray(conn, dtype=np.int64).reshape(-1, 3)
                if s < 0:
                    conn = conn[:, ::-1]
                tags_n, coords, _ = gmsh.model.mesh.getNodes(2, abs(s), includeBoundary=True)
                lookup = dict(zip(tags_n.tolist(), np.asarray(coords).reshape(-1, 3), strict=True))
                out.append(np.array([[lookup[n] for n in tri] for tri in conn]))
    return np.concatenate(out) if out else np.zeros((0, 3, 3))


def to_frame(points_mm: np.ndarray, origin_cad_mm: np.ndarray) -> np.ndarray:
    """CAD mm -> a root-oriented frame at origin_cad_mm, in metres."""
    return ((points_mm - origin_cad_mm) @ ROT.T) / 1000.0


def convex_pieces(tris: np.ndarray, pieces: int) -> list[np.ndarray]:
    """Slice along the part's long axis; MuJoCo collides each slice as its hull.

    A single hull of a curved finger covers its concave inner face and makes
    contact where the real finger has none. Slices follow the curve closely
    enough for grasping, without a convex-decomposition dependency.
    """
    cen = tris.mean(axis=1)
    c0 = cen.mean(0)
    _, vec = np.linalg.eigh(np.cov((cen - c0).T))
    s = (cen - c0) @ vec[:, -1]
    edges = np.quantile(s, np.linspace(0, 1, pieces + 1))
    out = []
    for i in range(pieces):
        sel = (s >= edges[i]) & (s <= edges[i + 1])
        if sel.sum() >= 4:
            out.append(tris[sel])
    return out


def write_stl(path: Path, tris_m: np.ndarray, label: str) -> str:
    from teleop_sim.envs.yam_assets import write_stl as encode

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encode(tris_m, label.encode()))
    return str(path.relative_to(OUT))


def main() -> None:
    if not ARCHIVE.is_file():
        sys.exit(f"missing {ARCHIVE}: this step needs the hardware archive")
    with tempfile.TemporaryDirectory() as tmp:
        step = Path(tmp) / "kronos.step"
        data = zipfile.ZipFile(ARCHIVE).read(MEMBER)
        step.write_bytes(data)
        gmsh = _import(step)
        try:
            extract = convert(gmsh, hashlib.sha256(data).hexdigest())
        finally:
            gmsh.finalize()
    (OUT / "kronos_extract.json").write_text(json.dumps(extract, indent=2) + "\n")
    print(
        f"wrote {OUT.relative_to(ROOT)}/kronos_extract.json and "
        f"{len(list((OUT / 'meshes').glob('*.stl')))} meshes"
    )


def convert(gmsh, source_sha: str) -> dict:
    groups = classify(gmsh)
    vol = lambda tags: sum(gmsh.model.occ.getMass(3, t) for t in tags) / 1000  # noqa: E731
    checks = {"camera_neck": 114.47, "mid_body": 33.29, "bottom_body": 42.65}
    for name, expected in checks.items():
        if abs(vol(groups[name]) - expected) > 0.5:
            raise ValueError(
                f"{name}: {vol(groups[name]):.2f} cm3, expected {expected}; "
                "the assembly is not the one this script was written against"
            )

    # ---- pivots: the servo output horns, axis along CAD z
    pivots = {}
    for side in ("left", "right"):
        bb = np.array(gmsh.model.getBoundingBox(3, groups[f"horn_{side}"][0]))
        pivots[side] = (bb[:3] + bb[3:]) / 2
    # ---- mass properties per simulated body
    base_parts = {
        "bottom_body": MASSES_G["bottom_body"][0],
        "mid_body": MASSES_G["mid_body"][0],
        "camera_neck": MASSES_G["camera_neck"][0],
        "board": MASSES_G["board"][0],
    }
    horn_share = {}
    for side in ("left", "right"):
        v_servo, v_horn = vol(groups[f"servo_{side}"]), vol(groups[f"horn_{side}"])
        horn_share[side] = MASSES_G["servo"][0] * v_horn / (v_servo + v_horn)
        base_parts[f"servo_{side}"] = MASSES_G["servo"][0] - horn_share[side]

    def combine(parts):  # [(tags, grams)] -> mass, com, inertia about com
        items = [mass_properties(gmsh, tags, g) for tags, g in parts if tags]
        m = sum(i[0] for i in items)
        com = sum(i[0] * i[1] for i in items) / m
        inertia = np.zeros((3, 3))
        for mi, ci, ii in items:
            d = ci - com
            inertia += ii + mi * (d @ d * np.eye(3) - np.outer(d, d))
        return m, com, inertia

    bodies = {}
    base = combine([(groups[k], g) for k, g in base_parts.items()])
    bodies["base"] = dict(origin_cad_mm=MOUNT_CAD, mass=base)
    for side in ("left", "right"):
        f = f"finger_{side}"
        shell = (groups[f"{f}_shell"], MASSES_G["finger_shell"][0])
        pad = (groups[f"{f}_tpu"] + groups[f"{f}_tape"], MASSES_G["finger_pad"][0])
        horn = (groups[f"horn_{side}"], horn_share[side])
        bodies[f] = dict(origin_cad_mm=pivots[side], mass=combine([shell, pad, horn]))

    # ---- surface meshes, only for what is visible; tiny board components are skipped
    gmsh.option.setNumber("Mesh.MeshSizeMax", 3.0)
    gmsh.option.setNumber("Mesh.MeshSizeMin", 0.4)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 12)
    keep = {
        t
        for k, ts in groups.items()
        for t in ts
        if k != "board" or gmsh.model.occ.getMass(3, t) > 50.0
    }
    gmsh.model.mesh.generate(2)

    visuals = {
        "base": [
            ("housing", ["bottom_body", "mid_body", "camera_neck"], "0.13 0.13 0.14 1"),
            ("servos", ["servo_left", "servo_right"], "0.22 0.22 0.24 1"),
            ("board", ["board"], "0.10 0.38 0.20 1"),
        ],
        "finger_left": [
            ("shell", ["finger_left_shell"], "0.13 0.13 0.14 1"),
            ("pad", ["finger_left_tpu"], "0.92 0.92 0.90 1"),
            ("tape", ["finger_left_tape"], "0.55 0.55 0.55 1"),
            ("dots", ["finger_left_dots"], "0.95 0.45 0.10 1"),
            ("horn", ["horn_left"], "0.30 0.30 0.32 1"),
        ],
        "finger_right": [
            ("shell", ["finger_right_shell"], "0.13 0.13 0.14 1"),
            ("pad", ["finger_right_tpu"], "0.92 0.92 0.90 1"),
            ("tape", ["finger_right_tape"], "0.55 0.55 0.55 1"),
            ("dots", ["finger_right_dots"], "0.95 0.45 0.10 1"),
            ("horn", ["horn_right"], "0.30 0.30 0.32 1"),
        ],
    }
    collisions = {  # (name, groups, number of convex slices)
        "base": [
            ("bottom", ["bottom_body"], 2),
            ("mid", ["mid_body"], 2),
            ("neck", ["camera_neck"], 4),
        ],
        "finger_left": [
            ("shell", ["finger_left_shell"], 5),
            ("pad", ["finger_left_tpu", "finger_left_tape"], 4),
        ],
        "finger_right": [
            ("shell", ["finger_right_shell"], 5),
            ("pad", ["finger_right_tpu", "finger_right_tape"], 4),
        ],
    }

    out: dict = {
        "source": {
            "archive_member": MEMBER,
            "sha256": source_sha,
            "note": "Fusion 'Big Master' export of the Kronos gripper",
        },
        "frame": {
            "mount_cad_mm": MOUNT_CAD.tolist(),
            "rotation_cad_to_root": ROT.tolist(),
            "mount_note": "INFERRED from video; not read from a bolt pattern",
        },
        "masses_g": {k: {"grams": g, "source": s} for k, (g, s) in MASSES_G.items()},
        "bodies": {},
    }
    for name, info in bodies.items():
        origin = info["origin_cad_mm"]
        m, com, inertia = info["mass"]
        root_pos = to_frame(origin[None], MOUNT_CAD)[0] if name != "base" else np.zeros(3)
        body = {
            "pos": root_pos.tolist(),
            "mass_kg": m / 1000.0,
            "com": to_frame(com[None], origin)[0].tolist(),
            # g mm^2 -> kg m^2, rotated into the root orientation
            "inertia": (ROT @ inertia @ ROT.T * 1e-9).tolist(),
            "visual": [],
            "collision": [],
        }
        for label, keys, rgba in visuals[name]:
            tags = [t for k in keys for t in groups[k] if t in keep]
            tris = triangles(gmsh, tags)
            if not len(tris):
                continue
            file = write_stl(
                OUT / "meshes" / f"{name}_{label}.stl",
                to_frame(tris, origin),
                f"kronos {name} {label}",
            )
            body["visual"].append({"mesh": file, "rgba": rgba, "faces": int(len(tris))})
        for label, keys, n in collisions[name]:
            tris = to_frame(triangles(gmsh, [t for k in keys for t in groups[k]]), origin)
            for i, piece in enumerate(convex_pieces(tris, n)):
                file = write_stl(
                    OUT / "meshes" / f"{name}_{label}_hull{i}.stl",
                    piece,
                    f"kronos {name} {label} collision {i}",
                )
                body["collision"].append({"mesh": file, "kind": label})
        out["bodies"][name] = body

    # Hinge axes: CAD z, in root orientation; each finger's closing direction is
    # positive (left rotates about -y_root, right about +y_root), so one mirror
    # coupling `right = left` holds.
    axis = (ROT @ np.array([0.0, 0.0, 1.0])).tolist()
    out["bodies"]["finger_left"]["axis"] = [-a for a in axis]
    out["bodies"]["finger_right"]["axis"] = axis

    out["camera_plate"] = camera_plate(gmsh, groups["camera_neck"][0])
    out["pivots_cad_mm"] = {k: v.tolist() for k, v in pivots.items()}
    return out


def camera_plate(gmsh, neck: int) -> dict:
    """The camera mounting plate: two M3 holes 20 mm apart near the neck top."""
    bb = np.array(gmsh.model.getBoundingBox(3, neck))
    best = None
    for _, s in gmsh.model.getBoundary([(3, neck)], oriented=False):
        if gmsh.model.getType(2, s) != "Plane":
            continue
        c = np.array(gmsh.model.occ.getCenterOfMass(2, s))
        if c[2] < bb[5] - 45:
            continue
        lo, hi = np.array(gmsh.model.getParametrizationBounds(2, s))
        n = np.array(gmsh.model.getNormal(s, list((lo + hi) / 2)))
        area = gmsh.model.occ.getMass(2, s)
        # the plate's front face: largest high face whose normal points forward (+y_cad)
        if n[1] > 0.5 and (best is None or area > best[0]):
            best = (area, c, n)
    area, c, n = best
    return {
        "centre_root_m": to_frame(c[None], MOUNT_CAD)[0].tolist(),
        "normal_root": (ROT @ n).tolist(),
        "area_mm2": area,
        "note": "front face of the camera plate; camera model ASSUMED (D405)",
    }


if __name__ == "__main__":
    main()
