"""Build one MuJoCo model from a RobotSpec.

A monolithic spec loads its MJCF directly. A composed spec is assembled with
MuJoCo's own model-composition API (MjSpec): the end effector's root body is
attached at the arm's flange, and the arm's root body is attached into the
scene -- or into a bare world when there is no scene, which is the view of the
robot the real driver has.

Two rules the assembler enforces, because MuJoCo only warns about them:

* Physics options come from the root. If a part asks for a different
  integrator or timestep than the scene provides, composition fails. MuJoCo
  itself keeps the scene's value and prints a warning, and a YAM arm silently
  losing its implicit integrator changes its dynamics.
* Names must be unique across parts. Parts namespace their own assets and
  default classes (``yam_linear_black``, not ``black``); MuJoCo rejects repeats
  at compile time, which is the loud failure wanted.
"""

from __future__ import annotations

from pathlib import Path

from teleop_sim.core.spec import RobotSpec, SpecError

#: Options a part may set, compared against the scene's.
_OPTIONS = ("timestep", "integrator", "cone", "impratio")


class ModelMismatch(SpecError):
    """The MJCF and the robot description disagree, or parts do not fit together."""


def _mujoco():
    import mujoco

    return mujoco


def _load(path: str | None, what: str):
    mujoco = _mujoco()
    if not path:
        raise ModelMismatch(f"{what} has no mjcf_path")
    if not Path(path).is_file():
        raise ModelMismatch(f"{what} model not found: {path}")
    spec = mujoco.MjSpec.from_file(path)
    _absolutize_assets(spec, Path(path).parent)
    return spec


def _absolutize_assets(spec, base: Path) -> None:
    """Pin asset files to absolute paths before attaching.

    meshdir and texturedir are resolved against the *root* model's directory at
    compile time, so a part's relative mesh paths would otherwise be looked up
    next to whichever scene it was attached into.
    """
    mesh_dir = base / spec.meshdir if spec.meshdir else base
    texture_dir = base / spec.texturedir if spec.texturedir else base
    for mesh in spec.meshes:
        if mesh.file and not Path(mesh.file).is_absolute():
            mesh.file = str((mesh_dir / mesh.file).resolve())
    for texture in spec.textures:
        if texture.file and not Path(texture.file).is_absolute():
            texture.file = str((texture_dir / texture.file).resolve())
    spec.meshdir = ""
    spec.texturedir = ""


def _body(spec, name: str, what: str):
    body = spec.body(name)
    if body is None:
        raise ModelMismatch(f"{what} names body {name!r}, which its MJCF does not define")
    return body


def _check_options(root, parts: list[tuple[object, str]], root_label: str) -> None:
    default = _mujoco().MjSpec().option
    for part, label in parts:
        for name in _OPTIONS:
            wanted, have = getattr(part.option, name), getattr(root.option, name)
            if wanted != getattr(default, name) and wanted != have:
                raise ModelMismatch(
                    f"{label} sets option {name}={wanted}, but {root_label} uses {have}. "
                    f"Physics options come from the scene; set {name} there to match."
                )


def _bare_world(arm):
    """A root with nothing in it but the arm's own physics options."""
    root = _mujoco().MjSpec()
    for name in _OPTIONS:
        setattr(root.option, name, getattr(arm.option, name))
    return root


def build_mjspec(spec: RobotSpec):
    """The robot (and its scene, if any) as an uncompiled MjSpec."""
    mujoco = _mujoco()
    if spec.assembly is None:
        if not spec.mjcf_path:
            raise ModelMismatch(f"robot {spec.name!r}: mjcf_path is not set")
        return mujoco.MjSpec.from_file(spec.mjcf_path)

    parts = spec.assembly
    arm_label = f"arm {parts.arm.name!r}"
    ee_label = f"end effector {parts.end_effector.name!r}"

    arm = _load(parts.arm.mjcf_path, arm_label)
    end_effector = _load(parts.end_effector.mjcf_path, ee_label)

    mount = parts.end_effector.mount
    flange = _body(arm, parts.arm.flange, arm_label)
    flange.add_frame(pos=list(mount[:3]), quat=list(mount[3:])).attach_body(
        _body(end_effector, parts.end_effector.root_body, ee_label), "", ""
    )

    if parts.scene is not None:
        root = _load(parts.scene.mjcf_path, f"scene {parts.scene.name!r}")
        root_label = f"scene {parts.scene.name!r}"
        arm_mount = parts.scene.arm_mount
    else:
        root = _bare_world(arm)
        root_label = "the bare world"
        arm_mount = (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
    _check_options(root, [(arm, arm_label), (end_effector, ee_label)], root_label)

    root.worldbody.add_frame(pos=list(arm_mount[:3]), quat=list(arm_mount[3:])).attach_body(
        _body(arm, parts.arm.root_body, arm_label), "", ""
    )
    return root


def build_model(spec: RobotSpec):
    """Compile the robot (and scene) into an MjModel."""
    try:
        return build_mjspec(spec).compile()
    except ValueError as exc:
        if isinstance(exc, SpecError):
            raise
        raise ModelMismatch(f"robot {spec.name!r} failed to compile: {exc}") from exc


def export_xml(spec: RobotSpec, path: str | Path) -> Path:
    """Write the composed model as one MJCF, for viewers and inspection.

    Asset paths in the export are absolute, so the file is a build artefact
    for this machine rather than something to commit.
    """
    mj_spec = build_mjspec(spec)
    mj_spec.compile()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_flatten_part_defaults(mj_spec.to_xml(), spec.name))
    return path


def _flatten_part_defaults(xml: str, name: str) -> str:
    """Make MjSpec.to_xml() output reloadable after attach.

    to_xml writes each attached part's top-level default as an *unnamed*
    <default> nested inside the root's, and MJCF allows only one unnamed
    default. The parts' classes are namespaced, so unwrapping those wrappers
    changes nothing -- unless a wrapper carries settings of its own, which
    would then apply to the whole model. That case is refused, not guessed at.
    """
    import xml.etree.ElementTree as ET

    root = ET.fromstring(xml)
    top = root.find("default")
    if top is None:
        return xml
    if any(c.tag != "default" for c in top):
        raise ModelMismatch(f"cannot export {name!r}: the root model sets unnamed defaults")
    # Parts attached into parts nest their wrappers (end effector inside arm), so
    # flatten at any depth, one wrapper at a time: ElementTree elements carry no
    # parent pointer, so the parent is looked up afresh on every pass. Each
    # wrapper was its own part's root default, so its classes belong at the top
    # level -- not under whatever class happened to hold the wrapper.
    while True:
        found = next(
            ((parent, child) for parent in top.iter("default") for child in parent
             if child.tag == "default" and child.get("class") is None),
            None,
        )
        if found is None:
            break
        parent, wrapper = found
        stray = [c.tag for c in wrapper if c.tag != "default"]
        if stray:
            raise ModelMismatch(
                f"cannot export {name!r}: an attached part sets unnamed defaults "
                f"{stray}, which would leak onto the whole model once flattened"
            )
        parent.remove(wrapper)
        top.extend(list(wrapper))
    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode") + "\n"
