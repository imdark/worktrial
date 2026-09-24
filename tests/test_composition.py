"""Composable descriptions: arm + end effector (+ scene).

The claims under test:

* splitting Menagerie's YAM into arm + stock gripper loses nothing;
* a different end effector attaches to the same arm without touching it;
* the same robot can be placed in a cell or stand alone (the real-robot view);
* every way parts can fail to fit together fails loudly, at construction.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

from teleop_sim.core.parts import ArmSpec, EndEffectorSpec, SceneSpec, compose
from teleop_sim.core.spec import RobotSpec, SpecError
from tests.conftest import PACKAGE_ROOT, REPO_ROOT

mujoco = pytest.importorskip("mujoco", reason="needs the 'sim' extra")

from teleop_sim.robots.sim.assembly import ModelMismatch, build_model, export_xml  # noqa: E402

ROBOTS = PACKAGE_ROOT / "robots"
YAM = ROBOTS / "specs" / "yam.yaml"
ARM = ROBOTS / "arms" / "yam.yaml"
LINEAR = ROBOTS / "end_effectors" / "yam_linear.yaml"
GLASSES = PACKAGE_ROOT / "scenes" / "glasses_table.yaml"
TEST_JAW_ROBOT = REPO_ROOT / "tests" / "assets" / "yam_test_jaw.yaml"
UPSTREAM = REPO_ROOT / "assets" / "i2rt_yam" / "upstream" / "yam.xml"

J, A, B, S = (mujoco.mjtObj.mjOBJ_JOINT, mujoco.mjtObj.mjOBJ_ACTUATOR,
              mujoco.mjtObj.mjOBJ_BODY, mujoco.mjtObj.mjOBJ_SITE)


def _name(model, kind, index):
    return mujoco.mj_id2name(model, kind, index)


def _id(model, kind, name):
    return mujoco.mj_name2id(model, kind, name)


@pytest.fixture(scope="module")
def upstream():
    return mujoco.MjModel.from_xml_path(str(UPSTREAM))


@pytest.fixture(scope="module")
def composed():
    return build_model(RobotSpec.from_yaml(YAM))


# ------------------------------------------------ the split is lossless


def test_arm_plus_stock_gripper_has_upstreams_joints(composed, upstream):
    for i in range(upstream.njnt):
        name = _name(upstream, J, i)
        j = _id(composed, J, name)
        assert j >= 0, name
        np.testing.assert_allclose(composed.jnt_range[j], upstream.jnt_range[i], err_msg=name)
        for field in ("dof_armature", "dof_frictionloss", "dof_damping"):
            got = getattr(composed, field)[composed.jnt_dofadr[j]]
            want = getattr(upstream, field)[upstream.jnt_dofadr[i]]
            assert got == pytest.approx(want), (name, field)


def test_arm_plus_stock_gripper_has_upstreams_actuators(composed, upstream):
    assert composed.nu == upstream.nu
    for i in range(upstream.nu):
        name = _name(upstream, A, i)
        a = _id(composed, A, name)
        for field in ("actuator_gainprm", "actuator_biasprm", "actuator_ctrlrange",
                      "actuator_forcerange"):
            np.testing.assert_allclose(getattr(composed, field)[a], getattr(upstream, field)[i],
                                       err_msg=f"{name}.{field}")


def test_arm_plus_stock_gripper_has_upstreams_mass(composed, upstream):
    """Every upstream body keeps its mass. Two things are added, both on
    purpose: the wrist camera (60 g, which upstream does not have) and the
    gripper's root body at 1 mg -- without that explicit inertia MuJoCo would
    derive 52 g from the finger rails' collision geoms, which upstream's
    link_6 inertia overrode."""
    from teleop_sim.envs.yam_assets import WristCamera

    for i in range(1, upstream.nbody):
        name = _name(upstream, B, i)
        assert composed.body_mass[_id(composed, B, name)] == pytest.approx(
            upstream.body_mass[i]), name
    added = WristCamera().mass + 1e-6
    assert composed.body_subtreemass[0] == pytest.approx(
        upstream.body_subtreemass[0] + added, abs=1e-6)


def test_arm_plus_stock_gripper_has_upstreams_kinematics(composed, upstream):
    rng = np.random.default_rng(0)
    cd, ud = mujoco.MjData(composed), mujoco.MjData(upstream)
    for _ in range(50):
        for k in range(1, 7):
            value = rng.uniform(*upstream.jnt_range[k - 1])
            cd.qpos[composed.jnt_qposadr[_id(composed, J, f"joint{k}")]] = value
            ud.qpos[upstream.jnt_qposadr[_id(upstream, J, f"joint{k}")]] = value
        mujoco.mj_kinematics(composed, cd)
        mujoco.mj_kinematics(upstream, ud)
        np.testing.assert_allclose(
            cd.site_xpos[_id(composed, S, "grasp_site")],
            ud.site_xpos[_id(upstream, S, "grasp_site")],
            atol=1e-12,
        )


def test_arm_home_is_upstreams_home_keyframe(upstream):
    key = upstream.key_qpos[_id(upstream, mujoco.mjtObj.mjOBJ_KEY, "home")]
    arm = ArmSpec.from_yaml(ARM)
    for joint, value in zip(arm.joints, arm.home, strict=True):
        assert key[upstream.jnt_qposadr[_id(upstream, J, joint.name)]] == pytest.approx(
            value, abs=1e-3), joint.name


# ------------------------------------------------ swapping end effectors


@pytest.fixture(scope="module")
def two_robots():
    return RobotSpec.from_yaml(YAM), RobotSpec.from_yaml(TEST_JAW_ROBOT)


def test_same_arm_description_serves_both_end_effectors(two_robots):
    stock, jaw = two_robots
    assert stock.assembly.arm == jaw.assembly.arm
    assert stock.joint_names == jaw.joint_names
    assert stock.gripper.joint != jaw.gripper.joint
    assert stock.ee_link == "yam_linear" and jaw.ee_link == "test_jaw"


def test_swapping_the_end_effector_leaves_the_arm_kinematics_alone(two_robots):
    models = [build_model(spec) for spec in two_robots]
    datas = [mujoco.MjData(m) for m in models]
    rng = np.random.default_rng(1)
    for _ in range(20):
        q = [rng.uniform(*models[0].jnt_range[k]) for k in range(6)]
        for m, d in zip(models, datas, strict=True):
            for k, value in enumerate(q, start=1):
                d.qpos[m.jnt_qposadr[_id(m, J, f"joint{k}")]] = value
            mujoco.mj_kinematics(m, d)
        flange = [d.xpos[_id(m, B, "link_6")] for m, d in zip(models, datas, strict=True)]
        np.testing.assert_allclose(flange[0], flange[1], atol=1e-12)


@pytest.mark.parametrize("robot", [YAM, TEST_JAW_ROBOT], ids=["yam_linear", "test_jaw"])
@pytest.mark.parametrize("gripper", [0.0, 1.0], ids=["open", "closed"])
def test_end_effector_touches_nothing_on_the_arm(robot, gripper):
    """A mounting mistake -- fingers placed where they hit the wrist -- stops
    the gripper closing, and nothing else catches it. The first version of
    the test jaw did exactly that against link_5."""
    spec = RobotSpec.from_yaml(robot)
    m = build_model(spec)
    d = mujoco.MjData(m)
    arm = spec.assembly.arm
    for joint, value in zip(arm.joints, spec.home_joints(), strict=True):
        d.qpos[m.jnt_qposadr[_id(m, J, joint.name)]] = value
        d.ctrl[_id(m, A, joint.name)] = value
    d.ctrl[_id(m, A, spec.gripper.actuator_name)] = spec.gripper.denormalize(gripper)
    for _ in range(1500):
        mujoco.mj_step(m, d)

    ee_root = _id(m, B, spec.assembly.end_effector.root_body)
    def in_end_effector(body):
        while body not in (0, ee_root):
            body = m.body_parentid[body]
        return body == ee_root
    arm_bodies = {b for b in range(1, m.nbody) if not in_end_effector(b)}
    hits = sorted({
        (_name(m, B, m.geom_bodyid[c.geom1]), _name(m, B, m.geom_bodyid[c.geom2]))
        for c in d.contact[: d.ncon]
        if {m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]} & arm_bodies
        and (in_end_effector(m.geom_bodyid[c.geom1]) or in_end_effector(m.geom_bodyid[c.geom2]))
    })
    assert not hits, f"end effector touches the arm: {hits}"


def test_end_effectors_hash_differently(two_robots):
    """A dataset recorded with the stock gripper must never pass for one
    recorded with the production gripper."""
    stock, jaw = two_robots
    assert stock.content_hash() != jaw.content_hash()


def test_editing_an_end_effector_mesh_changes_the_hash(tmp_path):
    """Hashing follows every mesh, so a changed finger is a changed robot."""
    ee_dir = tmp_path / "ee"
    shutil.copytree(REPO_ROOT / "assets" / "end_effectors" / "yam_linear", ee_dir)
    meshes = tmp_path / "meshes"
    shutil.copytree(UPSTREAM.parent / "assets", meshes)
    xml = ee_dir / "yam_linear.xml"
    xml.write_text(xml.read_text().replace('meshdir="../../i2rt_yam/upstream/assets"',
                                           f'meshdir="{meshes}"'))
    desc = tmp_path / "yam_linear.yaml"
    desc.write_text(LINEAR.read_text().replace(
        "../../../assets/end_effectors/yam_linear/yam_linear.xml", str(xml)))
    arm = ArmSpec.from_yaml(ARM)
    before = compose(arm, EndEffectorSpec.from_yaml(desc)).content_hash()

    finger = meshes / "model2__14.stl"
    finger.write_bytes(finger.read_bytes() + b"\0")
    after = compose(arm, EndEffectorSpec.from_yaml(desc)).content_hash()
    assert before != after


# ------------------------------------------------ scenes


def test_the_robot_needs_no_scene():
    """The real robot's view: arm + end effector, nothing else."""
    spec = RobotSpec.from_yaml(YAM)
    model = build_model(spec)
    assert spec.assembly.scene is None
    assert spec.camera_names == ["wrist"]
    assert spec.workspace is None
    assert _id(model, B, "glass_left") < 0


def test_placing_the_robot_in_a_scene_adds_the_cell_and_nothing_else():
    bare = RobotSpec.from_yaml(YAM)
    cell = bare.with_scene(SceneSpec.from_yaml(GLASSES))
    assert cell.camera_names == ["wrist", "top"]
    assert cell.workspace is not None
    assert cell.joint_names == bare.joint_names
    assert cell.gripper == bare.gripper
    assert cell.content_hash() != bare.content_hash()
    model = build_model(cell)
    assert _id(model, B, "glass_left") >= 0 and _id(model, B, "glass_right") >= 0


def test_scene_physics_options_reach_the_model():
    model = build_model(RobotSpec.from_yaml(YAM).with_scene(SceneSpec.from_yaml(GLASSES)))
    assert model.opt.cone == mujoco.mjtCone.mjCONE_ELLIPTIC
    assert model.opt.integrator == mujoco.mjtIntegrator.mjINT_IMPLICITFAST


def test_a_bare_robot_keeps_the_arms_own_physics():
    model = build_model(RobotSpec.from_yaml(YAM))
    assert model.opt.integrator == mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    assert model.opt.timestep == pytest.approx(1 / 600)


def test_run_config_places_the_robot_in_its_scene():
    from teleop_sim.core.clock import ManualClock
    from teleop_sim.core.config import RunConfig, build_system

    system = build_system(
        RunConfig.from_yaml(PACKAGE_ROOT / "configs" / "yam_glasses.yaml"), clock=ManualClock()
    )
    assert system.spec.assembly.scene.name == "glasses_table"
    assert sorted(system.robot.camera_config()) == ["top", "wrist"]


def test_export_reloads_as_the_same_model(tmp_path):
    spec = RobotSpec.from_yaml(YAM).with_scene(SceneSpec.from_yaml(GLASSES))
    built = build_model(spec)
    exported = mujoco.MjModel.from_xml_path(str(export_xml(spec, tmp_path / "m.xml")))
    for field in ("njnt", "nu", "nbody", "ngeom", "ncam"):
        assert getattr(exported, field) == getattr(built, field), field
    np.testing.assert_allclose(exported.jnt_range, built.jnt_range)
    np.testing.assert_allclose(exported.body_mass, built.body_mass)
    np.testing.assert_allclose(exported.actuator_gainprm, built.actuator_gainprm)


# ------------------------------------------------ parts that do not fit fail loudly


def _scene_file(tmp_path: Path, option: str) -> Path:
    xml = tmp_path / "scene.xml"
    xml.write_text(f'<mujoco><option {option}/><worldbody/></mujoco>')
    desc = tmp_path / "scene.yaml"
    desc.write_text(f"name: bad_physics\nmjcf_path: {xml}\n")
    return desc


def test_a_scene_that_drops_the_arms_integrator_is_refused(tmp_path):
    """MuJoCo keeps the scene's value and only warns. The YAM silently losing
    its implicit integrator changes its dynamics, so it is an error here."""
    scene = SceneSpec.from_yaml(
        _scene_file(tmp_path, 'integrator="Euler" timestep="0.0016666666666666668"'))
    with pytest.raises(ModelMismatch, match="integrator"):
        build_model(RobotSpec.from_yaml(YAM).with_scene(scene))


def test_a_scene_with_a_different_timestep_is_refused(tmp_path):
    scene = SceneSpec.from_yaml(
        _scene_file(tmp_path, 'integrator="implicitfast" timestep="0.001"'))
    with pytest.raises(ModelMismatch, match="timestep"):
        build_model(RobotSpec.from_yaml(YAM).with_scene(scene))


def test_colliding_asset_names_are_refused(tmp_path):
    """An end effector must namespace its assets; an un-namespaced 'black'
    material collides with the arm's and compilation stops."""
    xml = tmp_path / "clash.xml"
    xml.write_text(
        '<mujoco><asset><material name="black" rgba="0 0 0 1"/></asset><worldbody>'
        '<body name="clash"><geom type="box" size=".01 .01 .01" material="black"/>'
        '<body name="clash_finger"><joint name="clash_finger" type="slide" range="0 .01"/>'
        '<geom type="box" size=".005 .005 .005"/></body></body></worldbody>'
        '<actuator><position name="clash_grip" joint="clash_finger" ctrlrange="0 .01"/>'
        '</actuator></mujoco>'
    )
    desc = tmp_path / "clash.yaml"
    desc.write_text(
        f"name: clash\nmjcf_path: {xml}\nroot_body: clash\n"
        "gripper: {joint: clash_finger, actuator: clash_grip, open_pos: 0.0, closed_pos: 0.01}\n"
    )
    with pytest.raises(ModelMismatch, match="black"):
        build_model(compose(ArmSpec.from_yaml(ARM), EndEffectorSpec.from_yaml(desc)))


def test_an_end_effector_root_that_does_not_exist_is_named(tmp_path):
    desc = tmp_path / "wrong_root.yaml"
    desc.write_text(LINEAR.read_text().replace("root_body: yam_linear", "root_body: nope")
                    .replace("../../../assets", str(REPO_ROOT / "assets")))
    with pytest.raises(ModelMismatch, match="'nope'"):
        build_model(compose(ArmSpec.from_yaml(ARM), EndEffectorSpec.from_yaml(desc)))


def test_a_camera_declared_by_both_gripper_and_scene_is_refused(tmp_path):
    desc = tmp_path / "scene.yaml"
    desc.write_text(GLASSES.read_text().replace("name: top", "name: wrist")
                    .replace("../../assets", str(REPO_ROOT / "assets")))
    with pytest.raises(SpecError, match="wrist"):
        RobotSpec.from_yaml(YAM).with_scene(SceneSpec.from_yaml(desc))


def test_a_scene_camera_must_be_fixed(tmp_path):
    desc = tmp_path / "scene.yaml"
    desc.write_text(GLASSES.read_text().replace("mount: world", "mount: link_6")
                    .replace("../../assets", str(REPO_ROOT / "assets")))
    with pytest.raises(SpecError, match="belong to the end effector"):
        SceneSpec.from_yaml(desc)


def test_a_monolithic_robot_cannot_take_a_scene():
    so101 = RobotSpec.from_yaml(ROBOTS / "specs" / "so101.yaml")
    with pytest.raises(SpecError, match="monolithic"):
        so101.with_scene(SceneSpec.from_yaml(GLASSES))


@pytest.mark.parametrize(
    "text, message",
    [
        ("end_effector: x.yaml\n", "needs 'arm'"),
        ("arm: a.yaml\nend_effector: e.yaml\nextra: 1\n", "unknown field"),
    ],
)
def test_malformed_compositions_are_refused(tmp_path, text, message):
    path = tmp_path / "robot.yaml"
    path.write_text("name: broken\n" + text)
    with pytest.raises(SpecError, match=message):
        RobotSpec.from_yaml(path)
