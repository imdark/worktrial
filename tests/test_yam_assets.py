"""The vendored YAM model stays pristine and the generated files stay current."""

from __future__ import annotations

import hashlib
import importlib.util
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from teleop_sim.envs.yam_assets import (
    DEFAULT_GLASSES,
    EE_PREFIX,
    GENERATED_HEADER,
    build_arm,
    build_linear_gripper,
    look_at_xyaxes,
)
from tests.conftest import REPO_ROOT

YAM = REPO_ROOT / "assets" / "i2rt_yam"
UPSTREAM = YAM / "upstream"


def _build_script():
    spec = importlib.util.spec_from_file_location(
        "build_yam_assets", REPO_ROOT / "scripts" / "build_yam_assets.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_upstream_is_byte_for_byte_what_was_vendored():
    """VENDOR.md promises upstream/ is never edited. This makes it a check."""
    expected = {}
    for line in (YAM / "upstream.sha256").read_text().splitlines():
        digest, rel = line.split(maxsplit=1)
        expected[rel.removeprefix("./")] = digest
    actual = {
        str(p.relative_to(UPSTREAM)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(UPSTREAM.rglob("*"))
        if p.is_file()
    }
    assert actual == expected


def _png_pixels(data: bytes):
    """Decode the generator's own PNGs: 8-bit RGB, one IDAT, filter type 0 rows."""
    import struct
    import zlib

    width, height = struct.unpack(">II", data[16:24])
    start = data.index(b"IDAT") + 4
    length = struct.unpack(">I", data[start - 8:start - 4])[0]
    raw = zlib.decompress(data[start:start + length])
    rows = np.frombuffer(raw, np.uint8).reshape(height, 1 + width * 3)
    assert (rows[:, 0] == 0).all(), "unexpected PNG row filter"
    return rows[:, 1:].reshape(height, width, 3)


@pytest.mark.parametrize("path", list(_build_script().outputs()), ids=lambda p: p.name)
def test_generated_files_are_current(path):
    expected = _build_script().outputs()[path]
    if path.suffix == ".png":
        # A procedural texture: float maths may round one level differently on
        # another machine, so compare pixels, not bytes.
        on_disk, fresh = _png_pixels(path.read_bytes()), _png_pixels(expected)
        assert on_disk.shape == fresh.shape
        assert np.abs(on_disk.astype(int) - fresh.astype(int)).max() <= 1, (
            f"{path.name} differs from the generator's output -- run "
            "`python scripts/build_yam_assets.py`"
        )
        return
    if isinstance(expected, bytes):
        assert path.read_bytes() == expected, (
            f"{path.name} differs from the generator's output -- run "
            "`python scripts/build_yam_assets.py`"
        )
        return
    on_disk = path.read_text()
    assert on_disk.startswith(GENERATED_HEADER)
    assert on_disk == expected, (
        f"{path.name} differs from the generator's output -- "
        "run `python scripts/build_yam_assets.py`, and never edit it by hand"
    )


def test_the_arm_has_no_gripper_left_in_it():
    text = build_arm((UPSTREAM / "yam.xml").read_text())
    arm = ET.fromstring(text.removeprefix(GENERATED_HEADER))
    bodies = {b.get("name") for b in arm.iter("body")}
    assert not bodies & {"link_left_finger", "link_right_finger"}
    assert arm.find("equality") is None
    assert "gripper" not in {a.get("name") for a in arm.iter("position")}
    assert "grasp_site" not in {s.get("name") for s in arm.iter("site")}
    assert arm.find(".//body[@name='link_6']") is not None, "the flange must remain"


def test_the_gripper_is_namespaced_so_it_can_attach_to_any_arm():
    ee = ET.fromstring(
        build_linear_gripper((UPSTREAM / "yam.xml").read_text()).removeprefix(GENERATED_HEADER)
    )
    for d in ee.iter("default"):
        if d.get("class"):
            assert d.get("class").startswith(EE_PREFIX), d.get("class")
    for m in ee.iter("material"):
        assert m.get("name").startswith(EE_PREFIX), m.get("name")
    for m in ee.find("asset").iter("mesh"):
        assert m.get("name").startswith(EE_PREFIX), m.get("name")


def test_generator_refuses_an_upstream_it_does_not_recognise():
    """If Menagerie reshapes link_6, splitting by the old rules would move the
    wrong parts. Better to stop."""
    upstream = (UPSTREAM / "yam.xml").read_text().replace('name="grasp_site"', 'name="moved"')
    with pytest.raises(ValueError, match="Refusing to split"):
        build_arm(upstream)
    with pytest.raises(ValueError, match="link_6"):
        build_arm('<mujoco model="x"><compiler/><worldbody/></mujoco>')


def test_every_glass_has_a_unique_name():
    names = [g.name for g in DEFAULT_GLASSES]
    assert len(names) == len(set(names)) == 2


def test_look_at_rejects_a_degenerate_up_vector():
    with pytest.raises(ValueError, match="parallel"):
        look_at_xyaxes((0, 0, 0), (0, 0, 1), up=(0, 0, 1))


def test_vendored_license_is_present():
    assert "MIT License" in (UPSTREAM / "LICENSE").read_text()
