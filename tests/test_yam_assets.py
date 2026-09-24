"""The vendored YAM model stays pristine and the generated files stay current."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from teleop_sim.envs.yam_assets import (
    DEFAULT_GLASSES,
    GENERATED_HEADER,
    WristCamera,
    build_follower,
    build_glasses,
    look_at_xyaxes,
)
from tests.conftest import REPO_ROOT

YAM = REPO_ROOT / "assets" / "i2rt_yam"
UPSTREAM = YAM / "upstream"


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


@pytest.mark.parametrize(
    "path, build",
    [
        ("yam_follower.xml", lambda: build_follower((UPSTREAM / "yam.xml").read_text())),
        ("glasses.xml", build_glasses),
    ],
)
def test_generated_files_are_current(path, build):
    on_disk = (YAM / path).read_text()
    assert on_disk.startswith(GENERATED_HEADER)
    assert on_disk == build(), (
        f"{path} differs from the generator's output -- "
        "run `python scripts/build_yam_assets.py`, and never edit it by hand"
    )


def test_follower_changes_nothing_but_what_it_documents():
    upstream = (UPSTREAM / "yam.xml").read_text()
    follower = build_follower(upstream).removeprefix(GENERATED_HEADER)
    camera_block = WristCamera().mjcf("                  ")
    restored = (
        follower.replace(camera_block, "")
        .replace('meshdir="upstream/assets"', 'meshdir="assets"')
        .replace('model="yam_follower"', 'model="yam_v0"')
    )
    assert restored == upstream


def test_generator_refuses_an_upstream_it_does_not_recognise():
    with pytest.raises(ValueError, match="grasp_site"):
        build_follower('<mujoco model="yam_v0">\n  <compiler meshdir="assets"/>\n</mujoco>\n')


def test_every_glass_has_a_unique_name():
    names = [g.name for g in DEFAULT_GLASSES]
    assert len(names) == len(set(names)) == 2


def test_look_at_rejects_a_degenerate_up_vector():
    with pytest.raises(ValueError, match="parallel"):
        look_at_xyaxes((0, 0, 0), (0, 0, 1), up=(0, 0, 1))


def test_vendored_license_is_present():
    assert "MIT License" in (UPSTREAM / "LICENSE").read_text()


def test_scene_model_files_exist():
    for name in ("glasses_scene.xml", "yam_follower.xml", "glasses.xml", "VENDOR.md"):
        assert (YAM / name).is_file(), name
    assert Path(UPSTREAM / "assets").is_dir()
