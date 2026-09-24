# Vendored: I2RT YAM model

| | |
|---|---|
| Source | https://github.com/google-deepmind/mujoco_menagerie/tree/main/i2rt_yam |
| Commit | `c96a32d28fb5da84da38c1da4d749e7a13212855` |
| Fetched | 2026-09-24 (sparse clone of `i2rt_yam/` only, 6.4 MB) |
| License | MIT, © 2025 i2rt robotics — see `upstream/LICENSE` |

`upstream/` is **byte-for-byte unmodified**. Never edit it; everything we add
lives next to it so a future Menagerie release can be dropped in and diffed.

| File | Origin | What |
|---|---|---|
| `upstream/` | vendored | arm MJCF, meshes, license, upstream README |
| `yam_follower.xml` | **generated** | `upstream/yam.xml` + a D405-style wrist camera on top of the gripper |
| `glasses.xml` | **generated** | two water glasses |
| `glasses_scene.xml` | hand-written | table, lighting, overhead camera, timestep; includes the two above |

Regenerate the generated files with `python scripts/build_yam_assets.py`.
`tests/test_yam_assets.py` fails if they drift from what the generator produces.
