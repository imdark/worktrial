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
| `upstream/` | vendored | arm + stock gripper MJCF, meshes, license, upstream README |
| `yam_arm.xml` | **generated** | `upstream/yam.xml` with the stock gripper removed: the arm, ending at the `link_6` flange |
| `../end_effectors/yam_linear/yam_linear.xml` | **generated** | the stock gripper as an attachable part (namespaced `yam_linear_`), plus a D405-style camera on top |
| `../scenes/glasses_table/glasses.xml` | **generated** | two water glasses |

Menagerie ships arm and gripper as one file; the generator splits it so
end effectors can be swapped. `tests/test_composition.py` proves the split is
lossless. The one inseparable piece: `link_6`'s mesh fuses the wrist motor with
the stock gripper's mounting plate, so that plate stays drawn on the arm.

Regenerate the generated files with `python scripts/build_yam_assets.py`.
`tests/test_yam_assets.py` fails if they drift from what the generator produces.
