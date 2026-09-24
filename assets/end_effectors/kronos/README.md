# Kronos end effector (generated, not in the repository)

The production gripper's model is derived from proprietary CAD, so only this
file is tracked. Everything else in this directory is generated:

| file                 | from                          | by                                  |
|----------------------|-------------------------------|-------------------------------------|
| `meshes/*.stl`       | `Master Assembly.step`        | `scripts/build_kronos_assets.py`    |
| `kronos_extract.json`| CAD + slicer masses           | `scripts/build_kronos_assets.py`    |
| `kronos.xml`         | `kronos_extract.json`         | `scripts/build_kronos_model.py`     |

To build it, put `Hardware Archive.zip` in `assets/` (it is gitignored) and run:

```bash
uv pip install gmsh          # GPL; build-time only, never imported by teleop_sim
python scripts/build_kronos_assets.py
python scripts/build_kronos_model.py
```

Until then `yam_kronos` cannot be loaded and the Kronos tests skip.
