"""Write a few simple machined-part meshes to try the pipeline without real CAD files.

    python scripts/make_example_cad.py cad_examples

Creates STL files in millimeters (like most CAD exports) and an OBJ in meters, one class per file.
"""

import argparse
from pathlib import Path

import trimesh


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("out", nargs="?", default="cad_examples")
    out = Path(parser.parse_args().out)
    out.mkdir(parents=True, exist_ok=True)

    # L-bracket [mm]
    base = trimesh.creation.box((60, 40, 6))
    base.apply_translation((0, 0, 3))
    wall = trimesh.creation.box((6, 40, 50))
    wall.apply_translation((-27, 0, 25))
    trimesh.util.concatenate([base, wall]).export(out / "bracket.stl")

    # flanged shaft [mm]
    flange = trimesh.creation.cylinder(radius=40, height=12)
    shaft = trimesh.creation.cylinder(radius=15, height=80)
    shaft.apply_translation((0, 0, 46))
    trimesh.util.concatenate([flange, shaft]).export(out / "flange_shaft.stl")

    # hex nut blank [mm]
    trimesh.creation.cylinder(radius=30, height=20, sections=6).export(out / "hex_blank.stl")

    # washer [m]
    trimesh.creation.annulus(r_min=0.01, r_max=0.035, height=0.015).export(out / "washer.obj")
    print(f"wrote example parts to {out}")


if __name__ == "__main__":
    main()
