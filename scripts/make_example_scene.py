"""Write a small example machine-tending scene USD to try the scene options and the GUI.

    python scripts/make_example_scene.py assets/scenes/example_cell.usda

The scene is in centimeters with y up (as many exported scenes are), so it also exercises the unit and axis
conversion. It has a workbench (place parts on ``/Cell/Workbench/Top``), a small conveyor on the bench, a machine,
walls, a ceiling light and two cameras: ``/Cell/Cameras/BenchCam`` (close, looking down at the bench) and
``/Cell/Cameras/Overview``. ``/Cell/Parts`` holds parts of the scene itself (for ``scene.scene_parts``): two bolts,
a block on the bench and ``Block_02``, which rides along the conveyor (animated over 10 s).
"""

import argparse
from pathlib import Path

import numpy as np
from pxr import Gf, Usd, UsdGeom, UsdLux


def look_at_y_up(eye, target) -> Gf.Matrix4d:
    """USD camera transform (looks along -z, +y up) in a y-up scene."""
    eye, target = np.asarray(eye, float), np.asarray(target, float)
    f = (target - eye) / np.linalg.norm(target - eye)
    x = np.cross(f, [0.0, 1.0, 0.0])
    if np.linalg.norm(x) < 1e-6:  # looking straight up or down
        x = np.array([1.0, 0.0, 0.0])
    x /= np.linalg.norm(x)
    y = np.cross(x, f)
    m = np.eye(4)
    m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = x, y, -f, eye
    return Gf.Matrix4d(m.T.tolist())


def box(stage, path, center, size, color):
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    cube.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    xf = UsdGeom.Xformable(cube)
    xf.AddTranslateOp().Set(Gf.Vec3d(*center))
    xf.AddScaleOp().Set(Gf.Vec3f(*size))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("out", nargs="?", default="assets/scenes/example_cell.usda")
    out = Path(parser.parse_args().out)
    out.parent.mkdir(parents=True, exist_ok=True)

    stage = Usd.Stage.CreateNew(str(out))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(stage, 0.01)
    root = UsdGeom.Xform.Define(stage, "/Cell").GetPrim()
    stage.SetDefaultPrim(root)

    box(stage, "/Cell/Floor", (0, -1, 0), (800, 2, 800), (0.35, 0.36, 0.38))
    box(stage, "/Cell/WallBack", (0, 150, -250), (800, 300, 4), (0.75, 0.76, 0.74))
    box(stage, "/Cell/WallSide", (-300, 150, 0), (4, 300, 800), (0.70, 0.72, 0.70))
    # workbench: top at 92.5 cm
    box(stage, "/Cell/Workbench/Top", (0, 90, 0), (140, 5, 80), (0.22, 0.23, 0.25))
    for i, (x, z) in enumerate([(-65, -35), (65, -35), (-65, 35), (65, 35)]):
        box(stage, f"/Cell/Workbench/Leg_{i}", (x, 44, z), (5, 88, 5), (0.5, 0.5, 0.52))
    # conveyor on the back of the bench: belt top at 94.5 cm
    box(stage, "/Cell/Conveyor/Frame", (0, 93.0, -25), (120, 1, 16), (0.45, 0.46, 0.48))
    box(stage, "/Cell/Conveyor/Belt", (0, 94.0, -25), (116, 1, 12), (0.08, 0.08, 0.09))
    # parts of the scene itself
    for name, axis, center in [("Bolt_01", "Y", (-22, 95.5, 12)), ("Bolt_02", "X", (-5, 93.7, 18))]:
        bolt = UsdGeom.Cylinder.Define(stage, f"/Cell/Parts/{name}")
        bolt.CreateRadiusAttr(1.2)
        bolt.CreateHeightAttr(6.0)
        bolt.CreateAxisAttr(axis)
        bolt.CreateDisplayColorAttr([Gf.Vec3f(0.7, 0.7, 0.72)])
        UsdGeom.Xformable(bolt).AddTranslateOp().Set(Gf.Vec3d(*center))
    for name, center in [("Block_01", (18, 94.5, 10)), ("Block_02", (-50, 96.5, -25))]:
        block = UsdGeom.Cube.Define(stage, f"/Cell/Parts/{name}")
        block.CreateSizeAttr(4.0)
        block.CreateDisplayColorAttr([Gf.Vec3f(0.85, 0.45, 0.1)])
        op = UsdGeom.Xformable(block).AddTranslateOp()
        if name == "Block_02":  # rides the conveyor from x = -50 to 50 cm in 10 s
            op.Set(Gf.Vec3d(-50, 96.5, -25), 0)
            op.Set(Gf.Vec3d(50, 96.5, -25), 240)
        else:
            op.Set(Gf.Vec3d(*center))
    stage.SetStartTimeCode(0)
    stage.SetEndTimeCode(240)
    stage.SetTimeCodesPerSecond(24)

    # machine next to the bench
    box(stage, "/Cell/Machine/Body", (-160, 100, -60), (120, 200, 100), (0.82, 0.83, 0.80))
    box(stage, "/Cell/Machine/Door", (-160, 110, -9), (70, 80, 2), (0.25, 0.35, 0.45))

    light = UsdLux.RectLight.Define(stage, "/Cell/Lights/Ceiling")
    light.CreateWidthAttr(120)
    light.CreateHeightAttr(60)
    light.CreateIntensityAttr(15000)
    UsdGeom.Xformable(light).AddTransformOp().Set(look_at_y_up((0, 260, 0), (0, 0, 0)))
    dome = UsdLux.DomeLight.Define(stage, "/Cell/Lights/Ambient")
    dome.CreateIntensityAttr(400)

    for name, eye, target, focal in [
        ("BenchCam", (0, 150, 55), (0, 92.5, 0), 24.0),
        ("Overview", (180, 220, 250), (0, 92.5, 0), 18.0),
    ]:
        cam = UsdGeom.Camera.Define(stage, f"/Cell/Cameras/{name}")
        cam.CreateFocalLengthAttr(focal)
        cam.CreateHorizontalApertureAttr(20.955)
        cam.CreateClippingRangeAttr(Gf.Vec2f(1, 10000))
        UsdGeom.Xformable(cam).AddTransformOp().Set(look_at_y_up(eye, target))
    stage.GetRootLayer().Save()
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
