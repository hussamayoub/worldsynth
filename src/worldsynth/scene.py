"""USD scene for one randomized clip: workcell, lights, parts, clutter and camera. Needs a running Isaac Sim app."""

from __future__ import annotations

import numpy as np
from isaaclab.sim.utils.semantics import add_labels
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade

from .cad import Part, part_root_transform
from .config import WorldSynthCfg

HORIZONTAL_APERTURE = 20.955  # [mm], USD default
PARTS_ROOT = "/World/Parts"
CLUTTER_ROOT = "/World/Clutter"
CAMERA_PATH = "/World/Camera"


def set_transform(prim: Usd.Prim, t: np.ndarray):
    xf = UsdGeom.Xformable(prim)
    xf.ClearXformOpOrder()
    xf.AddTransformOp().Set(Gf.Matrix4d(np.asarray(t, dtype=float).T.tolist()))


def world_transform(prim: Usd.Prim) -> np.ndarray:
    m = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    return np.array(m, dtype=float).T


def make_material(stage: Usd.Stage, path: str, params: dict) -> UsdShade.Material:
    mat = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*params["diffuseColor"]))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(params["roughness"]))
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(float(params["metallic"]))
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return mat


def bind_material(prim: Usd.Prim, mat: UsdShade.Material, override_children: bool = False):
    strength = UsdShade.Tokens.strongerThanDescendants if override_children else UsdShade.Tokens.weakerThanDescendants
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat, bindingStrength=strength)


class WorkcellScene:
    """Static parts of the scene plus per-clip rebuilding of parts, clutter, materials and lights."""

    def __init__(self, stage: Usd.Stage, cfg: WorldSynthCfg):
        self.stage = stage
        self.cfg = cfg
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        world = UsdGeom.Xform.Define(stage, "/World").GetPrim()
        stage.SetDefaultPrim(world)
        self.table_z = float(cfg.scene.table.size[2])
        self._build_workcell()
        self._build_lights()
        self._build_camera()

    # --- static ------------------------------------------------------------------------------------------------------

    def _box(self, path: str, center, size) -> Usd.Prim:
        cube = UsdGeom.Cube.Define(self.stage, path)
        cube.CreateSizeAttr(1.0)
        t = np.diag([*size, 1.0])
        t[:3, 3] = center
        set_transform(cube.GetPrim(), t)
        return cube.GetPrim()

    def _build_workcell(self):
        sx, sy, h = self.cfg.scene.table.size
        self.surfaces = {
            "floor": self._box("/World/Workcell/Floor", (0, 0, -0.01), (12, 12, 0.02)),
            "table": self._box("/World/Workcell/Table", (0, 0, h - 0.02), (sx, sy, 0.04)),
            "table_base": self._box(
                "/World/Workcell/TableBase", (0, 0, (h - 0.04) / 2), (sx * 0.9, sy * 0.9, h - 0.04)
            ),
        }
        if self.cfg.scene.walls:
            self.surfaces["wall_back"] = self._box("/World/Workcell/WallBack", (2.5, 0, 1.5), (0.05, 6, 3))
            self.surfaces["wall_left"] = self._box("/World/Workcell/WallLeft", (0, 2.5, 1.5), (6, 0.05, 3))
            self.surfaces["wall_right"] = self._box("/World/Workcell/WallRight", (0, -2.5, 1.5), (6, 0.05, 3))
        # background prims get labels too, so the full-scene segmentation (Cosmos seg control) covers them
        for name, prim in self.surfaces.items():
            add_labels(prim, [f"scene_{name}"], instance_name="class")
        if self.cfg.scene.environment_usd:
            env = self.stage.DefinePrim("/World/Environment", "Xform")
            env.GetReferences().AddReference(self.cfg.scene.environment_usd)
            add_labels(env, ["scene_environment"], instance_name="class")
            x, y, z, yaw = self.cfg.scene.environment_pose
            t = np.eye(4)
            c, s = np.cos(np.deg2rad(yaw)), np.sin(np.deg2rad(yaw))
            t[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
            t[:3, 3] = (x, y, z)
            set_transform(env, t)

    def _build_lights(self):
        self.dome = UsdLux.DomeLight.Define(self.stage, "/World/Lights/Dome")
        self.dome.CreateEnableColorTemperatureAttr(True)

    def _build_camera(self):
        cam = UsdGeom.Camera.Define(self.stage, CAMERA_PATH)
        w, h = self.cfg.dataset.resolution
        cam.CreateHorizontalApertureAttr(HORIZONTAL_APERTURE)
        cam.CreateVerticalApertureAttr(HORIZONTAL_APERTURE * h / w)
        cam.CreateClippingRangeAttr(Gf.Vec2f(0.01, 50.0))
        cam.CreateFocusDistanceAttr(1.0)
        cam.CreateFStopAttr(0.0)  # no depth of field
        self.camera = cam

    # --- per clip ----------------------------------------------------------------------------------------------------

    def background_ids(self) -> dict[str, int]:
        """Segmentation ids of the labelled static prims (surfaces and environment)."""
        ids = {str(p.GetPath()): 1000 + i for i, p in enumerate(self.surfaces.values())}
        if self.cfg.scene.environment_usd:
            ids["/World/Environment"] = 999
        return ids

    def placement_center(self) -> np.ndarray:
        (x0, x1), (y0, y1) = self.cfg.scene.placement_area
        return np.array([(x0 + x1) / 2, (y0 + y1) / 2, self.table_z])

    def apply_lights(self, lights: dict):
        d = lights["dome"]
        self.dome.CreateIntensityAttr(d["intensity"])
        self.dome.CreateColorTemperatureAttr(d["color_temperature"])
        set_transform(self.dome.GetPrim(), _rot_z4(np.deg2rad(d["rotation_deg"])))
        self.stage.RemovePrim("/World/Lights/Ceiling")
        for i, light in enumerate(lights["rect_lights"]):
            rect = UsdLux.RectLight.Define(self.stage, f"/World/Lights/Ceiling/Light_{i:02d}")
            rect.CreateWidthAttr(light["size"][0])
            rect.CreateHeightAttr(light["size"][1])
            rect.CreateIntensityAttr(light["intensity"])
            rect.CreateEnableColorTemperatureAttr(True)
            rect.CreateColorTemperatureAttr(light["color_temperature"])
            t = np.eye(4)
            t[:3, 3] = light["position"]  # rect lights emit along local -z, i.e. straight down
            set_transform(rect.GetPrim(), t)

    def apply_surfaces(self, materials: dict[str, dict]):
        for name, params in materials.items():
            mat = make_material(self.stage, f"/World/Looks/{name}", params)
            bind_material(self.surfaces[name], mat)

    def clear_clip(self):
        self.stage.RemovePrim(PARTS_ROOT)
        self.stage.RemovePrim(CLUTTER_ROOT)

    def add_part(self, index: int, part: Part, pose: np.ndarray, material: dict | None) -> str:
        """Reference a part under ``/World/Parts/part_###`` with its pose and semantic class; return the prim path."""
        path = f"{PARTS_ROOT}/part_{index:03d}"
        root = UsdGeom.Xform.Define(self.stage, path).GetPrim()
        set_transform(root, pose)
        geom = UsdGeom.Xform.Define(self.stage, f"{path}/geom").GetPrim()
        set_transform(geom, part_root_transform(part))
        asset = self.stage.DefinePrim(f"{path}/geom/asset", "Xform")
        asset.GetReferences().AddReference(str(part.usd_path), part.root_prim_path)
        add_labels(root, [part.category], instance_name="class")
        if material is not None:
            bind_material(root, make_material(self.stage, f"{path}/Looks/Material", material), override_children=True)
        return path

    def add_clutter(self, index: int, item: dict, pose: np.ndarray, material: dict) -> str:
        path = f"{CLUTTER_ROOT}/item_{index:03d}"
        shape = item["shape"]
        if shape == "cube":
            prim = UsdGeom.Cube.Define(self.stage, path)
            prim.CreateSizeAttr(1.0)
        elif shape == "sphere":
            prim = UsdGeom.Sphere.Define(self.stage, path)
            prim.CreateRadiusAttr(1.0)
        else:
            prim = (UsdGeom.Cylinder if shape == "cylinder" else UsdGeom.Cone).Define(self.stage, path)
            prim.CreateRadiusAttr(0.5)
            prim.CreateHeightAttr(1.0)
        sx, sy, sz = item["scale"]
        if shape == "sphere":
            sx = sy = sz = sx  # scale holds the radius
        t = pose @ np.diag([sx, sy, sz, 1.0])
        set_transform(prim.GetPrim(), t)
        bind_material(prim.GetPrim(), make_material(self.stage, f"{path}_Looks/Material", material))
        add_labels(prim.GetPrim(), ["scene_clutter"], instance_name="class")
        return path

    def set_camera(self, transform: np.ndarray, focal_length: float):
        self.camera.CreateFocalLengthAttr(float(focal_length))
        set_transform(self.camera.GetPrim(), transform)


def clutter_half_height(item: dict) -> float:
    """Distance from a clutter primitive's center to its lowest point [m]."""
    s = item["scale"]
    return s[0] if item["shape"] == "sphere" else s[2] / 2.0


def _rot_z4(angle: float) -> np.ndarray:
    t = np.eye(4)
    c, s = np.cos(angle), np.sin(angle)
    t[:2, :2] = [[c, -s], [s, c]]
    return t
