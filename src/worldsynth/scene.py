"""USD scene for one randomized clip: workcell, lights, parts, clutter and camera. Needs a running Isaac Sim app."""

from __future__ import annotations

import numpy as np
from isaaclab.sim.utils.semantics import add_labels
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade

from . import scene_info
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
    """Static parts of the scene plus per-clip rebuilding of parts, clutter, materials and lights.

    The static part is either the built-in workcell (table, floor, walls) or a predesigned USD scene
    (``scene.scene_usd``). Either way it provides the placement area, the surface height and the bounds for clutter.
    """

    def __init__(self, stage: Usd.Stage, cfg: WorldSynthCfg):
        self.stage = stage
        self.cfg = cfg
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        world = UsdGeom.Xform.Define(stage, "/World").GetPrim()
        stage.SetDefaultPrim(world)
        self.surfaces: dict[str, Usd.Prim] = {}
        """Randomized surfaces (table, floor, walls); empty with a USD scene, whose materials are kept."""
        self.scene_labels: dict[str, int] = {}
        self.scene_camera: tuple[np.ndarray, float, float] | None = None
        """World transform, focal length and horizontal aperture of the chosen scene camera."""
        self.horizontal_aperture = HORIZONTAL_APERTURE
        if cfg.scene.scene_usd:
            self._load_scene()
        else:
            self.table_z = float(cfg.scene.table.size[2])
            self.placement_area = cfg.scene.placement_area
            sx, sy, _ = cfg.scene.table.size
            self.bounds = ((-sx / 2 + 0.05, sx / 2 - 0.05), (-sy / 2 + 0.05, sy / 2 - 0.05))
            self._build_workcell()
        self.lights_enabled = not (cfg.scene.scene_usd and cfg.scene.scene_lights_only)
        if self.lights_enabled:
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

    def _load_scene(self):
        sc = self.cfg.scene
        if not sc.scene_surface:
            raise ValueError("scene.scene_surface is required with scene.scene_usd (the prim to place parts on)")
        info = scene_info.inspect(sc.scene_usd)
        root = UsdGeom.Xform.Define(self.stage, scene_info.SCENE_ROOT).GetPrim()
        set_transform(root, scene_info.correction(info.meters_per_unit, info.up_axis))
        root.GetReferences().AddReference(sc.scene_usd, info.root_prim)

        surface = self.stage.GetPrimAtPath(scene_info.stage_path(sc.scene_surface, info.root_prim))
        if not surface.IsValid():
            raise ValueError(f"scene_surface {sc.scene_surface} not found in {sc.scene_usd}")
        bb = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
        rng = bb.ComputeWorldBound(surface).ComputeAlignedRange()
        lo, hi = np.array(rng.GetMin()), np.array(rng.GetMax())
        m = sc.surface_margin
        if hi[0] - lo[0] <= 2 * m or hi[1] - lo[1] <= 2 * m:
            raise ValueError(f"scene_surface {sc.scene_surface} is smaller than twice surface_margin ({m} m)")
        self.table_z = float(hi[2])
        self.placement_area = ((lo[0] + m, hi[0] - m), (lo[1] + m, hi[1] - m))
        self.bounds = ((lo[0] + 0.005, hi[0] - 0.005), (lo[1] + 0.005, hi[1] - 0.005))

        # label the scene's top-level objects so the full-scene segmentation (Cosmos seg control) covers them
        groups = [p for p in root.GetChildren() if not p.IsA(UsdGeom.Camera) and not p.HasAPI(UsdLux.LightAPI)]
        if len(groups) == 1 and groups[0].GetChildren():
            groups = list(groups[0].GetChildren())
        for i, prim in enumerate(groups):
            add_labels(prim, [f"scene_{prim.GetName()}"], instance_name="class")
            self.scene_labels[str(prim.GetPath())] = 1000 + i

        if sc.scene_camera:
            cam_prim = self.stage.GetPrimAtPath(scene_info.stage_path(sc.scene_camera, info.root_prim))
            if not cam_prim.IsA(UsdGeom.Camera):
                raise ValueError(f"scene_camera {sc.scene_camera} is not a camera in {sc.scene_usd}")
            cam = UsdGeom.Camera(cam_prim)
            self.scene_camera = (
                scene_info.orthonormal(world_transform(cam_prim)),
                float(cam.GetFocalLengthAttr().Get() or 50.0),
                float(cam.GetHorizontalApertureAttr().Get() or HORIZONTAL_APERTURE),
            )
            self.horizontal_aperture = self.scene_camera[2]

    def _build_lights(self):
        self.dome = UsdLux.DomeLight.Define(self.stage, "/World/Lights/Dome")
        self.dome.CreateEnableColorTemperatureAttr(True)

    def _build_camera(self):
        cam = UsdGeom.Camera.Define(self.stage, CAMERA_PATH)
        w, h = self.cfg.dataset.resolution
        # square pixels: the vertical aperture follows the image aspect ratio
        cam.CreateHorizontalApertureAttr(self.horizontal_aperture)
        cam.CreateVerticalApertureAttr(self.horizontal_aperture * h / w)
        cam.CreateClippingRangeAttr(Gf.Vec2f(0.01, 50.0))
        cam.CreateFocusDistanceAttr(1.0)
        cam.CreateFStopAttr(0.0)  # no depth of field
        self.camera = cam

    # --- per clip ----------------------------------------------------------------------------------------------------

    def background_ids(self) -> dict[str, int]:
        """Segmentation ids of the labelled static prims (workcell surfaces or the scene's objects)."""
        ids = {str(p.GetPath()): 1000 + i for i, p in enumerate(self.surfaces.values())}
        ids.update(self.scene_labels)
        return ids

    def placement_center(self) -> np.ndarray:
        (x0, x1), (y0, y1) = self.placement_area
        return np.array([(x0 + x1) / 2, (y0 + y1) / 2, self.table_z])

    def apply_lights(self, lights: dict):
        if not self.lights_enabled:
            return
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
