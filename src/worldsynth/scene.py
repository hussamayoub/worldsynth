"""USD scene for one randomized clip: workcell, lights, parts, clutter and camera. Needs a running Isaac Sim app."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from isaaclab.sim.utils.semantics import add_labels
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade

from . import randomize as R
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


@dataclass
class ScenePart:
    """A labelled part that lives in the USD scene (a child of ``scene.scene_parts``)."""

    path: str
    """Stage path."""
    name: str
    category: str
    vertices: np.ndarray
    """Vertices in the part frame [m]: the prim's own frame without its scale."""
    faces: np.ndarray
    shape: np.ndarray
    """Scale/shear of the prim's world transform (``world = R @ shape``), kept when the part is moved."""
    pose: np.ndarray
    """Part frame to world as authored in the scene."""


def polar(m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split a 3x3 matrix into rotation and the remaining scale/shear: ``m = r @ s``."""
    u, _, vt = np.linalg.svd(m)
    r = u @ vt
    if np.linalg.det(r) < 0:  # mirrored prim: keep a proper rotation, the mirror goes into s
        u[:, -1] *= -1
        r = u @ vt
    return r, r.T @ m


def gprim_mesh(prim: Usd.Prim) -> tuple[np.ndarray, np.ndarray] | None:
    """Vertices and triangles of a mesh or of an implicit USD shape (cube, sphere, cylinder, cone, capsule)."""
    import trimesh

    if prim.IsA(UsdGeom.Mesh):
        from .cad import _triangulate

        mesh = UsdGeom.Mesh(prim)
        pts = np.asarray(mesh.GetPointsAttr().Get() or [], dtype=float)
        counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get() or [], dtype=np.int64)
        idx = np.asarray(mesh.GetFaceVertexIndicesAttr().Get() or [], dtype=np.int64)
        return (pts, _triangulate(counts, idx)) if pts.size and counts.size else None
    if prim.IsA(UsdGeom.Cube):
        m = trimesh.creation.box([UsdGeom.Cube(prim).GetSizeAttr().Get() or 2.0] * 3)
    elif prim.IsA(UsdGeom.Sphere):
        m = trimesh.creation.icosphere(subdivisions=2, radius=UsdGeom.Sphere(prim).GetRadiusAttr().Get() or 1.0)
    elif prim.IsA(UsdGeom.Cylinder) or prim.IsA(UsdGeom.Cone) or prim.IsA(UsdGeom.Capsule):
        g = UsdGeom.Cylinder(prim) if prim.IsA(UsdGeom.Cylinder) else None
        g = g or (UsdGeom.Cone(prim) if prim.IsA(UsdGeom.Cone) else UsdGeom.Capsule(prim))
        r, h = g.GetRadiusAttr().Get() or 1.0, g.GetHeightAttr().Get() or 2.0
        if prim.IsA(UsdGeom.Cone):
            m = trimesh.creation.cone(radius=r, height=h)
            m.apply_translation((0, 0, -h / 2))  # USD cones are centered
        elif prim.IsA(UsdGeom.Capsule):
            m = trimesh.creation.capsule(height=h, radius=r)
        else:
            m = trimesh.creation.cylinder(radius=r, height=h)
        axis = str(g.GetAxisAttr().Get() or "Z")
        if axis == "X":
            m.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
        elif axis == "Y":
            m.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0]))
    else:
        return None
    return np.asarray(m.vertices), np.asarray(m.faces)


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
        self.scene_parts: list[ScenePart] = []
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

        parts_root = None
        if sc.scene_parts:
            parts_root = self.stage.GetPrimAtPath(scene_info.stage_path(sc.scene_parts, info.root_prim))
            if not parts_root.IsValid():
                raise ValueError(f"scene_parts {sc.scene_parts} not found in {sc.scene_usd}")
            self._load_scene_parts(parts_root)

        # label the scene's top-level objects so the full-scene segmentation (Cosmos seg control) covers them;
        # groups that contain the scene parts are split so the parts keep their own labels
        groups = [p for p in root.GetChildren() if scene_info.is_geometry_group(p)]
        if len(groups) == 1 and groups[0].GetChildren():
            groups = list(groups[0].GetChildren())
        labelled = []
        while groups:
            prim = groups.pop(0)
            if not scene_info.is_geometry_group(prim) or (parts_root is not None and prim == parts_root):
                continue
            if parts_root is not None and str(parts_root.GetPath()).startswith(str(prim.GetPath()) + "/"):
                groups.extend(prim.GetChildren())
                continue
            labelled.append(prim)
        for i, prim in enumerate(labelled):
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
            if sc.placement_in_view:
                w, h = self.cfg.dataset.resolution
                (vx0, vx1), (vy0, vy1) = R.view_footprint(
                    self.scene_camera[0], self.scene_camera[1], self.horizontal_aperture, w / h, self.table_z
                )
                (px0, px1), (py0, py1) = self.placement_area
                area = ((max(px0, vx0), min(px1, vx1)), (max(py0, vy0), min(py1, vy1)))
                if area[0][1] - area[0][0] < 0.02 or area[1][1] - area[1][0] < 0.02:
                    raise ValueError(f"scene_camera {sc.scene_camera} does not see scene_surface {sc.scene_surface}")
                self.placement_area = area

    def _load_scene_parts(self, parts_root: Usd.Prim):
        xf = UsdGeom.XformCache()
        for child in parts_root.GetChildren():
            if not scene_info.is_geometry_group(child):
                continue
            world = _mat(xf.GetLocalToWorldTransform(child))
            rot, shape = polar(world[:3, :3])
            pose = np.eye(4)
            pose[:3, :3], pose[:3, 3] = rot, world[:3, 3]
            verts, faces, offset = [], [], 0
            for prim in Usd.PrimRange(child):
                geo = gprim_mesh(prim) if prim.IsA(UsdGeom.Gprim) else None
                if geo is None:
                    continue
                t = _mat(xf.GetLocalToWorldTransform(prim))
                pts_world = geo[0] @ t[:3, :3].T + t[:3, 3]
                verts.append((pts_world - pose[:3, 3]) @ rot)  # into the part frame: R^T (p - t)
                faces.append(geo[1] + offset)
                offset += len(geo[0])
            if not verts:
                continue
            labels = child.GetAttribute("semantics:labels:class").Get() if child.HasAttribute(
                "semantics:labels:class"
            ) else None  # fmt: skip
            name = child.GetName()
            category = str(labels[0]) if labels else scene_info.part_class_name(name)
            add_labels(child, [category], instance_name="class")
            self.scene_parts.append(
                ScenePart(
                    str(child.GetPath()), name, category, np.concatenate(verts), np.concatenate(faces), shape, pose
                )
            )
        if not self.scene_parts:
            raise ValueError(f"No parts with geometry under scene_parts {parts_root.GetPath()}")

    def scene_part(self, path: str) -> ScenePart:
        return next(s for s in self.scene_parts if s.path == path)

    def show_scene_parts(self, paths: set[str]):
        """Show the given scene parts and hide the others."""
        for sp in self.scene_parts:
            img = UsdGeom.Imageable(self.stage.GetPrimAtPath(sp.path))
            img.MakeVisible() if sp.path in paths else img.MakeInvisible()

    def pose_scene_part(self, path: str, pose: np.ndarray | None, material: dict | None) -> str:
        """Move a scene part so its part frame is at ``pose`` (None: leave it, including any animation) and set or
        clear its random material."""
        prim = self.stage.GetPrimAtPath(path)
        sp = self.scene_part(path)
        if pose is not None:
            parent = world_transform(prim.GetParent())
            target = pose.copy()
            target[:3, :3] = pose[:3, :3] @ sp.shape
            set_transform(prim, np.linalg.inv(parent) @ target)
        if material is not None:
            bind_material(prim, make_material(self.stage, f"{path}/WorldSynthLooks/Material", material), True)
        else:
            UsdShade.MaterialBindingAPI(prim).UnbindDirectBinding()
        return path

    def scene_part_pose(self, path: str, time: float | None = None) -> np.ndarray:
        """Current part frame of a scene part [m], at stage time code ``time`` (None: default time)."""
        tc = Usd.TimeCode.Default() if time is None else Usd.TimeCode(time)
        m = np.array(UsdGeom.Xformable(self.stage.GetPrimAtPath(path)).ComputeLocalToWorldTransform(tc)).T
        rot, _ = polar(m[:3, :3])
        out = np.eye(4)
        out[:3, :3], out[:3, 3] = rot, m[:3, 3]
        return out

    def current_time_code(self) -> float:
        """Stage time code of the frame being rendered (the timeline advances while capturing)."""
        import omni.timeline

        return omni.timeline.get_timeline_interface().get_current_time() * self.stage.GetTimeCodesPerSecond()

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


def _mat(m) -> np.ndarray:
    """Gf.Matrix4d (row-vector convention) to a column-vector numpy transform."""
    return np.array(m, dtype=np.float64).T


def clutter_half_height(item: dict) -> float:
    """Distance from a clutter primitive's center to its lowest point [m]."""
    s = item["scale"]
    return s[0] if item["shape"] == "sphere" else s[2] / 2.0


def _rot_z4(angle: float) -> np.ndarray:
    t = np.eye(4)
    c, s = np.cos(angle), np.sin(angle)
    t[:2, :2] = [[c, -s], [s, c]]
    return t
