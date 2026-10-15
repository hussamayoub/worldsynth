"""Read a predesigned USD scene: units, up axis, cameras and candidate placement surfaces. Needs only ``pxr``.

All positions and sizes are reported in WorldSynth's frame: meters, z up. The scene is referenced under
``/World/Scene``; :func:`stage_path` maps a prim path of the scene file to its path there.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

SCENE_ROOT = "/World/Scene"
MIN_SURFACE_SIZE = 0.1
"""Smallest x and y extent [m] for a prim to be listed as a placement surface."""


@dataclass
class CameraInfo:
    path: str
    position: list[float]
    """[m], z up"""
    focal_length: float
    horizontal_aperture: float


@dataclass
class SurfaceInfo:
    path: str
    type_name: str
    top_z: float
    """Height of the top of the bounding box [m]."""
    size: list[float]
    """x and y extent of the bounding box [m]."""


@dataclass
class SceneInfo:
    path: str
    root_prim: str
    """Prim of the file that is referenced (the default prim, or the first root prim)."""
    meters_per_unit: float
    up_axis: str
    cameras: list[CameraInfo] = field(default_factory=list)
    surfaces: list[SurfaceInfo] = field(default_factory=list)


def correction(meters_per_unit: float, up_axis: str) -> np.ndarray:
    """Transform from the scene file's frame to meters and z up (column-vector convention)."""
    t = np.diag([meters_per_unit, meters_per_unit, meters_per_unit, 1.0])
    if up_axis.upper() == "Y":
        t = np.array([[1.0, 0, 0, 0], [0, 0, -1.0, 0], [0, 1.0, 0, 0], [0, 0, 0, 1.0]]) @ t
    return t


def stage_path(file_path: str, root_prim: str) -> str:
    """Path of a scene-file prim once the file's ``root_prim`` is referenced at :data:`SCENE_ROOT`."""
    if file_path != root_prim and not file_path.startswith(root_prim + "/"):
        raise ValueError(f"{file_path} is not under the scene's root prim {root_prim}")
    return SCENE_ROOT + file_path[len(root_prim) :]


def orthonormal(t: np.ndarray) -> np.ndarray:
    """Remove scale from a transform's rotation part (scene unit scaling ends up in camera transforms)."""
    out = t.copy()
    u, _, vt = np.linalg.svd(t[:3, :3])
    out[:3, :3] = u @ vt
    return out


def box_corners_world(lo: np.ndarray, hi: np.ndarray, t: np.ndarray) -> np.ndarray:
    """The 8 corners of an axis-aligned box transformed by ``t``, shape [8, 3]."""
    idx = np.array([[i >> 2 & 1, i >> 1 & 1, i & 1] for i in range(8)])
    corners = np.where(idx, hi, lo)
    return corners @ t[:3, :3].T + t[:3, 3]


def inspect(path: str) -> SceneInfo:
    """Cameras and candidate placement surfaces of a USD scene, sorted with the largest surfaces first."""
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise RuntimeError(f"Could not open {path}")
    root = stage.GetDefaultPrim() or next(iter(stage.GetPseudoRoot().GetChildren()), None)
    if root is None:
        raise RuntimeError(f"{path} has no prims")
    mpu = UsdGeom.GetStageMetersPerUnit(stage)
    up = str(UsdGeom.GetStageUpAxis(stage))
    fix = correction(mpu, up)
    info = SceneInfo(path=str(path), root_prim=str(root.GetPath()), meters_per_unit=mpu, up_axis=up)
    xf_cache = UsdGeom.XformCache()
    bb_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    for prim in Usd.PrimRange(root):
        if prim.IsA(UsdGeom.Camera):
            cam = UsdGeom.Camera(prim)
            t = fix @ np.array(xf_cache.GetLocalToWorldTransform(prim), dtype=float).T
            info.cameras.append(
                CameraInfo(
                    path=str(prim.GetPath()),
                    position=np.round(t[:3, 3], 3).tolist(),
                    focal_length=float(cam.GetFocalLengthAttr().Get() or 50.0),
                    horizontal_aperture=float(cam.GetHorizontalApertureAttr().Get() or 20.955),
                )
            )
        elif (prim.IsA(UsdGeom.Imageable) or not prim.GetTypeName()) and prim != root:
            # typeless prims are implicit groups (e.g. a table made of several boxes)
            rng = bb_cache.ComputeWorldBound(prim).ComputeAlignedRange()
            if rng.IsEmpty():
                continue
            pts = box_corners_world(np.array(rng.GetMin()), np.array(rng.GetMax()), fix)
            size = pts[:, :2].max(0) - pts[:, :2].min(0)
            if min(size) < MIN_SURFACE_SIZE:
                continue
            info.surfaces.append(
                SurfaceInfo(
                    path=str(prim.GetPath()),
                    type_name=str(prim.GetTypeName()),
                    top_z=round(float(pts[:, 2].max()), 4),
                    size=np.round(size, 3).tolist(),
                )
            )
    info.surfaces.sort(key=lambda s: -s.size[0] * s.size[1])
    return info
