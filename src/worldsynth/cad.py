"""Find CAD files in a folder, convert them to USD and read their geometry.

Conversion needs a running Isaac Sim app (``omni.kit.asset_converter`` for meshes, the HOOPS CAD converter for B-rep
formats). :func:`discover` and :func:`load_part` only need ``pxr``.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger("worldsynth")

USD_EXTS = {".usd", ".usda", ".usdc", ".usdz"}
MESH_EXTS = {".obj", ".stl", ".fbx", ".gltf", ".glb"}
BREP_EXTS = {
    ".step", ".stp", ".iges", ".igs", ".jt", ".sldprt", ".sldasm", ".x_t", ".x_b", ".catpart", ".catproduct",
    ".prt", ".asm", ".ipt", ".iam", ".sat", ".3dxml", ".3dm",
}  # fmt: skip
UNIT_SCALE = {"m": 1.0, "cm": 0.01, "mm": 0.001, "inch": 0.0254}
MAX_PART_EXTENT = 2.0
"""Parts larger than this [m] under ``units: auto`` are assumed to be modelled in millimeters."""


@dataclass
class CadFile:
    path: Path
    category: str


@dataclass
class Part:
    """A converted CAD part ready to be referenced into the scene."""

    name: str
    category: str
    source: Path
    usd_path: Path
    root_prim_path: str
    """Top-level prim of the USD file that the scene references."""
    scale: float
    """Factor from the USD file's units to meters."""
    vertices: np.ndarray
    """Vertices in the part frame [m], shape [N, 3]. The part frame is the CAD file's frame scaled to meters, so
    pose labels refer to the same origin and axes as the CAD model."""
    faces: np.ndarray
    scene_prim: str | None = None
    """Stage path of the prim when the part comes from the USD scene (``scene.scene_parts``) instead of a file."""
    bbox_min: np.ndarray = field(init=False)
    bbox_max: np.ndarray = field(init=False)

    def __post_init__(self):
        self.bbox_min = self.vertices.min(axis=0)
        self.bbox_max = self.vertices.max(axis=0)

    @property
    def extent(self) -> np.ndarray:
        return self.bbox_max - self.bbox_min


def discover(cad_dir: str | Path, class_from: str = "file") -> list[CadFile]:
    """All supported CAD/mesh/USD files below ``cad_dir``, sorted by path."""
    cad_dir = Path(cad_dir)
    if not cad_dir.is_dir():
        raise FileNotFoundError(f"CAD folder not found: {cad_dir}")
    exts = USD_EXTS | MESH_EXTS | BREP_EXTS
    files = []
    for p in sorted(cad_dir.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in exts:
            continue
        if class_from == "folder":
            category = p.parent.name if p.parent != cad_dir else p.stem
        elif class_from == "file":
            category = p.stem
        else:
            raise ValueError(f"cad.class_from must be 'file' or 'folder', got {class_from!r}")
        files.append(CadFile(path=p, category=category))
    if not files:
        raise FileNotFoundError(f"No CAD files found in {cad_dir} (supported: {sorted(exts)})")
    return files


# ---------------------------------------------------------------------------------------------------------------------
# conversion (needs Isaac Sim)


def convert_all(files: list[CadFile], cache_dir: str | Path, tessellation_lod: int = 2) -> dict[Path, Path]:
    """Convert CAD files to USD in ``cache_dir``; unchanged files are taken from the cache.

    Returns:
        Mapping from source file to USD file. Files that fail to convert are logged and left out.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    result = {}
    for f in files:
        ext = f.path.suffix.lower()
        if ext in USD_EXTS:
            result[f.path] = f.path
            continue
        key = str(f.path.resolve())
        digest = _file_digest(f.path) + f"-lod{tessellation_lod}"
        out = cache_dir / f"{f.path.stem}_{hashlib.sha1(key.encode()).hexdigest()[:8]}{ext.replace('.', '_')}.usd"
        if manifest.get(key) == digest and out.exists():
            result[f.path] = out
            continue
        logger.info("Converting %s -> %s", f.path, out)
        try:
            if ext in MESH_EXTS:
                _run(_convert_mesh(f.path, out))
            else:
                _run(_convert_brep(f.path, out, tessellation_lod))
        except Exception as e:  # keep going: one broken file must not stop a dataset run
            logger.error("Failed to convert %s: %s", f.path, e)
            continue
        manifest[key] = digest
        manifest_path.write_text(json.dumps(manifest, indent=2))
        result[f.path] = out
    return result


def _file_digest(path: Path) -> str:
    st = path.stat()
    return f"{st.st_size}-{int(st.st_mtime)}"


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


async def _convert_mesh(src: Path, dst: Path):
    from isaaclab.sim.utils import enable_extension

    enable_extension("omni.kit.asset_converter")
    import omni.kit.asset_converter as conv

    ctx = conv.AssetConverterContext()
    ctx.ignore_animations = True
    ctx.ignore_camera = True
    ctx.ignore_light = True
    ctx.merge_all_meshes = False
    ctx.use_meter_as_world_unit = True
    ctx.baking_scales = True
    task = conv.get_instance().create_converter_task(str(src), str(dst), None, ctx)
    if not await task.wait_until_finished():
        raise RuntimeError(task.get_error_message())


async def _convert_brep(src: Path, dst: Path, tessellation_lod: int):
    from isaaclab.sim.utils import enable_extension

    enable_extension("omni.kit.converter.hoops_core")
    import omni.kit.converter.hoops_core as hoops

    if not hoops.is_format_supported(str(src)):
        raise RuntimeError(f"format not supported by the HOOPS converter: {src.suffix}")
    args = {"tessLOD": str(tessellation_lod), "instancing": "false", "convertHidden": "false"}
    out, status = await hoops.get_instance().create_converter_task(str(src), str(dst), args)
    if not out:
        raise RuntimeError(f"HOOPS error {status}")


# ---------------------------------------------------------------------------------------------------------------------
# geometry


def load_part(cad: CadFile, usd_path: Path, units: str = "auto") -> Part:
    """Read the meshes of a converted USD file and resolve its scale to meters."""
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(str(usd_path))
    root = stage.GetDefaultPrim()
    if not root:
        # the scene references a top-level prim, so pick one when the file has no default prim
        root = next((p for p in stage.GetPseudoRoot().GetChildren() if p.IsA(UsdGeom.Imageable)), None)
        if root is None:
            raise RuntimeError(f"No top-level prim in {usd_path}")
    xf_cache = UsdGeom.XformCache()
    verts, faces, offset = [], [], 0
    for prim in Usd.PrimRange(root):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        img = UsdGeom.Imageable(prim)
        if img.ComputeVisibility() == UsdGeom.Tokens.invisible or img.ComputePurpose() in ("guide", "proxy"):
            continue
        mesh = UsdGeom.Mesh(prim)
        pts = np.asarray(mesh.GetPointsAttr().Get() or [], dtype=np.float64)
        counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get() or [], dtype=np.int64)
        idx = np.asarray(mesh.GetFaceVertexIndicesAttr().Get() or [], dtype=np.int64)
        if pts.size == 0 or counts.size == 0:
            continue
        # the root is top-level, so its world transform is what the reference composes under our part prim
        t = _mat(xf_cache.GetLocalToWorldTransform(prim))
        verts.append(pts @ t[:3, :3].T + t[:3, 3])
        faces.append(_triangulate(counts, idx) + offset)
        offset += len(pts)
    if not verts:
        raise RuntimeError(f"No visible meshes in {usd_path}")
    v = np.concatenate(verts)
    f = np.concatenate(faces)

    meters_per_unit = UsdGeom.GetStageMetersPerUnit(stage)
    if units == "auto":
        scale = meters_per_unit
        if np.max(v.max(0) - v.min(0)) * scale > MAX_PART_EXTENT:
            logger.warning(
                "%s is %.1f units across; assuming millimeters. Set cad.units to override.",
                cad.path.name,
                np.max(v.max(0) - v.min(0)),
            )
            scale = 0.001
    elif units in UNIT_SCALE:
        scale = UNIT_SCALE[units]
    else:
        raise ValueError(f"cad.units must be auto or one of {list(UNIT_SCALE)}, got {units!r}")
    return Part(
        name=cad.path.stem,
        category=cad.category,
        source=cad.path,
        usd_path=Path(usd_path),
        root_prim_path=str(root.GetPath()),
        scale=float(scale),
        vertices=v * scale,
        faces=f,
    )


def part_root_transform(part: Part) -> np.ndarray:
    """Transform from the USD file's frame to the part frame (unit scale only; the stage up axis is ignored because
    the converters keep the source coordinates)."""
    return np.diag([part.scale, part.scale, part.scale, 1.0])


def _triangulate(counts: np.ndarray, idx: np.ndarray) -> np.ndarray:
    if np.all(counts == 3):
        return idx.reshape(-1, 3)
    tris = []
    start = 0
    for c in counts:
        for k in range(1, c - 1):
            tris.append((idx[start], idx[start + k], idx[start + k + 1]))
        start += c
    return np.asarray(tris, dtype=np.int64).reshape(-1, 3)


def _mat(m) -> np.ndarray:
    """Gf.Matrix4d (row-vector convention) to a column-vector numpy transform."""
    return np.array(m, dtype=np.float64).T
