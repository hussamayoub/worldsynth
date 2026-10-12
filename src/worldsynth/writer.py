"""Write one rendered frame (images + labels) to disk. Pure NumPy/OpenCV; the renderer output is passed in."""

from __future__ import annotations

import json
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from . import labels as L
from .randomize import USD_TO_CV, matrix_to_quat

MIN_PIXEL_AREA = 16
"""Visible objects smaller than this [px] are flagged ``ignore``."""


@dataclass
class SceneObject:
    """A labelled part instance in the current clip."""

    instance_id: int
    category: str
    class_id: int
    prim_path: str
    part_name: str
    vertices: np.ndarray
    """Part-frame vertices [m] of the mesh used for the amodal silhouette, shape [N, 3]."""
    faces: np.ndarray
    """Triangles; :func:`simplify_for_silhouette` output is enough, it is only used for visibility."""
    bbox_min: np.ndarray
    bbox_max: np.ndarray
    T_world_obj: np.ndarray
    """Part frame (CAD frame in meters) to world."""


def simplify_for_silhouette(vertices: np.ndarray, faces: np.ndarray, cells: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """Vertex-clustering simplification: merge vertices on a grid of ``cells`` steps across the part.

    The silhouette changes by less than one grid step, which is enough to measure visibility and keeps the
    per-frame rasterization cheap for dense CAD tessellations.
    """
    lo = vertices.min(axis=0)
    step = max(float(np.max(vertices.max(axis=0) - lo)) / cells, 1e-9)
    keys = np.floor((vertices - lo) / step).astype(np.int64)
    _, cluster, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    cluster = cluster.reshape(-1)
    merged = np.zeros((len(counts), 3))
    np.add.at(merged, cluster, vertices)
    merged /= counts[:, None]
    f = cluster[faces]
    f = f[(f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])]
    f = np.unique(np.sort(f, axis=1), axis=0)
    return merged, f


def rasterize_mesh(
    k: np.ndarray, t_cam_obj: np.ndarray, vertices: np.ndarray, faces: np.ndarray, width: int, height: int
) -> tuple[int, int, list[int] | None]:
    """Amodal silhouette of a mesh, ignoring all other objects.

    The part of the silhouette outside the image is counted too (up to one image size beyond each border).

    Returns:
        Silhouette area [px] in total, the part of it inside the image, and its box ``[x0, y0, x1, y1]`` clipped to
        the image (``None`` if it does not overlap the image).
    """
    pts = L.transform_points(t_cam_obj, vertices)
    tri = faces[np.all(pts[faces][:, :, 2] > 1e-3, axis=1)]  # drop triangles that reach behind the camera
    if tri.size == 0:
        return 0, 0, None
    uv_all = L.project(k, pts)
    uv = uv_all[np.unique(tri)]
    # canvas = projected bounds, limited to one image size around the image
    x0 = int(np.clip(np.floor(uv[:, 0].min()), -width, 2 * width))
    y0 = int(np.clip(np.floor(uv[:, 1].min()), -height, 2 * height))
    x1 = int(np.clip(np.ceil(uv[:, 0].max()) + 1, -width, 2 * width))
    y1 = int(np.clip(np.ceil(uv[:, 1].max()) + 1, -height, 2 * height))
    if x1 <= x0 or y1 <= y0:
        return 0, 0, None
    canvas = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
    poly = np.round((uv_all[tri] - np.array([x0, y0])) * 4).astype(np.int64)  # 2 bits of sub-pixel precision
    poly = np.clip(poly, -(2**28), 2**28).astype(np.int32)
    for p in poly:  # one by one: cv2.fillPoly leaves holes where polygons overlap
        cv2.fillConvexPoly(canvas, p, 1, lineType=cv2.LINE_8, shift=2)
    total = int(canvas.sum())
    ix0, iy0, ix1, iy1 = max(x0, 0), max(y0, 0), min(x1, width), min(y1, height)
    if ix1 <= ix0 or iy1 <= iy0:
        return total, 0, None
    inside = canvas[iy0 - y0 : iy1 - y0, ix0 - x0 : ix1 - x0]
    box = L.bbox_from_mask(inside)
    if box is not None:
        box = [box[0] + ix0, box[1] + iy0, box[2] + ix0, box[3] + iy0]
    return total, int(inside.sum()), box


class FrameWriter:
    """Writes images with a thread pool (PNG encoding releases the GIL) and labels as JSON."""

    def __init__(self, num_workers: int = 6):
        self._pool = ThreadPoolExecutor(max_workers=num_workers)
        self._pending: list[Future] = []

    def _imwrite(self, path: Path, img: np.ndarray):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._pending.append(self._pool.submit(cv2.imwrite, str(path), img))

    def flush(self):
        for f in self._pending:
            if not f.result():
                raise OSError("cv2.imwrite failed")
        self._pending.clear()

    def close(self):
        self.flush()
        self._pool.shutdown()

    def write(
        self,
        clip_dir: Path,
        frame: int,
        rgb: np.ndarray,
        depth: np.ndarray | None,
        instance_raw: np.ndarray,
        instance_paths: dict[int, str],
        objects: list[SceneObject],
        t_world_cam_usd: np.ndarray,
        k: np.ndarray,
        normals_world: np.ndarray | None = None,
        background_ids: dict[str, int] | None = None,
        min_visible_fraction: float = 0.05,
    ) -> dict:
        """Write all outputs of one frame and return its label dict.

        Args:
            rgb: Color image [H, W, 3], uint8, RGB order.
            depth: Distance to the image plane [m], [H, W].
            instance_raw: Renderer instance ids, [H, W].
            instance_paths: Renderer instance id to labelled prim path.
            normals_world: World-frame normals [H, W, 3].
            background_ids: Segmentation ids of labelled non-part prims; when given, a whole-scene
                ``segmentation`` map is written as well.
        """
        h, w = rgb.shape[:2]
        name = f"{frame:06d}"
        self._imwrite(clip_dir / "rgb" / f"{name}.png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

        # renderer ids -> our per-clip instance ids (0 = background, unlabelled geometry and clutter)
        by_path = {o.prim_path: o for o in objects}
        lut = np.zeros(max(instance_paths, default=0) + 1, dtype=np.uint16)
        for rid, path in instance_paths.items():
            obj = by_path.get(path) or next((o for o in objects if path.startswith(o.prim_path + "/")), None)
            if obj is not None:
                lut[rid] = obj.instance_id
        raw = np.minimum(instance_raw, len(lut) - 1)  # ids missing from the mapping count as background
        instance = lut[raw]
        if background_ids is not None:
            seg_lut = lut.copy()
            for rid, path in instance_paths.items():
                if path in background_ids:
                    seg_lut[rid] = background_ids[path]
            self._imwrite(clip_dir / "segmentation" / f"{name}.png", seg_lut[raw])
        class_of = np.zeros(max((o.instance_id for o in objects), default=0) + 1, dtype=np.uint8)
        for o in objects:
            class_of[o.instance_id] = o.class_id
        self._imwrite(clip_dir / "instance" / f"{name}.png", instance)
        self._imwrite(clip_dir / "semantic" / f"{name}.png", class_of[instance])
        if depth is not None:
            self._imwrite(clip_dir / "depth" / f"{name}.png", L.encode_depth_mm(depth))

        t_world_cam = t_world_cam_usd @ USD_TO_CV
        t_cam_world = np.linalg.inv(t_world_cam)
        if normals_world is not None:
            n_cam = normals_world @ t_world_cam[:3, :3]  # R^T n for row vectors
            self._imwrite(
                clip_dir / "normals" / f"{name}.png", cv2.cvtColor(L.encode_normals(n_cam), cv2.COLOR_RGB2BGR)
            )

        objs = []
        for o in objects:
            mask = instance == o.instance_id
            area = int(mask.sum())
            if area == 0:
                continue
            t_cam_obj = t_cam_world @ o.T_world_obj
            amodal_total, amodal_inside, amodal_box = rasterize_mesh(k, t_cam_obj, o.vertices, o.faces, w, h)
            visible = min(1.0, area / amodal_total) if amodal_total else 1.0
            corners = L.box_corners(o.bbox_min, o.bbox_max)
            corners_cam = L.transform_points(t_cam_obj, corners)
            center_cam = L.transform_points(t_cam_obj, ((o.bbox_min + o.bbox_max) / 2)[None])[0]
            objs.append(
                {
                    "instance_id": o.instance_id,
                    "class": o.category,
                    "class_id": o.class_id,
                    "part": o.part_name,
                    "prim_path": o.prim_path,
                    "bbox_2d": L.bbox_from_mask(mask),
                    "bbox_2d_amodal": amodal_box,
                    "pixel_area": area,
                    "visible_fraction": round(visible, 4),
                    "truncation": round(1.0 - amodal_inside / amodal_total, 4) if amodal_total else 0.0,
                    "ignore": bool(visible < min_visible_fraction or area < MIN_PIXEL_AREA),
                    "mask_rle": L.rle_encode(mask),
                    "pose": {
                        "T_cam_obj": t_cam_obj.tolist(),
                        "position": t_cam_obj[:3, 3].tolist(),
                        "quaternion_wxyz": matrix_to_quat(t_cam_obj[:3, :3]).tolist(),
                        "T_world_obj": o.T_world_obj.tolist(),
                    },
                    "bbox_3d": {
                        "min_obj": o.bbox_min.tolist(),
                        "max_obj": o.bbox_max.tolist(),
                        "center_cam": center_cam.tolist(),
                        "corners_cam": corners_cam.tolist(),
                        "corners_2d": L.project(k, corners_cam).tolist(),
                    },
                }
            )
        label = {
            "frame": frame,
            "width": w,
            "height": h,
            "files": {
                "rgb": f"rgb/{name}.png",
                "depth": f"depth/{name}.png" if depth is not None else None,
                "instance": f"instance/{name}.png",
                "semantic": f"semantic/{name}.png",
                "segmentation": f"segmentation/{name}.png" if background_ids is not None else None,
                "normals": f"normals/{name}.png" if normals_world is not None else None,
            },
            "camera": {"K": k.tolist(), "T_world_cam": t_world_cam.tolist()},
            "objects": objs,
        }
        (clip_dir / "labels").mkdir(parents=True, exist_ok=True)
        (clip_dir / "labels" / f"{name}.json").write_text(json.dumps(label))
        return label
