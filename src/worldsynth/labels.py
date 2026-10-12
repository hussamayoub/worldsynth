"""Label encoding and dataset export (COCO). Pure NumPy/OpenCV, no Isaac Sim needed."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------------------------------------------------
# masks


def rle_encode(mask: np.ndarray) -> dict:
    """COCO compressed RLE of a binary mask (same output as ``pycocotools.mask.encode``)."""
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    flat = mask.flatten(order="F").astype(np.int8)
    change = np.flatnonzero(np.diff(flat)) + 1
    bounds = np.concatenate([[0], change, [flat.size]])
    counts = np.diff(bounds).tolist()
    if flat.size and flat[0] == 1:
        counts = [0] + counts  # runs start with zeros
    return {"size": [h, w], "counts": _counts_to_string(counts)}


def _counts_to_string(counts: list[int]) -> str:
    out = []
    for i, x in enumerate(counts):
        if i > 2:
            x -= counts[i - 2]
        more = True
        while more:
            c = x & 0x1F
            x >>= 5
            more = (x != -1) if (c & 0x10) else (x != 0)
            if more:
                c |= 0x20
            out.append(chr(c + 48))
    return "".join(out)


def rle_decode(rle: dict) -> np.ndarray:
    h, w = rle["size"]
    s = rle["counts"]
    counts: list[int] = []
    p = 0
    while p < len(s):
        x, k, more = 0, 0, True
        while more:
            c = ord(s[p]) - 48
            x |= (c & 0x1F) << (5 * k)
            more = bool(c & 0x20)
            p += 1
            k += 1
            if not more and (c & 0x10):
                x |= -1 << (5 * k)
        if len(counts) > 2:
            x += counts[-2]
        counts.append(x)
    flat = np.zeros(h * w, dtype=bool)
    pos, val = 0, False
    for c in counts:
        if val:
            flat[pos : pos + c] = True
        pos += c
        val = not val
    return flat.reshape((w, h)).T


def mask_to_polygons(mask: np.ndarray, min_points: int = 3) -> list[list[float]]:
    import cv2

    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [c.reshape(-1).astype(float).tolist() for c in contours if len(c) >= min_points]


def bbox_from_mask(mask: np.ndarray) -> list[int] | None:
    """Tight box ``[x_min, y_min, x_max, y_max]`` in pixels, max exclusive; ``None`` for an empty mask."""
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


# ---------------------------------------------------------------------------------------------------------------------
# geometry


def box_corners(lo, hi) -> np.ndarray:
    """8 corners of an axis-aligned box, shape [8, 3]."""
    lo, hi = np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)
    idx = np.array([[i >> 2 & 1, i >> 1 & 1, i & 1] for i in range(8)])
    return np.where(idx, hi, lo)


def transform_points(t: np.ndarray, pts: np.ndarray) -> np.ndarray:
    return pts @ t[:3, :3].T + t[:3, 3]


def project(k: np.ndarray, pts_cam: np.ndarray) -> np.ndarray:
    """Project OpenCV-camera-frame points [N, 3] to pixels [N, 2]."""
    z = np.clip(pts_cam[:, 2:3], 1e-6, None)
    return (pts_cam[:, :2] / z) * np.array([k[0, 0], k[1, 1]]) + np.array([k[0, 2], k[1, 2]])


# ---------------------------------------------------------------------------------------------------------------------
# images


def encode_depth_mm(depth_m: np.ndarray) -> np.ndarray:
    """Depth [m] to uint16 millimeters; invalid or out-of-range pixels become 0."""
    d = np.nan_to_num(np.asarray(depth_m, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    d = np.round(d * 1000.0)
    d[(d <= 0) | (d > 65535)] = 0
    return d.astype(np.uint16)


def encode_normals(normals: np.ndarray) -> np.ndarray:
    """Unit normals [-1, 1] to RGB uint8."""
    return np.clip((normals + 1.0) * 127.5, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------------------------------------------------
# COCO


def build_coco(dataset_dir: str | Path, image_dir: str | list[str] = "rgb", out_path: str | Path | None = None) -> dict:
    """Aggregate the per-frame label files of a dataset into one COCO file.

    Objects flagged ``ignore`` (too occluded or tiny) become ``iscrowd=1`` regions so COCO evaluation ignores them
    instead of counting detections on them as false positives.

    Args:
        dataset_dir: WorldSynth dataset root (contains ``dataset.json`` and ``clips/``).
        image_dir: Image folder(s) inside each clip: ``rgb`` for Isaac Sim renders, ``rgb_cosmos`` for Cosmos
            output. With several folders, every labelled frame appears once per folder that has its image.
        out_path: Output file; default ``<dataset_dir>/annotations/coco_<image_dir>.json``.
    """
    dataset_dir = Path(dataset_dir)
    image_dirs = [image_dir] if isinstance(image_dir, str) else list(image_dir)
    meta = json.loads((dataset_dir / "dataset.json").read_text())
    categories = [{"id": c["id"], "name": c["name"], "supercategory": "part"} for c in meta["categories"]]
    images, annotations = [], []
    jobs = [
        (clip_dir, label_file, d)
        for clip_dir in sorted((dataset_dir / "clips").iterdir())
        for label_file in sorted((clip_dir / "labels").glob("*.json"))
        for d in image_dirs
    ]
    for clip_dir, label_file, img_dir in jobs:
        img_rel = clip_dir.relative_to(dataset_dir) / img_dir / f"{label_file.stem}.png"
        if not (dataset_dir / img_rel).exists():
            continue
        frame = json.loads(label_file.read_text())
        image_id = len(images) + 1
        images.append(
            {
                "id": image_id,
                "file_name": str(img_rel),
                "width": frame["width"],
                "height": frame["height"],
                "clip": clip_dir.name,
                "frame": int(label_file.stem),
                "source": img_dir,
            }
        )
        for obj in frame["objects"]:
            if obj.get("bbox_2d") is None:
                continue
            x0, y0, x1, y1 = obj["bbox_2d"]
            seg = obj["mask_rle"]
            if not obj["ignore"] and meta.get("coco_segmentation", "rle") == "polygon":
                seg = mask_to_polygons(rle_decode(seg)) or seg
            annotations.append(
                {
                    "id": len(annotations) + 1,
                    "image_id": image_id,
                    "category_id": obj["class_id"],
                    "bbox": [x0, y0, x1 - x0, y1 - y0],
                    "area": obj["pixel_area"],
                    "segmentation": seg,
                    "iscrowd": int(obj["ignore"]),
                    "instance_id": obj["instance_id"],
                    "visible_fraction": obj["visible_fraction"],
                }
            )
    coco = {
        "info": {"description": f"WorldSynth dataset {dataset_dir.name} ({', '.join(image_dirs)})", "version": "1.0"},
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }
    default_name = f"coco_{'+'.join(image_dirs)}.json"
    out_path = Path(out_path) if out_path else dataset_dir / "annotations" / default_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(coco))
    return coco
