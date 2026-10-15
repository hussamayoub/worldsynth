"""Draw WorldSynth labels on images: masks, 2D boxes, projected 3D boxes, pose axes and class names."""

import cv2
import numpy as np

from . import labels as L

EDGES = [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7), (0, 4), (1, 5), (2, 6), (3, 7)]


def color(i: int) -> tuple[int, int, int]:
    hue = (i * 47) % 180
    c = cv2.cvtColor(np.uint8([[[hue, 220, 255]]]), cv2.COLOR_HSV2BGR)[0, 0]
    return int(c[0]), int(c[1]), int(c[2])


def draw(img: np.ndarray, label: dict) -> np.ndarray:
    out = img.copy()
    overlay = img.copy()
    k = np.array(label["camera"]["K"])
    for obj in label["objects"]:
        c = color(obj["instance_id"])
        mask = L.rle_decode(obj["mask_rle"])
        overlay[mask] = c
        x0, y0, x1, y1 = obj["bbox_2d"]
        style = 1 if obj["ignore"] else 2
        cv2.rectangle(out, (x0, y0), (x1 - 1, y1 - 1), c, style)
        pts = np.round(np.array(obj["bbox_3d"]["corners_2d"])).astype(int)
        for a, b in EDGES:
            cv2.line(out, tuple(pts[a]), tuple(pts[b]), c, 1, cv2.LINE_AA)
        t = np.array(obj["pose"]["T_cam_obj"])
        size = 0.5 * float(np.min(np.array(obj["bbox_3d"]["max_obj"]) - np.array(obj["bbox_3d"]["min_obj"])))
        axes = L.transform_points(t, np.array([[0, 0, 0], [size, 0, 0], [0, size, 0], [0, 0, size]], dtype=float))
        if np.all(axes[:, 2] > 0):
            uv = np.round(L.project(k, axes)).astype(int)
            for j, col in zip((1, 2, 3), ((0, 0, 255), (0, 255, 0), (255, 0, 0)), strict=True):
                cv2.line(out, tuple(uv[0]), tuple(uv[j]), col, 2, cv2.LINE_AA)
        text = f"{obj['class']} v={obj['visible_fraction']:.2f}"
        cv2.putText(out, text, (x0, max(10, y0 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, c, 1, cv2.LINE_AA)
    return cv2.addWeighted(overlay, 0.35, out, 0.65, 0)
