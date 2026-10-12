"""Draw labels on dataset images for a visual check: masks, 2D boxes, projected 3D boxes and pose axes.

    python scripts/visualize.py output/run_000 --clip 0 --frames 0 30 60
    python scripts/visualize.py output/run_000 --image_dir rgb_cosmos --all --video   # also writes an mp4 per clip

Outputs go to ``<dataset>/viz/<image_dir>/``.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from worldsynth import labels as L

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


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("dataset")
    parser.add_argument("--image_dir", default="rgb", help="Image folder in each clip (rgb or rgb_cosmos).")
    parser.add_argument("--clip", type=int, nargs="*", default=[0])
    parser.add_argument("--frames", type=int, nargs="*", default=[0])
    parser.add_argument("--all", action="store_true", help="All frames of the selected clips.")
    parser.add_argument("--video", action="store_true", help="Write an mp4 per clip instead of PNGs.")
    args = parser.parse_args()

    root = Path(args.dataset)
    out_dir = root / "viz" / args.image_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    fps = json.loads((root / "dataset.json").read_text())["config"]["dataset"]["fps"]
    for clip in args.clip:
        clip_dir = root / "clips" / f"clip_{clip:04d}"
        label_files = sorted((clip_dir / "labels").glob("*.json"))
        if not args.all:
            label_files = [clip_dir / "labels" / f"{f:06d}.json" for f in args.frames]
        frames = []
        for lf in label_files:
            label = json.loads(lf.read_text())
            img = cv2.imread(str(clip_dir / args.image_dir / f"{lf.stem}.png"))
            if img is None:
                print(f"missing image for {lf}")
                continue
            vis = draw(img, label)
            if args.video:
                frames.append(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
            else:
                cv2.imwrite(str(out_dir / f"{clip_dir.name}_{lf.stem}.png"), vis)
        if args.video and frames:
            import imageio.v2 as imageio

            imageio.mimwrite(out_dir / f"{clip_dir.name}.mp4", frames, fps=fps, quality=8, macro_block_size=1)
    print(f"wrote to {out_dir}")


if __name__ == "__main__":
    main()
