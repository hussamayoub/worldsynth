"""Draw labels on dataset images for a visual check: masks, 2D boxes, projected 3D boxes and pose axes.

    python scripts/visualize.py output/run_000 --clip 0 --frames 0 30 60
    python scripts/visualize.py output/run_000 --image_dir rgb_cosmos --all --video   # also writes an mp4 per clip
    python scripts/visualize.py output/run_000 --image_dir rgb_cosmos --clip 0 1 --side_by_side

Outputs go to ``<dataset>/viz/<image_dir>/``; ``--side_by_side`` writes ``viz/side_by_side_<image_dir>/<clip>.mp4``
with the render, the ``--image_dir`` images and those images with labels next to each other (all frames).
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from worldsynth.viz import draw


def side_by_side(root: Path, image_dir: str, clips: list[int], panel_height: int = 360):
    """One mp4 per clip: rendered rgb | ``image_dir`` | ``image_dir`` with labels."""
    import imageio.v2 as imageio

    fps = json.loads((root / "dataset.json").read_text())["config"]["dataset"]["fps"]
    out_dir = root / "viz" / f"side_by_side_{image_dir}"
    out_dir.mkdir(parents=True, exist_ok=True)
    for clip in clips:
        clip_dir = root / "clips" / f"clip_{clip:04d}"
        frames = []
        for lf in sorted((clip_dir / "labels").glob("*.json")):
            render = cv2.imread(str(clip_dir / "rgb" / f"{lf.stem}.png"))
            img = cv2.imread(str(clip_dir / image_dir / f"{lf.stem}.png"))
            if render is None or img is None:
                continue
            w = round(render.shape[1] * panel_height / render.shape[0])
            panels = [render, img, draw(img, json.loads(lf.read_text()))]
            row = np.hstack([cv2.resize(p, (w, panel_height), interpolation=cv2.INTER_AREA) for p in panels])
            frames.append(cv2.cvtColor(row, cv2.COLOR_BGR2RGB))
        if frames:
            imageio.mimwrite(out_dir / f"{clip_dir.name}.mp4", frames, fps=fps, quality=8, macro_block_size=1)
        else:
            print(f"{clip_dir.name}: no {image_dir} images")
    print(f"wrote to {out_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("dataset")
    parser.add_argument("--image_dir", default="rgb", help="Image folder in each clip (rgb or rgb_cosmos).")
    parser.add_argument("--clip", type=int, nargs="*", default=[0])
    parser.add_argument("--frames", type=int, nargs="*", default=[0])
    parser.add_argument("--all", action="store_true", help="All frames of the selected clips.")
    parser.add_argument("--video", action="store_true", help="Write an mp4 per clip instead of PNGs.")
    parser.add_argument(
        "--side_by_side", action="store_true", help="Write render | image_dir | labels videos (all frames)."
    )
    parser.add_argument("--all_clips", action="store_true", help="Every clip that has the image_dir images.")
    args = parser.parse_args()

    root = Path(args.dataset)
    if args.all_clips:
        args.clip = [
            int(p.name.split("_")[1]) for p in sorted((root / "clips").glob("clip_*")) if (p / args.image_dir).is_dir()
        ]
    if args.side_by_side:
        side_by_side(root, args.image_dir, args.clip)
        return
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
