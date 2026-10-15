"""Find and read the videos of a run for viewing: rendered clips (as image sequences, optionally with labels, depth or
segmentation), imported Cosmos frames, Cosmos input videos, Cosmos results and side-by-side videos."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

RENDER_VIEWS = ("rgb", "rgb + labels", "depth", "segmentation", "instance")
COSMOS_VIEWS = ("rgb", "rgb + labels")


@dataclass
class VideoSource:
    group: str
    """Isaac Sim, Cosmos (imported), Cosmos inputs, Cosmos results or Side by side."""
    name: str
    views: tuple[str, ...] = ("video",)
    clip_dir: Path | None = None
    """Clip folder of an image sequence (frames are ``<clip_dir>/<image_dir>/*.png``)."""
    image_dir: str = "rgb"
    video: Path | None = None
    """An mp4 file."""
    fps: float = 20.0
    _frames: list = field(default_factory=list, repr=False)

    @property
    def key(self) -> str:
        return f"{self.group}/{self.name}"


def find_sources(dataset: Path | None, cosmos: Path | None) -> list[VideoSource]:
    """Every video of a dataset folder and a Cosmos folder (either may be missing)."""
    out: list[VideoSource] = []
    fps = 20.0
    if dataset and (dataset / "dataset.json").exists():
        meta = json.loads((dataset / "dataset.json").read_text())
        fps = float(meta.get("config", {}).get("dataset", {}).get("fps", fps))
    if dataset and (dataset / "clips").is_dir():
        clips = sorted(p for p in (dataset / "clips").iterdir() if (p / "rgb").is_dir())
        for clip in clips:
            out.append(VideoSource("Isaac Sim", clip.name, RENDER_VIEWS, clip, "rgb", fps=fps))
        for clip in clips:
            for d in sorted(p for p in clip.iterdir() if p.is_dir() and p.name.startswith("rgb_cosmos")):
                out.append(
                    VideoSource("Cosmos (imported)", f"{clip.name} {d.name}", COSMOS_VIEWS, clip, d.name, fps=fps)
                )
        for mp4 in sorted((dataset / "viz").glob("side_by_side_*/*.mp4")):
            out.append(VideoSource("Side by side", f"{mp4.stem} ({mp4.parent.name[13:]})", video=mp4, fps=fps))
        for mp4 in sorted((dataset / "viz").glob("rgb*/*.mp4")):
            out.append(VideoSource("Label previews", f"{mp4.stem} ({mp4.parent.name})", video=mp4, fps=fps))
    if cosmos and (cosmos / "inputs").is_dir():
        for clip in sorted((cosmos / "inputs").iterdir()):
            for mp4 in sorted(clip.glob("*.mp4")):
                out.append(VideoSource("Cosmos inputs", f"{clip.name} {mp4.stem}", video=mp4, fps=fps))
    if cosmos and (cosmos / "results").is_dir():
        for mp4 in sorted((cosmos / "results").glob("*/vision.mp4")):
            out.append(VideoSource("Cosmos results", mp4.parent.name, video=mp4, fps=fps))
    return out


def frame_count(src: VideoSource) -> int:
    if src.video is not None:
        return len(_video_frames(src))
    return len(sorted((src.clip_dir / src.image_dir).glob("*.png")))


def _video_frames(src: VideoSource) -> list[np.ndarray]:
    if not src._frames:
        import imageio.v3 as iio

        src._frames = [cv2.cvtColor(np.asarray(f)[..., :3], cv2.COLOR_RGB2BGR) for f in iio.imiter(src.video)]
    return src._frames


def palette(ids: np.ndarray) -> np.ndarray:
    """Fixed random BGR color per id, black for 0."""
    rng = np.random.default_rng(7)
    colors = rng.integers(60, 256, size=(int(ids.max()) + 1, 3), dtype=np.uint8)
    colors[0] = 0
    return colors[ids]


def read_frame(src: VideoSource, index: int, view: str) -> np.ndarray | None:
    """Frame ``index`` of a source as a BGR image, in the given view; None if it does not exist."""
    if src.video is not None:
        frames = _video_frames(src)
        return frames[index] if 0 <= index < len(frames) else None
    files = sorted((src.clip_dir / src.image_dir).glob("*.png"))
    if not 0 <= index < len(files):
        return None
    name = files[index].name
    if view == "depth":
        d = cv2.imread(str(src.clip_dir / "depth" / name), cv2.IMREAD_UNCHANGED)
        if d is None:
            return None
        inv = np.where(d > 0, 1.0 / np.maximum(d, 1), 0.0)  # near = bright
        valid = inv[inv > 0]
        lo, hi = np.percentile(valid, [1, 99]) if valid.size else (0, 1)
        g = (np.clip((inv - lo) / max(hi - lo, 1e-12), 0, 1) * 255).astype(np.uint8)
        return cv2.applyColorMap(g, cv2.COLORMAP_INFERNO)
    if view in ("segmentation", "instance"):
        ids = cv2.imread(str(src.clip_dir / view / name), cv2.IMREAD_UNCHANGED)
        return None if ids is None else palette(ids)
    img = cv2.imread(str(files[index]))
    if img is not None and view == "rgb + labels":
        label_file = src.clip_dir / "labels" / f"{files[index].stem}.json"
        if label_file.exists():
            from .viz import draw

            img = draw(img, json.loads(label_file.read_text()))
    return img
