"""Cosmos Transfer stage: export clips as control videos + specs, and import the generated videos as a new image set.

The labels never change: Cosmos is conditioned on edges, depth and segmentation of the rendered clip, so the parts
stay where they were rendered and every label of the clip still applies to the generated frames. The import step
measures how well that holds (:func:`alignment_score`) and can drop clips where it does not.
"""

from __future__ import annotations

import copy
import json
import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger("worldsynth")

DEFAULT_SPEC = {
    "model_mode": "video2video",
    "resolution": "480",
    "aspect_ratio": "16,9",
    "shift": 10.0,
    "num_steps": 50,
    "seed": 2026,
    "num_video_frames_per_chunk": 121,
    "num_conditional_frames": 1,
    "num_first_chunk_conditional_frames": 0,
    "share_vision_temporal_positions": True,
    "negative_metadata_mode": "none",
    "negative_prompt_keep_metadata": False,
    "guidance": 3.0,
    "control_guidance": 1.5,
    "emphasize_control_in_prompt": False,
}
"""Spec fields from the Cosmos3 Transfer runs that worked on the DGX Spark (kuka/scripts/cosmos/specs)."""

DEFAULT_CONTROLS = {
    "edge": {"weight": 1.0, "preset_edge_threshold": "medium"},
    "depth": {"weight": 0.5},
    "seg": {"weight": 0.5},
}


COSMOS_OUTPUT_SIZES = {
    ("256", "16,9"): (320, 192),
    ("256", "1,1"): (256, 256),
    ("480", "16,9"): (832, 480),
}
"""Output (width, height) per Cosmos3 resolution preset and aspect ratio, measured on the DGX Spark."""

ASPECT_TOLERANCE = 0.005


def check_aspect(dataset_size: tuple[int, int], output_size: tuple[int, int]):
    """Raise if the dataset and the Cosmos output differ in aspect ratio.

    Cosmos center-crops its input to the output aspect ratio. Stretching the output back to the dataset size would
    then shift every label toward the image borders, so the aspect ratios have to match.
    """
    (w, h), (cw, ch) = dataset_size, output_size
    if abs(w / h - cw / ch) > ASPECT_TOLERANCE * cw / ch:
        raise ValueError(
            f"Dataset images are {w}x{h} (aspect {w / h:.3f}) but Cosmos outputs {cw}x{ch} (aspect {cw / ch:.3f}). "
            f"Cosmos crops to its aspect ratio, which would misalign the labels. Render the dataset at "
            f"{round(h * cw / ch)}x{h} (dataset.resolution=[{round(h * cw / ch)},{h}]) or another size with the "
            f"aspect {cw}:{ch}."
        )


# ---------------------------------------------------------------------------------------------------------------------
# control videos


def depth_to_control(depths_mm: list[np.ndarray]) -> list[np.ndarray]:
    """Depth maps [mm] of one clip to 8-bit RGB frames, near = bright (like monocular depth estimators output).

    Normalized with one inverse-depth range for the whole clip so brightness does not flicker between frames.
    """
    inv = [np.where(d > 0, 1000.0 / np.maximum(d, 1), 0.0) for d in depths_mm]
    valid = np.concatenate([x[x > 0] for x in inv]) if inv else np.array([1.0])
    lo, hi = np.percentile(valid, [0.5, 99.9]) if valid.size else (0.0, 1.0)
    out = []
    for x in inv:
        g = np.clip((x - lo) / max(hi - lo, 1e-9), 0, 1)
        g = (g * 255).astype(np.uint8)
        out.append(np.repeat(g[..., None], 3, axis=2))
    return out


def segmentation_to_control(segs: list[np.ndarray], seed: int = 0) -> list[np.ndarray]:
    """Segment-id maps of one clip to RGB frames with a fixed random color per id (0 = black)."""
    max_id = max(int(s.max()) for s in segs) if segs else 0
    rng = np.random.default_rng(seed)
    palette = rng.integers(40, 256, size=(max_id + 1, 3), dtype=np.uint8)
    palette[0] = 0
    return [palette[s] for s in segs]


def write_video(path: Path, frames: list[np.ndarray], fps: int, size: tuple[int, int] | None = None):
    """H.264 mp4 at high quality; ``size`` = (width, height) to resize first."""
    import imageio.v2 as imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    if size is not None:
        frames = [cv2.resize(f, size, interpolation=cv2.INTER_AREA) for f in frames]
    imageio.mimwrite(path, frames, fps=fps, codec="libx264", quality=9, macro_block_size=1, pixelformat="yuv420p")


def read_video(path: Path) -> list[np.ndarray]:
    import imageio.v2 as imageio

    return [np.asarray(f)[..., :3] for f in imageio.mimread(path, memtest=False)]


# ---------------------------------------------------------------------------------------------------------------------
# prompts


def load_prompt(path: Path, categories: list[str]) -> dict:
    """Structured Cosmos prompt from a template; ``{parts}`` is replaced by the dataset's class names."""
    text = path.read_text()
    names = [c.replace("_", " ") for c in categories]
    parts = ", ".join(names[:-1]) + (" and " if len(names) > 1 else "") + names[-1] if names else "parts"
    return json.loads(text.replace("{parts}", parts))


CLUTTER_LOOKS = {
    "cube": ("small cardboard box", "small cardboard boxes"),
    "cylinder": ("plastic cup", "plastic cups"),
    "sphere": ("rubber ball", "rubber balls"),
    "cone": ("small plastic traffic cone", "small plastic traffic cones"),
}
"""What each clutter primitive is called in the prompt: everyday non-metal items, so Cosmos does not turn the
unlabelled clutter into objects that look like the labelled parts."""


def add_clutter_subject(prompt: dict, clutter: list[dict]) -> dict:
    """Copy of ``prompt`` with a subject describing the clip's clutter; unchanged if the clip has none.

    Mentioning clutter only when it exists keeps Cosmos from inventing extra objects in clutter-free clips.
    """
    if not clutter:
        return prompt
    counts: dict[str, int] = {}
    for item in clutter:
        counts[item["shape"]] = counts.get(item["shape"], 0) + 1
    items = []
    for shape, n in sorted(counts.items()):
        one, many = CLUTTER_LOOKS.get(shape, ("small object", "small objects"))
        items.append(f"{n} {many}" if n > 1 else f"1 {one}")
    prompt = copy.deepcopy(prompt)
    prompt.setdefault("subjects", []).append(
        {
            "description": "Unrelated everyday items lying on the table: " + ", ".join(items),
            "appearance_details": "Brown corrugated cardboard, matte colored plastic and rubber; clearly not machined "
            "parts and not metal",
            "relationship": "Leftover packaging and supplies that happen to be on the table, not workpieces",
            "location": "Scattered between and around the parts",
            "pose": "Resting on the table",
            "action": "Static",
        }
    )
    return prompt


# ---------------------------------------------------------------------------------------------------------------------
# prepare


def prepare(
    dataset_dir: str | Path,
    out_dir: str | Path,
    prompt_files: list[Path],
    negative_prompt_file: Path | None,
    remote_dir: str | None = None,
    clips: list[int] | None = None,
    variants_per_clip: int = 1,
    spec_overrides: dict | None = None,
    controls: dict | None = None,
    video_size: tuple[int, int] | None = None,
    seed: int = 0,
    job_prefix: str | None = None,
) -> dict:
    """Write a self-contained Cosmos batch folder for the clips of a dataset.

    Layout::

        <out_dir>/inputs/<clip>/{vision,depth,seg}.mp4
        <out_dir>/prompts/<job>.json   (the look plus the clip's clutter)
        <out_dir>/specs/<job>.json     (job = <job_prefix>__<clip>__<prompt>)
        <out_dir>/manifest.json
        <out_dir>/run_batch.sh

    Args:
        remote_dir: Absolute path of ``out_dir`` on the machine that runs Cosmos; spec paths point there.
            Defaults to ``out_dir`` itself (running Cosmos on this machine).
        variants_per_clip: Number of different prompts per clip, each giving its own image set.
        controls: Control name to settings; defaults to :data:`DEFAULT_CONTROLS`. ``edge`` is computed by Cosmos
            from the vision video, ``depth`` and ``seg`` are rendered by WorldSynth.
        video_size: Optional (width, height) to downscale the uploaded videos (e.g. the Cosmos output size).
        job_prefix: Start of every job name, unique per batch because all jobs share one Cosmos outputs folder.
            Defaults to the name of ``out_dir``.
    """
    dataset_dir, out_dir = Path(dataset_dir), Path(out_dir)
    meta = json.loads((dataset_dir / "dataset.json").read_text())
    fps = int(meta["config"]["dataset"]["fps"])
    categories = [c["name"] for c in meta["categories"]]
    remote = Path(remote_dir) if remote_dir else out_dir.resolve()
    controls = copy.deepcopy(controls or DEFAULT_CONTROLS)
    negative = json.loads(negative_prompt_file.read_text()) if negative_prompt_file else None

    spec_settings = {**DEFAULT_SPEC, **(spec_overrides or {})}
    output_size = COSMOS_OUTPUT_SIZES.get((str(spec_settings["resolution"]), str(spec_settings["aspect_ratio"])))
    if output_size is not None:
        check_aspect(tuple(meta["config"]["dataset"]["resolution"]), output_size)
    else:
        logger.warning(
            "Output size of Cosmos resolution %s / aspect %s is not known; cosmos_import checks the aspect ratio.",
            spec_settings["resolution"],
            spec_settings["aspect_ratio"],
        )

    prompts = {pf.stem: load_prompt(pf, categories) for pf in prompt_files}
    if not prompts:
        raise ValueError("At least one prompt file is needed.")
    names = sorted(prompts)

    clip_dirs = sorted(p for p in (dataset_dir / "clips").iterdir() if (p / "clip.json").exists())
    if clips is not None:
        wanted = {f"clip_{c:04d}" for c in clips}
        clip_dirs = [c for c in clip_dirs if c.name in wanted]
    offset = int(np.random.default_rng(seed).integers(len(names)))
    manifest = {"dataset": str(dataset_dir.resolve()), "fps": fps, "jobs": []}
    for clip_dir in clip_dirs:
        frames = sorted((clip_dir / "rgb").glob("*.png"))
        if not frames:
            continue
        rgb = [cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB) for f in frames]
        inputs = out_dir / "inputs" / clip_dir.name
        write_video(inputs / "vision.mp4", rgb, fps, video_size)
        if "depth" in controls:
            depth = [cv2.imread(str(clip_dir / "depth" / f.name), cv2.IMREAD_UNCHANGED) for f in frames]
            if any(d is None for d in depth):
                raise FileNotFoundError(f"{clip_dir}: depth control needs depth images (dataset.annotations: depth)")
            write_video(inputs / "depth.mp4", depth_to_control(depth), fps, video_size)
        if "seg" in controls:
            seg = [cv2.imread(str(clip_dir / "segmentation" / f.name), cv2.IMREAD_UNCHANGED) for f in frames]
            if any(s is None for s in seg):
                raise FileNotFoundError(f"{clip_dir}: seg control needs segmentation (dataset.annotations)")
            write_video(inputs / "seg.mp4", segmentation_to_control(seg, seed), fps, video_size)

        # round-robin over the looks so every prompt gets about the same number of clips
        start = len(manifest["jobs"]) + offset
        chosen = [names[(start + j) % len(names)] for j in range(min(max(1, variants_per_clip), len(names)))]
        clutter = json.loads((clip_dir / "clip.json").read_text()).get("clutter", [])
        for prompt_name in chosen:
            name = f"{job_prefix or out_dir.name}__{clip_dir.name}__{prompt_name}"
            spec = {"name": name, **DEFAULT_SPEC, **(spec_overrides or {})}
            spec.update(num_frames=len(frames), fps=fps)
            # one prompt per job: the look plus the clutter that is actually in this clip
            prompt = add_clutter_subject(prompts[prompt_name], clutter)
            (out_dir / "prompts").mkdir(parents=True, exist_ok=True)
            (out_dir / "prompts" / f"{name}.json").write_text(json.dumps(prompt, indent=2))
            spec["prompt_path"] = str(remote / "prompts" / f"{name}.json")
            spec["vision_path"] = str(remote / "inputs" / clip_dir.name / "vision.mp4")
            for ctrl, settings in controls.items():
                spec[ctrl] = dict(settings)
                if ctrl in ("depth", "seg"):
                    spec[ctrl]["control_path"] = str(remote / "inputs" / clip_dir.name / f"{ctrl}.mp4")
            if negative is not None:
                # a negative_prompt_file is ignored once defaults set one, so embed it in the spec
                spec["negative_prompt"] = json.dumps(negative)
            (out_dir / "specs").mkdir(parents=True, exist_ok=True)
            (out_dir / "specs" / f"{name}.json").write_text(json.dumps(spec, indent=2))
            manifest["jobs"].append(
                {"name": name, "clip": clip_dir.name, "prompt": prompt_name, "num_frames": len(frames)}
            )
        logger.info("%s: %d frames, prompts %s", clip_dir.name, len(frames), list(chosen))

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    script = out_dir / "run_batch.sh"
    script.write_text(RUN_BATCH.format(remote=remote))
    script.chmod(0o755)
    return manifest


RUN_BATCH = """#!/bin/bash
# Run every spec of this batch with Cosmos3 Transfer, skipping specs whose output already exists.
# Uses run_transfer.sh from kuka/scripts/cosmos (Cosmos3-Nano with a memory watchdog).
# Usage on the Cosmos machine: {remote}/run_batch.sh [path/to/run_transfer.sh]
set -u
RUN=${{1:-$HOME/cosmos-work/run_transfer.sh}}
OUT=${{COSMOS_ROOT:-$HOME/cosmos-work/cosmos}}/cookbooks/cosmos3/generator/transfer/outputs/Cosmos3-Nano
for spec in {remote}/specs/*.json; do
  name=$(basename "$spec" .json)
  if [ -f "$OUT/$name/vision.mp4" ]; then echo "skip $name"; continue; fi
  echo "$(date +%T) $name"
  "$RUN" "$spec"
done
echo "outputs in $OUT"
"""


# ---------------------------------------------------------------------------------------------------------------------
# import


def alignment_score(image: np.ndarray, segmentation: np.ndarray, tolerance: int = 3) -> float:
    """Fraction of segment boundaries that lie within ``tolerance`` px of an image edge.

    Computed on the generated frame and on the rendered frame of the same clip, the ratio shows whether Cosmos kept
    the scene geometry (ratio near 1) or moved, dropped or invented objects (ratio well below 1).
    """
    seg = segmentation.astype(np.int32)
    boundary = np.zeros(seg.shape, dtype=bool)
    boundary[:, 1:] |= seg[:, 1:] != seg[:, :-1]
    boundary[1:, :] |= seg[1:, :] != seg[:-1, :]
    if not boundary.any():
        return 1.0
    # per color channel, so boundaries between parts and background of similar brightness count too
    edges = np.zeros(seg.shape, dtype=bool)
    for c in range(3):
        ch = cv2.GaussianBlur(np.ascontiguousarray(image[..., c]), (5, 5), 0)
        median = float(np.median(ch))  # thresholds adapt to the image's brightness and contrast
        edges |= cv2.Canny(ch, int(0.33 * median), int(min(255, 0.66 * median))) > 0
    kernel = np.ones((2 * tolerance + 1, 2 * tolerance + 1), np.uint8)
    near_edge = cv2.dilate(edges.astype(np.uint8), kernel) > 0
    return float((boundary & near_edge).sum() / boundary.sum())


def import_results(
    dataset_dir: str | Path,
    batch_dir: str | Path,
    results_dir: str | Path,
    image_dir: str = "rgb_cosmos",
    min_alignment: float = 0.5,
    score_every: int = 5,
) -> dict:
    """Copy Cosmos outputs into the dataset as ``clips/<clip>/<image_dir>[_<prompt>]/`` next to the rendered images.

    Args:
        results_dir: Folder with one ``<spec name>/vision.mp4`` per job (the Cosmos ``outputs/Cosmos3-Nano``).
        min_alignment: Jobs whose alignment ratio (generated / rendered, see :func:`alignment_score`) is below this
            are reported and not imported.
        score_every: Score every n-th frame (alignment scoring is the slow part).

    Returns:
        A report with the alignment of each job, also written to ``<dataset>/annotations/<image_dir>_report.json``.
    """
    dataset_dir, batch_dir, results_dir = Path(dataset_dir), Path(batch_dir), Path(results_dir)
    manifest = json.loads((batch_dir / "manifest.json").read_text())
    prompts_per_clip: dict[str, int] = {}
    for job in manifest["jobs"]:
        prompts_per_clip[job["clip"]] = prompts_per_clip.get(job["clip"], 0) + 1
    multi = max(prompts_per_clip.values(), default=1) > 1
    report = {"image_dirs": [], "jobs": []}
    for job in manifest["jobs"]:
        video = results_dir / job["name"] / "vision.mp4"
        entry = {"name": job["name"], "clip": job["clip"], "prompt": job["prompt"], "status": "missing"}
        report["jobs"].append(entry)
        if not video.exists():
            logger.warning("missing output %s", video)
            continue
        clip_dir = dataset_dir / "clips" / job["clip"]
        rendered = sorted((clip_dir / "rgb").glob("*.png"))
        frames = read_video(video)
        h, w = cv2.imread(str(rendered[0])).shape[:2]
        entry["output_size"] = list(frames[0].shape[1::-1])
        entry["output_frames"] = len(frames)
        try:
            check_aspect((w, h), tuple(entry["output_size"]))
        except ValueError as e:
            entry["status"] = "aspect_mismatch"
            logger.error("%s not imported: %s", job["name"], e)
            continue
        if len(frames) != len(rendered):
            # keep frames aligned with labels by timestamp if Cosmos changed the frame count
            logger.warning(
                "%s: %d generated vs %d rendered frames; resampling", job["name"], len(frames), len(rendered)
            )
            idx = np.round(np.linspace(0, len(frames) - 1, len(rendered))).astype(int)
            frames = [frames[i] for i in idx]
        frames = [cv2.resize(f, (w, h), interpolation=cv2.INTER_CUBIC) for f in frames]

        ratios = []
        for i in range(0, len(rendered), max(1, score_every)):
            seg_path = clip_dir / "segmentation" / rendered[i].name
            seg = cv2.imread(str(seg_path if seg_path.exists() else clip_dir / "instance" / rendered[i].name), -1)
            ref = alignment_score(cv2.cvtColor(cv2.imread(str(rendered[i])), cv2.COLOR_BGR2RGB), seg)
            gen = alignment_score(frames[i], seg)
            ratios.append(gen / max(ref, 1e-6))
        entry["alignment"] = round(float(np.median(ratios)), 3)
        entry["alignment_min"] = round(float(np.min(ratios)), 3)
        if entry["alignment"] < min_alignment:
            entry["status"] = "rejected"
            logger.warning("%s rejected: alignment %.2f < %.2f", job["name"], entry["alignment"], min_alignment)
            continue
        target = f"{image_dir}_{job['prompt']}" if multi else image_dir
        out = clip_dir / target
        out.mkdir(parents=True, exist_ok=True)
        for f, path in zip(frames, rendered, strict=True):
            cv2.imwrite(str(out / path.name), cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        entry["status"] = "imported"
        entry["image_dir"] = target
        if target not in report["image_dirs"]:
            report["image_dirs"].append(target)
        logger.info("%s -> %s (alignment %.2f)", job["name"], out, entry["alignment"])
    (dataset_dir / "annotations").mkdir(parents=True, exist_ok=True)
    (dataset_dir / "annotations" / f"{image_dir}_report.json").write_text(json.dumps(report, indent=2))
    return report
