"""WorldSynth desktop GUI: pick the CAD folder, an optional USD scene with its camera and surface, the dataset and
Cosmos settings, the output folders and the Spark login; adjust any config value on the Configuration tab (load a
YAML, override values); then preview a frame or run the whole pipeline (``scripts/run_pipeline.sh``).

This module holds the settings and everything the window does without the UI toolkit (checks, commands, configs,
prompts); the Qt window is in :mod:`worldsynth.gui_qt`. Start it with the Isaac Lab Python (it needs ``pxr`` to read
scenes): ``scripts/gui.sh``. It starts with the default settings; ``--restore`` loads the settings of the last run
(``--preview`` then renders a preview right away).

The SSH password is only handed to ``ssh`` through the environment of the pipeline process (via ``SSH_ASKPASS``).
It is never written to disk and not saved with the other settings.
"""

from __future__ import annotations

import dataclasses
import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from .config import load_config
from .demos import CONVERT_ARGS

ROOT = Path(__file__).resolve().parents[2]
SETTINGS_FILE = Path(os.environ.get("WORLDSYNTH_GUI_SETTINGS", Path.home() / ".config" / "worldsynth" / "gui.json"))
DEFAULT_CONFIG = ROOT / "configs" / "default.yaml"
RUN_CONFIG = ROOT / "cache" / "gui_run_config.yaml"
"""Merged configuration (base YAML + Configuration tab overrides) handed to each preview and run."""
PROMPT_DIR = ROOT / "cache" / "gui_prompts"
"""Prompts (with the Prompts tab edits) handed to each run."""
STAGES = ("generate", "prepare", "send", "fetch")
"""Steps of every run; recorded demos can add ``convert`` (see :func:`run_all_stages`)."""
ASKPASS = Path.home() / ".cache" / "worldsynth" / "askpass.sh"
ALL_LOOKS = "all looks in configs/cosmos/prompts"
RANDOM_CAMERA = "(random cameras around the parts)"
NO_SCENE_PARTS = "(none: parts only from the CAD folder)"


@dataclass
class GuiSettings:
    source: str = "cad"
    """``cad``: render CAD parts in Isaac Sim; ``demos``: recorded robot demos from an HDF5 file."""
    demos_hdf5: str = ""
    demo_cameras: str = "scene_cam,wrist_cam"
    max_demos: int = 0
    """0: all demos."""
    seg_names: str = ""
    """Class names of the demos' segmentation ids, e.g. ``1=robot,10-15=soup_can``."""
    demos_out: str = ""
    """HDF5 with the Cosmos images; empty: ``<dataset folder>/demos_cosmos.hdf5``."""
    keep_missing: bool = False
    """Also export demos where only some cameras have Cosmos frames."""
    convert_script: str = ""
    """Converter from HDF5 demos to training data (robot project), e.g. ``convert_to_gr00t_lerobot.py``."""
    convert_python: str = ""
    """Interpreter with the converter's dependencies; empty: Isaac Lab's."""
    convert_out: str = ""
    """Training dataset folder; empty: ``<dataset folder>/training_data``."""
    convert_args: str = CONVERT_ARGS
    cosmos_only: bool = False
    """Convert only the Cosmos demos, not the original ones too."""
    convert_overwrite: bool = False
    cad_dir: str = ""
    units: str = "auto"
    class_from: str = "file"
    use_scene: bool = False
    scene_usd: str = ""
    scene_camera: str = ""
    """Empty: sample cameras around the parts."""
    scene_surface: str = ""
    scene_parts: str = ""
    """Prim in the scene whose children are labelled parts; empty: none."""
    scene_parts_mode: str = "place"
    scene_lights_only: bool = True
    """A predesigned scene is usually lit already; added random lights tend to wash it out."""
    camera_azimuth: float = 10.0
    """± degrees the scene camera moves around its target over a clip."""
    camera_elevation: float = 3.0
    name: str = "run_001"
    dataset_dir: str = ""
    """Where Isaac Sim writes the dataset; empty: ``output/<name>``."""
    cosmos_dir: str = ""
    """Cosmos batch folder, results are copied back to its ``results/``; empty: ``output/<name>_cosmos``."""
    clips: int = 20
    frames: int = 61
    parts_min: int = 3
    parts_max: int = 8
    clutter: bool = True
    clutter_min: int = 0
    clutter_max: int = 8
    resolution: str = "256"
    steps: int = 50
    looks: int = 1
    prompt: str = ALL_LOOKS
    negative: str = "configs/cosmos/negative_prompt.json"
    config_file: str = ""
    """Base YAML config; empty: ``configs/default.yaml``."""
    config_overrides: dict[str, str] = field(default_factory=dict)
    """Configuration tab overrides: dotted key -> value in YAML syntax."""
    spark_host: str = "100.123.185.58"
    spark_user: str = "h"
    password: str = field(default="", metadata={"secret": True})

    def to_saved(self) -> dict:
        """Settings to persist: everything except secrets."""
        return {f.name: getattr(self, f.name) for f in dataclasses.fields(self) if not f.metadata.get("secret")}

    @classmethod
    def from_saved(cls, data: dict) -> GuiSettings:
        names = {f.name for f in dataclasses.fields(cls) if not f.metadata.get("secret")}
        return cls(**{k: v for k, v in data.items() if k in names})


def load_settings(path: Path = SETTINGS_FILE) -> GuiSettings:
    try:
        return GuiSettings.from_saved(json.loads(path.read_text()))
    except (OSError, ValueError, TypeError):
        return GuiSettings()


def save_settings(s: GuiSettings, path: Path = SETTINGS_FILE):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(s.to_saved(), indent=2))


def dataset_path(s: GuiSettings, root: Path = ROOT) -> Path:
    return Path(s.dataset_dir).expanduser() if s.dataset_dir else root / "output" / s.name


def demos_export_path(s: GuiSettings, root: Path = ROOT) -> Path:
    """HDF5 that step 4 writes for recorded demos: the demos with the Cosmos camera images."""
    return Path(s.demos_out) if s.demos_out else dataset_path(s, root) / "demos_cosmos.hdf5"


def training_path(s: GuiSettings, root: Path = ROOT) -> Path:
    """Folder that step 5 writes: the converted training dataset (e.g. LeRobot)."""
    return Path(s.convert_out) if s.convert_out else dataset_path(s, root) / "training_data"


def cosmos_path(s: GuiSettings, root: Path = ROOT) -> Path:
    return Path(s.cosmos_dir).expanduser() if s.cosmos_dir else root / "output" / f"{s.name}_cosmos"


def base_config_path(s: GuiSettings) -> Path:
    return Path(s.config_file) if s.config_file else DEFAULT_CONFIG


# --- configuration (YAML) --------------------------------------------------------------------------------------------


def format_value(value) -> str:
    """A config value as a one-line YAML string."""

    def plain(v):
        return [plain(x) for x in v] if isinstance(v, list | tuple) else v

    text = yaml.safe_dump(plain(value), default_flow_style=True, width=10000).strip()
    return text[: -len("...")].strip() if text.endswith("...") else text


def flatten(data: dict, prefix: str = "") -> dict:
    out = {}
    for key, value in data.items():
        if isinstance(value, dict):
            out.update(flatten(value, f"{prefix}{key}."))
        else:
            out[f"{prefix}{key}"] = value
    return out


def config_values(config_file: Path | None, overrides: dict[str, str] | None = None) -> dict:
    """Flat ``key -> value`` of a config file (None: defaults) with overrides; raises on unknown keys."""
    items = [f"{k}={v}" for k, v in (overrides or {}).items()]
    return flatten(load_config(config_file, items).to_dict())


def config_comments(path: Path = DEFAULT_CONFIG) -> dict[str, str]:
    """The ``# comment`` of each key in a YAML file such as ``configs/default.yaml``, keyed by dotted path."""
    comments, stack = {}, []
    for line in path.read_text().splitlines():
        m = re.match(r"^(\s*)([A-Za-z_][\w]*):(.*)$", line)
        if not m:
            continue
        indent, key, rest = len(m.group(1)), m.group(2), m.group(3)
        while stack and stack[-1][0] >= indent:
            stack.pop()
        full = ".".join([k for _, k in stack] + [key])
        if "#" in rest:
            comments[full] = rest.split("#", 1)[1].strip()
        if not rest.split("#", 1)[0].strip():  # a section
            stack.append((indent, key))
    return comments


def pipeline_keys(s: GuiSettings, root: Path = ROOT) -> dict[str, str]:
    """Config keys the Pipeline tab sets (they override the YAML and the Configuration tab)."""
    width = 1200 if s.resolution == "256" else 1248
    keys = {
        "cad.dir": s.cad_dir or "null",
        "cad.cache_dir": str(root / "cache" / "usd"),
        "cad.units": s.units,
        "cad.class_from": s.class_from,
        "output.dir": str(dataset_path(s, root)),
        "dataset.num_clips": str(s.clips),
        "dataset.frames_per_clip": str(s.frames),
        "dataset.resolution": f"[{width}, 720]",
        "randomization.clutter.enabled": str(s.clutter).lower(),
        "scene.parts_per_clip": f"[{s.parts_min}, {s.parts_max}]",
    }
    if s.clutter:
        keys["randomization.clutter.count"] = f"[{s.clutter_min}, {s.clutter_max}]"
    if s.use_scene:
        keys.update({"scene.scene_usd": s.scene_usd, "scene.scene_surface": s.scene_surface})
        keys["scene.scene_lights_only"] = str(s.scene_lights_only).lower()
        if s.scene_parts:
            keys.update({"scene.scene_parts": s.scene_parts, "scene.scene_parts_mode": s.scene_parts_mode})
        if s.scene_camera:
            a, e = abs(s.camera_azimuth), abs(s.camera_elevation)
            keys.update({
                "scene.scene_camera": s.scene_camera,
                "camera.azimuth_sweep": f"[{-a}, {a}]",
                "camera.elevation_sweep": f"[{-e}, {e}]",
                "camera.distance_sweep": "[0, 0]",
            })  # fmt: skip
    return keys


def write_run_config(s: GuiSettings, path: Path = RUN_CONFIG) -> Path:
    """Merge the base YAML and the Configuration tab overrides into ``path`` (checked: unknown keys raise)."""
    cfg = load_config(base_config_path(s), [f"{k}={v}" for k, v in s.config_overrides.items()])

    def plain(v):
        if isinstance(v, dict):
            return {k: plain(x) for k, x in v.items()}
        return [plain(x) for x in v] if isinstance(v, list | tuple) else v

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "# Written by the WorldSynth GUI: base config + Configuration tab overrides.\n"
        + yaml.safe_dump(plain(cfg.to_dict()), sort_keys=False, default_flow_style=None, width=120)
    )
    return path


# --- commands --------------------------------------------------------------------------------------------------------


def validate(s: GuiSettings, stages: tuple[str, ...] = STAGES, for_preview: bool = False) -> list[str]:
    """Problems that block running ``stages`` (or a preview), as messages for the user."""
    errors = []
    if "convert" in stages and not for_preview:
        if s.source != "demos":
            errors.append("Convert for training is for recorded demos (Source: Recorded demos).")
        elif not s.convert_script or not Path(s.convert_script).expanduser().is_file():
            errors.append("Choose the converter script (Training data).")
        if s.convert_python and not Path(s.convert_python).expanduser().is_file():
            errors.append(f"Converter Python not found: {s.convert_python}")
        if "{inputs}" not in s.convert_args.split():
            errors.append("Converter arguments need an {inputs} word.")
    if s.source == "demos" and (for_preview or "generate" in stages):
        if not s.demos_hdf5 or not Path(s.demos_hdf5).is_file():
            errors.append("Choose the HDF5 file with the recorded demos.")
        if not [c for c in s.demo_cameras.split(",") if c.strip()]:
            errors.append("Enter the camera observations of the demos, e.g. scene_cam,wrist_cam.")
    elif for_preview or "generate" in stages:
        scene_parts = s.use_scene and s.scene_parts
        if s.cad_dir and not Path(s.cad_dir).is_dir():
            errors.append(f"CAD folder not found: {s.cad_dir}")
        elif not s.cad_dir and not scene_parts:
            errors.append("Choose the folder with the CAD/STL files (or parts in the USD scene).")
        if s.use_scene:
            if not s.scene_usd or not Path(s.scene_usd).is_file():
                errors.append("Choose the USD scene file.")
            if not s.scene_surface:
                errors.append("Choose the surface in the scene to place the parts on.")
    if not for_preview:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", s.name or ""):
            errors.append("Run name: letters, digits, '_', '-' and '.' only.")
        if ("send" in stages or "fetch" in stages) and not (s.spark_host and s.spark_user):
            errors.append("Enter the Spark host and user.")
    if s.parts_min > s.parts_max or s.clutter_min > s.clutter_max:
        errors.append("A minimum is larger than its maximum.")
    if s.config_file and not Path(s.config_file).is_file():
        errors.append(f"Config file not found: {s.config_file}")
    try:
        config_values(base_config_path(s), s.config_overrides)
    except Exception as e:  # noqa: BLE001  (shown to the user)
        errors.append(f"Configuration: {e}")
    return errors


def common_overrides(s: GuiSettings) -> list[str]:
    """``--set`` overrides for the Pipeline tab values that run_pipeline.sh has no option for."""
    keys = pipeline_keys(s)
    names = ["scene.parts_per_clip", "randomization.clutter.count", "camera.azimuth_sweep", "camera.elevation_sweep",
             "camera.distance_sweep"]  # fmt: skip
    return [f"{k}={keys[k]}" for k in names if k in keys]


def scene_args(s: GuiSettings) -> list[str]:
    if not s.use_scene:
        return []
    args = ["--scene", s.scene_usd, "--scene-surface", s.scene_surface]
    if s.scene_camera:
        args += ["--scene-camera", s.scene_camera]
    if s.scene_lights_only:
        args.append("--scene-lights-only")
    if s.scene_parts:
        args += ["--scene-parts", s.scene_parts, "--scene-parts-mode", s.scene_parts_mode]
    return args


def look_files(s: GuiSettings, root: Path = ROOT) -> list[Path]:
    """Look prompt files a run uses: the chosen one, or all in ``configs/cosmos/prompts``."""
    if s.prompt == ALL_LOOKS:
        return sorted((root / "configs" / "cosmos" / "prompts").glob("*.json"))
    return [root / s.prompt]


def write_prompts(
    s: GuiSettings, edits: dict[str, str], out_dir: Path = PROMPT_DIR, root: Path = ROOT
) -> tuple[list[Path], Path]:
    """Write the look prompts and the negative prompt of a run, with the Prompts tab edits, to ``out_dir``.

    Args:
        edits: Prompt file (relative to the repo) -> edited JSON text. Files without edits are copied as they are.

    Returns:
        The look prompt files and the negative prompt file to pass to the pipeline. Raises ``ValueError`` if an
        edited prompt is not valid JSON.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.json"):
        old.unlink()

    def materialize(path: Path, name: str) -> Path:
        rel = str(path.relative_to(root)) if path.is_relative_to(root) else str(path)
        text = edits.get(rel, path.read_text())
        try:
            json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(f"{rel} is not valid JSON: {e}") from e
        out = out_dir / name
        out.write_text(text)
        return out

    looks = [materialize(p, p.name) for p in look_files(s, root)]
    negative = materialize(root / s.negative, "negative_prompt.json")
    return looks, negative


def pipeline_command(
    s: GuiSettings,
    config: Path,
    root: Path = ROOT,
    stages: tuple[str, ...] = STAGES,
    prompts: list[Path] | None = None,
    negative: Path | None = None,
) -> list[str]:
    """Command line of ``run_pipeline.sh`` for these settings (without secrets).

    Args:
        prompts, negative: Prompt files to use (e.g. from :func:`write_prompts`); default: the chosen files.
    """
    cmd = [str(root / "scripts" / "run_pipeline.sh"), "--name", s.name, "--config", str(config)]
    cmd += ["--stages", ",".join(stages)]
    if s.cad_dir:
        cmd += ["--cad", s.cad_dir]
    cmd += ["--dataset-dir", str(dataset_path(s, root)), "--cosmos-dir", str(cosmos_path(s, root))]
    cmd += ["--clips", str(s.clips), "--frames", str(s.frames), "--res", s.resolution, "--steps", str(s.steps)]
    cmd += ["--looks", str(s.looks), "--units", s.units, "--class-from", s.class_from]
    cmd += ["--spark", f"{s.spark_user}@{s.spark_host}", "--negative", str(negative or root / s.negative)]
    for p in prompts if prompts is not None else ([] if s.prompt == ALL_LOOKS else [root / s.prompt]):
        cmd += ["--prompts", str(p)]
    if not s.clutter:
        cmd.append("--no-clutter")
    if s.source == "demos":
        cmd += ["--demos", s.demos_hdf5, "--demo-cameras", ",".join(c.strip() for c in s.demo_cameras.split(","))]
        if s.max_demos > 0:
            cmd += ["--max-demos", str(s.max_demos)]
        if s.seg_names:
            cmd += ["--seg-names", s.seg_names]
        if s.demos_out:
            cmd += ["--demos-out", s.demos_out]
        if s.keep_missing:
            cmd.append("--keep-missing")
        if s.convert_script:
            cmd += ["--convert-script", s.convert_script, "--convert-args", s.convert_args]
            if s.convert_python:
                cmd += ["--convert-python", s.convert_python]
            if s.convert_out:
                cmd += ["--convert-out", s.convert_out]
            cmd += ["--cosmos-only"] * s.cosmos_only + ["--convert-overwrite"] * s.convert_overwrite
        return cmd
    cmd += scene_args(s)
    for item in common_overrides(s):
        cmd += ["--set", item]
    return cmd


def run_all_stages(s: GuiSettings) -> tuple[str, ...]:
    """Steps of "Run all steps": recorded demos with a converter also get converted for training."""
    return STAGES + ("convert",) if s.source == "demos" and s.convert_script else STAGES


def preview_command(s: GuiSettings, config: Path, out_dir: Path, isaaclab_python: str, root: Path = ROOT) -> list[str]:
    """One-frame render with ``generate.py`` to check the camera, surface and part scale."""
    keys = pipeline_keys(s, root)
    keys.update({"output.dir": str(out_dir), "dataset.num_clips": "1", "dataset.frames_per_clip": "1"})
    cmd = [isaaclab_python, str(root / "scripts" / "generate.py"), "--config", str(config), "output.overwrite=true"]
    return cmd + [f"{k}={v}" for k, v in keys.items()]


def ssh_env(s: GuiSettings, base: dict | None = None) -> dict:
    """Environment for processes that call ssh: the password goes to ssh through SSH_ASKPASS, nowhere else."""
    env = dict(os.environ if base is None else base)
    env["SPARK"] = f"{s.spark_user}@{s.spark_host}"
    if s.password:
        ASKPASS.parent.mkdir(parents=True, exist_ok=True)
        if not ASKPASS.exists():
            ASKPASS.write_text("#!/bin/sh\nprintf '%s\\n' \"$WS_SSH_PASSWORD\"\n")
        ASKPASS.chmod(0o700)
        env.update(WS_SSH_PASSWORD=s.password, SSH_ASKPASS=str(ASKPASS), SSH_ASKPASS_REQUIRE="force")
        env.setdefault("DISPLAY", ":0")
    return env


def isaaclab_python() -> str:
    """The Isaac Lab venv Python (Isaac Sim lives there); the GUI itself normally runs in it too."""
    root = Path(os.environ.get("ISAACLAB_DIR", ROOT.parent / "IsaacLab"))
    py = root / ".venv" / "bin" / "python"
    return str(py) if py.exists() else sys.executable


def demo_preview(s: GuiSettings) -> tuple[np.ndarray, str]:
    """First frame of the first demo with its segmentation labels, straight from the HDF5 (no Isaac Sim).

    Returns:
        The BGR image and a caption. Raises ``ValueError`` if the demo has no such camera.
    """
    import h5py

    from . import demos as dm
    from .viz import draw

    cam = s.demo_cameras.split(",")[0].strip()
    with h5py.File(s.demos_hdf5, "r") as f:
        name = sorted(f["data"].keys(), key=lambda n: (len(n), n))[0]
        obs = f["data"][name]["obs"]
        if cam not in obs:
            raise ValueError(f"{name} has no observation '{cam}'. It has: {', '.join(obs)}")
        rgb = obs[cam][0]
        seg = obs[f"{cam}_seg"][0].reshape(rgb.shape[:2]) if f"{cam}_seg" in obs else None
        n_demos, n_frames = len(f["data"]), len(obs[cam])
    img = rgb[..., ::-1].copy()
    extra = "no segmentation recorded"
    if seg is not None:
        names = dm.parse_seg_names(s.seg_names)
        ids = sorted(int(i) for i in np.unique(seg) if i)
        classes = {names.get(i, f"id_{i}") for i in ids}
        objs = dm._labels_from_segmentation(seg, names, {c: k + 1 for k, c in enumerate(sorted(classes))})
        img = draw(img, {"camera": {}, "objects": objs})
        extra = f"segmentation ids {ids}"
    return img, f"{name} / {cam}, frame 0 ({n_demos} demos, {n_frames} frames in this one; {extra})"


def connection_check_command(s: GuiSettings) -> list[str]:
    """ssh command that checks the Spark: login, Cosmos runner, free memory, running Cosmos jobs."""
    remote = (
        "echo connected to $(hostname); "
        "test -x ~/cosmos-work/run_transfer.sh && echo 'Cosmos runner found (~/cosmos-work/run_transfer.sh)' "
        "|| echo 'WARNING: ~/cosmos-work/run_transfer.sh is missing (copy scripts/cosmos/run_transfer.sh there)'; "
        'free -g | awk \'/Mem:/{print "free memory: " $7 " GB"}\'; '
        "pgrep -f '[c]osmos_framework.scripts.inference' > /dev/null && echo 'a Cosmos job is running now' || true"
    )
    return ["ssh", "-o", "ConnectTimeout=15", "-o", "StrictHostKeyChecking=accept-new",
            "-o", "ControlMaster=auto", "-o", f"ControlPath={Path.home() / '.ssh' / 'ws-%C'}",
            "-o", "ControlPersist=4h", f"{s.spark_user}@{s.spark_host}", remote]  # fmt: skip


def main():
    from .gui_qt import run

    sys.exit(run(sys.argv))


if __name__ == "__main__":
    main()
