"""WorldSynth desktop GUI: pick the CAD folder, an optional USD scene with its camera and surface, the dataset and
Cosmos settings, the output folders and the Spark login; adjust any config value on the Configuration tab (load a
YAML, override values); then preview a frame or run the whole pipeline (``scripts/run_pipeline.sh``).

Start it with the Isaac Lab Python (it needs ``pxr`` to read scenes): ``scripts/gui.sh``. It starts with the default
settings; ``--restore`` loads the settings of the last run (``--preview`` then renders a preview right away).

The SSH password is only handed to ``ssh`` through the environment of the pipeline process (via ``SSH_ASKPASS``).
It is never written to disk and not saved with the other settings.
"""

from __future__ import annotations

import base64
import contextlib
import dataclasses
import json
import os
import queue
import re
import shlex
import signal
import subprocess
import sys
import threading
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


# ---------------------------------------------------------------------------------------------------------------------
# Tk application


def main():  # noqa: C901  (UI construction is long but flat)
    import tkinter as tk
    from tkinter import filedialog, messagebox, scrolledtext, ttk

    from . import scene_info

    app = tk.Tk()
    app.title("WorldSynth")
    app.geometry("1500x940")
    style = ttk.Style(app)
    if "clam" in style.theme_names():
        style.theme_use("clam")

    # every start uses the defaults; "Restore last session" brings back the settings of the last run
    defaults = GuiSettings()
    v: dict[str, tk.Variable] = {}
    for f in dataclasses.fields(GuiSettings):
        val = getattr(defaults, f.name)
        if isinstance(val, dict):
            continue
        v[f.name] = (
            tk.BooleanVar(value=val)
            if isinstance(val, bool)
            else tk.IntVar(value=val)
            if isinstance(val, int)
            else tk.DoubleVar(value=val)
            if isinstance(val, float)
            else tk.StringVar(value=val)
        )

    state = {
        "proc": None,
        "queue": queue.Queue(),
        "after_run": None,
        "overrides": {},
        "comments": config_comments(),
        "select": defaults,  # camera/surface/parts to select once a scene is read
        "prompt_edits": {},  # prompt file (relative) -> edited text
        "prompt_shown": {},  # text box -> file it shows
    }

    def settings() -> GuiSettings:
        data = {"config_overrides": dict(state["overrides"])}
        for f in dataclasses.fields(GuiSettings):
            if f.name in data:
                continue
            try:
                data[f.name] = v[f.name].get()
            except tk.TclError:
                data[f.name] = getattr(GuiSettings(), f.name)
        cam = data["scene_camera"]
        data["scene_camera"] = "" if cam == RANDOM_CAMERA else cam.split("   ")[0]
        data["scene_surface"] = data["scene_surface"].split("   ")[0]
        parts = data["scene_parts"]
        data["scene_parts"] = "" if parts == NO_SCENE_PARTS else parts.split("   ")[0]
        return GuiSettings(**data)

    def apply_settings(s: GuiSettings):
        for f in dataclasses.fields(GuiSettings):
            if f.name in v and f.name != "password":  # a typed password stays
                v[f.name].set(getattr(s, f.name))
        state["overrides"] = dict(s.config_overrides)
        state["select"] = s
        update_source_state()
        if s.use_scene and s.scene_usd:
            load_scene()
        refresh_config()

    # --- layout: tabs on the left, buttons + preview + log on the right -----------------------------------------------
    notebook = ttk.Notebook(app, width=560)
    notebook.pack(side="left", fill="y", padx=(8, 0), pady=8)
    right = ttk.Frame(app, padding=8)
    right.pack(side="left", fill="both", expand=True)

    # pipeline tab: scrollable column of sections
    pipe_outer = ttk.Frame(notebook)
    notebook.add(pipe_outer, text="Pipeline")
    canvas = tk.Canvas(pipe_outer, highlightthickness=0, width=545)
    scroll = ttk.Scrollbar(pipe_outer, orient="vertical", command=canvas.yview)
    left = ttk.Frame(canvas, padding=(4, 4, 8, 4))
    left.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
    window_id = canvas.create_window((0, 0), window=left, anchor="nw")
    canvas.bind("<Configure>", lambda e: canvas.itemconfigure(window_id, width=e.width))  # content follows tab width
    canvas.configure(yscrollcommand=scroll.set)
    canvas.pack(side="left", fill="both", expand=True)
    scroll.pack(side="right", fill="y")

    def wheel(event):
        if notebook.index("current") == 0:
            canvas.yview_scroll(-1 if (event.num == 4 or event.delta > 0) else 1, "units")

    for seq in ("<Button-4>", "<Button-5>", "<MouseWheel>"):
        pipe_outer.bind_all(seq, wheel, add="+")

    def section(title):
        fr = ttk.LabelFrame(left, text=title, padding=6)
        fr.pack(fill="x", pady=4)
        fr.columnconfigure(1, weight=1)
        return fr

    def row(fr, r, label, widget, extra=None):
        ttk.Label(fr, text=label).grid(row=r, column=0, sticky="w", padx=(0, 6), pady=2)
        widget.grid(row=r, column=1, sticky="ew", pady=2)
        if extra is not None:
            extra.grid(row=r, column=2, padx=(4, 0), pady=2)

    def browse_dir(var):
        d = filedialog.askdirectory(initialdir=var.get() or str(ROOT))
        if d:
            var.set(d)

    def hint(fr, r, text):
        ttk.Label(fr, text=text, foreground="#555", wraplength=400).grid(row=r, column=1, columnspan=2, sticky="w")

    session = ttk.Frame(left)
    session.pack(fill="x", pady=(0, 2))
    ttk.Button(session, text="Restore last session", command=lambda: restore_session()).pack(side="left")
    ttk.Button(session, text="Defaults", command=lambda: apply_settings(GuiSettings())).pack(side="left", padx=6)
    ttk.Label(session, text="(starts with defaults)", foreground="#555").pack(side="left")

    # source: render CAD parts, or use recorded robot demos
    fr = section("Source")
    src_mode = ttk.Frame(fr)
    ttk.Radiobutton(src_mode, text="Render CAD parts in Isaac Sim", variable=v["source"], value="cad",
                    command=lambda: update_source_state()).pack(side="left")  # fmt: skip
    ttk.Radiobutton(src_mode, text="Recorded demos (HDF5)", variable=v["source"], value="demos",
                    command=lambda: update_source_state()).pack(side="left", padx=8)  # fmt: skip
    src_mode.grid(row=0, column=0, columnspan=3, sticky="w")

    def browse_file(var, save=False, kind=(("HDF5", "*.hdf5 *.h5"), ("All files", "*"))):
        kind = list(kind)
        start_dir = str(Path(var.get()).parent) if var.get() else str(ROOT)
        f = (filedialog.asksaveasfilename(initialdir=start_dir, defaultextension=".hdf5", filetypes=kind) if save
             else filedialog.askopenfilename(initialdir=start_dir, filetypes=kind))  # fmt: skip
        if f:
            var.set(f)

    demo_frame = ttk.Frame(fr)
    demo_frame.grid(row=1, column=0, columnspan=3, sticky="ew")
    demo_frame.columnconfigure(1, weight=1)
    row(demo_frame, 0, "Demos file", ttk.Entry(demo_frame, textvariable=v["demos_hdf5"], width=30),
        ttk.Button(demo_frame, text="Browse...", command=lambda: browse_file(v["demos_hdf5"])))  # fmt: skip
    row(demo_frame, 1, "Cameras", ttk.Entry(demo_frame, textvariable=v["demo_cameras"], width=30))
    row(demo_frame, 2, "Max demos", ttk.Spinbox(demo_frame, from_=0, to=100000, textvariable=v["max_demos"], width=8))
    row(demo_frame, 3, "Seg classes", ttk.Entry(demo_frame, textvariable=v["seg_names"], width=30))
    row(demo_frame, 4, "Output HDF5", ttk.Entry(demo_frame, textvariable=v["demos_out"], width=30),
        ttk.Button(demo_frame, text="Browse...", command=lambda: browse_file(v["demos_out"], save=True)))  # fmt: skip
    hint(demo_frame, 5, "Max demos 0 = all. Seg classes name the ids of <camera>_seg, e.g. "
                        "1=robot,2=table,3=ground,4-5=bin,10-15=soup_can. Output HDF5: copy with the Cosmos images "
                        "(empty: <dataset folder>/demos_cosmos.hdf5)")  # fmt: skip

    # training data: original + Cosmos demos through the robot project's converter (e.g. LeRobot for GR00T)
    fr = section("Training data (demos)")
    train_frame = fr
    py_kind = (("Python", "*.py"), ("All files", "*"))
    any_kind = (("All files", "*"),)
    row(fr, 0, "Converter", ttk.Entry(fr, textvariable=v["convert_script"], width=30),
        ttk.Button(fr, text="Browse...", command=lambda: browse_file(v["convert_script"], kind=py_kind)))  # fmt: skip
    row(fr, 1, "Python", ttk.Entry(fr, textvariable=v["convert_python"], width=30),
        ttk.Button(fr, text="Browse...", command=lambda: browse_file(v["convert_python"], kind=any_kind)))  # fmt: skip
    row(fr, 2, "Arguments", ttk.Entry(fr, textvariable=v["convert_args"], width=30))
    row(fr, 3, "Output folder", ttk.Entry(fr, textvariable=v["convert_out"], width=30),
        ttk.Button(fr, text="Browse...", command=lambda: browse_dir(v["convert_out"])))  # fmt: skip
    opts = ttk.Frame(fr)
    ttk.Checkbutton(opts, text="Cosmos demos only", variable=v["cosmos_only"]).pack(side="left")
    ttk.Checkbutton(opts, text="Overwrite output", variable=v["convert_overwrite"]).pack(side="left", padx=8)
    ttk.Checkbutton(opts, text="Export partial demos", variable=v["keep_missing"]).pack(side="left")
    opts.grid(row=4, column=1, columnspan=2, sticky="w")
    hint(fr, 5, "Step 5 converts the original demos and the Cosmos demos (one HDF5 per look) with the robot "
                "project's converter, e.g. kuka/scripts/gr00t/convert_to_gr00t_lerobot.py with "
                "~/venvs/convert/bin/python for GR00T (LeRobot). {inputs} = the HDF5 files, {output} = the output "
                "folder (empty: <dataset folder>/training_data). Python empty = Isaac Lab's. Export partial demos: "
                "also write demos where only some cameras have Cosmos frames.")  # fmt: skip

    # parts
    fr = section("Parts (CAD / STL folder)")
    parts_frame = fr
    row(fr, 0, "Folder", ttk.Entry(fr, textvariable=v["cad_dir"], width=30),
        ttk.Button(fr, text="Browse...", command=lambda: browse_dir(v["cad_dir"])))  # fmt: skip
    row(fr, 1, "Units", ttk.Combobox(fr, textvariable=v["units"], values=["auto", "mm", "m", "cm", "inch"],
                                     state="readonly", width=10))  # fmt: skip
    row(fr, 2, "Class per", ttk.Combobox(fr, textvariable=v["class_from"], values=["file", "folder"],
                                         state="readonly", width=10))  # fmt: skip
    hint(fr, 3, "optional when the USD scene provides the parts (Scene: Parts in scene)")

    # scene
    fr = section("Scene")
    scene_frame = fr
    mode = ttk.Frame(fr)
    ttk.Radiobutton(mode, text="Built-in workcell", variable=v["use_scene"], value=False,
                    command=lambda: update_scene_state()).pack(side="left")  # fmt: skip
    ttk.Radiobutton(mode, text="USD scene", variable=v["use_scene"], value=True,
                    command=lambda: update_scene_state()).pack(side="left", padx=8)  # fmt: skip
    mode.grid(row=0, column=0, columnspan=3, sticky="w")
    scene_entry = ttk.Entry(fr, textvariable=v["scene_usd"], width=30)

    def browse_scene():
        f = filedialog.askopenfilename(
            initialdir=str(Path(v["scene_usd"].get()).parent) if v["scene_usd"].get() else str(ROOT),
            filetypes=[("USD", "*.usd *.usda *.usdc *.usdz"), ("All files", "*")],
        )
        if f:
            v["scene_usd"].set(f)
            state["select"] = GuiSettings()
            load_scene()

    scene_btn = ttk.Button(fr, text="Open...", command=browse_scene)
    row(fr, 1, "USD file", scene_entry, scene_btn)
    scene_info_label = ttk.Label(fr, text="", foreground="#555")
    scene_info_label.grid(row=2, column=1, columnspan=2, sticky="w")
    camera_box = ttk.Combobox(fr, textvariable=v["scene_camera"], state="readonly", width=30)
    row(fr, 3, "Camera", camera_box)
    surface_box = ttk.Combobox(fr, textvariable=v["scene_surface"], state="readonly", width=30)
    row(fr, 4, "Place parts on", surface_box)
    parts_box = ttk.Combobox(fr, textvariable=v["scene_parts"], state="readonly", width=30)
    row(fr, 5, "Parts in scene", parts_box)
    parts_mode = ttk.Frame(fr)
    parts_mode_widgets = [
        ttk.Radiobutton(parts_mode, text="place randomly", variable=v["scene_parts_mode"], value="place"),
        ttk.Radiobutton(parts_mode, text="keep as in scene", variable=v["scene_parts_mode"], value="keep"),
    ]
    for w in parts_mode_widgets:
        w.pack(side="left", padx=(0, 8))
    row(fr, 6, "Scene parts", parts_mode)
    motion = ttk.Frame(fr)
    ttk.Label(motion, text="± azimuth").pack(side="left")
    ttk.Spinbox(motion, from_=0, to=90, increment=1, textvariable=v["camera_azimuth"], width=5).pack(side="left")
    ttk.Label(motion, text="°   ± elevation").pack(side="left")
    ttk.Spinbox(motion, from_=0, to=45, increment=1, textvariable=v["camera_elevation"], width=5).pack(side="left")
    ttk.Label(motion, text="°").pack(side="left")
    row(fr, 7, "Camera motion", motion)
    lights_cb = ttk.Checkbutton(fr, text="Use only the scene's own lights", variable=v["scene_lights_only"])
    lights_cb.grid(row=8, column=1, sticky="w")
    scene_widgets = [scene_entry, scene_btn, camera_box, surface_box, parts_box, lights_cb]
    scene_widgets += list(motion.winfo_children()) + parts_mode_widgets

    def update_scene_state():
        on = v["use_scene"].get()
        for w in scene_widgets:
            with contextlib.suppress(tk.TclError):
                w.configure(state=("readonly" if isinstance(w, ttk.Combobox) else "normal") if on else "disabled")

    def set_state(widget, enabled: bool):
        for child in widget.winfo_children():
            with contextlib.suppress(tk.TclError):
                ro = isinstance(child, ttk.Combobox)
                child.configure(state=("readonly" if ro else "normal") if enabled else "disabled")
            set_state(child, enabled)

    def update_source_state():
        demos_on = v["source"].get() == "demos"
        set_state(demo_frame, demos_on)
        set_state(train_frame, demos_on)
        step_btns[4].configure(state="normal" if demos_on else "disabled")
        for frame in (parts_frame, scene_frame, dataset_frame):
            set_state(frame, not demos_on)
        if not demos_on:
            update_scene_state()
        step_btns[0].configure(text="1 Import demos" if demos_on else "1 Generate (Isaac Sim)")

    def load_scene():
        path = v["scene_usd"].get()
        if not path or not Path(path).is_file():
            return
        scene_info_label.configure(text="reading scene...")

        def work():
            try:
                state["queue"].put(("scene", scene_info.inspect(path), None))
            except Exception as e:  # noqa: BLE001  (shown to the user)
                state["queue"].put(("scene_error", str(e), None))

        threading.Thread(target=work, daemon=True).start()

    def show_scene(info):
        sel = state["select"]
        cams = [RANDOM_CAMERA] + [f"{c.path}   (at {c.position}, f={c.focal_length:g})" for c in info.cameras]
        surfs = [f"{s.path}   (top {s.top_z:.3f} m, {s.size[0]:.2f} × {s.size[1]:.2f} m)" for s in info.surfaces]
        groups = [NO_SCENE_PARTS] + [
            f"{g.path}   ({len(g.children)}: {', '.join(g.children[:4])}{', ...' if len(g.children) > 4 else ''})"
            for g in info.groups
        ]
        camera_box.configure(values=cams)
        surface_box.configure(values=surfs)
        parts_box.configure(values=groups)

        def pick(options, saved, fallback):
            return next((o for o in options if saved and o.split("   ")[0] == saved), fallback)

        v["scene_camera"].set(pick(cams, sel.scene_camera, cams[1] if len(cams) > 1 else cams[0]))
        # default surface: the second largest (the largest is usually the floor)
        v["scene_surface"].set(
            pick(surfs, sel.scene_surface, surfs[1] if len(surfs) > 1 else surfs[0] if surfs else "")
        )
        v["scene_parts"].set(pick(groups, sel.scene_parts, groups[0]))
        scene_info_label.configure(
            text=f"{len(info.cameras)} cameras, {len(info.surfaces)} surfaces; file units "
            f"{info.meters_per_unit:g} m, {info.up_axis}-up (converted to m, z-up)"
        )

    # output
    fr = section("Output")
    row(fr, 0, "Run name", ttk.Entry(fr, textvariable=v["name"], width=24))
    row(fr, 1, "Dataset folder", ttk.Entry(fr, textvariable=v["dataset_dir"], width=30),
        ttk.Button(fr, text="Browse...", command=lambda: browse_dir(v["dataset_dir"])))  # fmt: skip
    hint(fr, 2, "Isaac Sim output; empty: output/<run name>")
    row(fr, 3, "Cosmos folder", ttk.Entry(fr, textvariable=v["cosmos_dir"], width=30),
        ttk.Button(fr, text="Browse...", command=lambda: browse_dir(v["cosmos_dir"])))  # fmt: skip
    hint(
        fr, 4, "Cosmos input videos and batch; results copied back to <folder>/results; empty: output/<run name>_cosmos"
    )

    # dataset
    fr = section("Dataset")
    dataset_frame = fr
    nums = ttk.Frame(fr)
    for label, key, hi in [("clips", "clips", 10000), ("frames/clip", "frames", 241)]:
        ttk.Spinbox(nums, from_=1, to=hi, textvariable=v[key], width=6).pack(side="left")
        ttk.Label(nums, text=f" {label}   ").pack(side="left")
    row(fr, 0, "Size", nums)
    parts = ttk.Frame(fr)
    ttk.Spinbox(parts, from_=1, to=50, textvariable=v["parts_min"], width=4).pack(side="left")
    ttk.Label(parts, text=" to ").pack(side="left")
    ttk.Spinbox(parts, from_=1, to=50, textvariable=v["parts_max"], width=4).pack(side="left")
    row(fr, 1, "Parts per clip", parts)
    clut = ttk.Frame(fr)
    ttk.Checkbutton(clut, text="on,", variable=v["clutter"]).pack(side="left")
    ttk.Spinbox(clut, from_=0, to=50, textvariable=v["clutter_min"], width=4).pack(side="left")
    ttk.Label(clut, text=" to ").pack(side="left")
    ttk.Spinbox(clut, from_=0, to=50, textvariable=v["clutter_max"], width=4).pack(side="left")
    ttk.Label(clut, text=" items").pack(side="left")
    row(fr, 2, "Clutter", clut)
    hint(fr, 3, "all other settings: Configuration tab")

    # cosmos
    fr = section("Cosmos Transfer")
    look_options = [str(p.relative_to(ROOT)) for p in sorted((ROOT / "configs" / "cosmos").rglob("*.json"))
                    if "negative" not in p.name]  # fmt: skip
    negatives = [str(p.relative_to(ROOT)) for p in sorted((ROOT / "configs" / "cosmos").rglob("negative*.json"))]
    res = ttk.Frame(fr)
    ttk.Radiobutton(res, text="256 (fast, ~3 min/clip)", variable=v["resolution"], value="256").pack(side="left")
    ttk.Radiobutton(res, text="480 (sharp, ~36 min/clip)", variable=v["resolution"], value="480").pack(side="left")
    row(fr, 0, "Resolution", res)
    sl = ttk.Frame(fr)
    ttk.Spinbox(sl, from_=10, to=100, textvariable=v["steps"], width=5).pack(side="left")
    ttk.Label(sl, text=" steps    ").pack(side="left")
    ttk.Spinbox(sl, from_=1, to=10, textvariable=v["looks"], width=4).pack(side="left")
    ttk.Label(sl, text=" looks per clip").pack(side="left")
    row(fr, 1, "Sampling", sl)
    row(fr, 2, "Looks used", ttk.Combobox(fr, textvariable=v["prompt"], values=[ALL_LOOKS] + look_options,
                                          state="readonly", width=30))  # fmt: skip
    row(fr, 3, "Negative", ttk.Combobox(fr, textvariable=v["negative"], values=negatives, state="readonly", width=30))
    hint(fr, 4, "edit the prompt texts on the Prompts tab")

    # spark
    fr = section("Spark (runs Cosmos)")
    row(fr, 0, "Host / IP", ttk.Entry(fr, textvariable=v["spark_host"], width=24))
    row(fr, 1, "User", ttk.Entry(fr, textvariable=v["spark_user"], width=24))
    row(fr, 2, "Password", ttk.Entry(fr, textvariable=v["password"], show="*", width=24),
        ttk.Button(fr, text="Test connection", command=lambda: test_connection()))  # fmt: skip
    hint(fr, 3, "not saved; leave empty for SSH keys")

    # --- configuration tab -------------------------------------------------------------------------------------------
    cfg_tab = ttk.Frame(notebook, padding=6)
    notebook.add(cfg_tab, text="Configuration")
    top = ttk.Frame(cfg_tab)
    top.pack(fill="x")
    ttk.Label(top, text="Base config").pack(side="left")
    base_label = ttk.Label(top, text="", foreground="#333")
    base_label.pack(side="left", padx=6)
    cfg_btns = ttk.Frame(cfg_tab)
    cfg_btns.pack(fill="x", pady=(4, 4))
    tree_box = ttk.Frame(cfg_tab)
    tree_box.pack(fill="both", expand=True)
    tree = ttk.Treeview(tree_box, columns=("value", "source"), height=24)
    tree_scroll = ttk.Scrollbar(tree_box, orient="vertical", command=tree.yview)
    tree.configure(yscrollcommand=tree_scroll.set)
    tree.pack(side="left", fill="both", expand=True)
    tree_scroll.pack(side="right", fill="y")
    tree.heading("#0", text="Setting")
    tree.heading("value", text="Value (double-click to edit)")
    tree.heading("source", text="From")
    tree.column("#0", width=230)
    tree.column("value", width=200)
    tree.column("source", width=100)
    tree.tag_configure("override", foreground="#0b5cad", font=("TkDefaultFont", 9, "bold"))
    tree.tag_configure("file", foreground="#1b7a1b")
    tree.tag_configure("pipeline", foreground="#888")
    desc = ttk.Label(cfg_tab, text="Select a setting to see what it does.", foreground="#444", wraplength=520,
                     justify="left")  # fmt: skip
    desc.pack(fill="x", pady=(4, 0))
    legend = ("blue: changed here   green: from the loaded YAML   grey: set on the Pipeline tab (it wins)   "
              "black: default")  # fmt: skip
    ttk.Label(cfg_tab, text=legend, foreground="#666", wraplength=520).pack(fill="x")

    def refresh_config():
        s = settings()
        base = base_config_path(s)
        base_label.configure(text=str(base.relative_to(ROOT)) if base.is_relative_to(ROOT) else str(base))
        try:
            defaults_flat = config_values(None)
            from_file = config_values(base)
            current = config_values(base, state["overrides"])
        except Exception as e:  # noqa: BLE001  (shown to the user)
            messagebox.showerror("WorldSynth", f"Configuration error: {e}")
            return
        locked = pipeline_keys(s)

        def nodes(parent=""):
            for i in tree.get_children(parent):
                yield i
                yield from nodes(i)

        all_nodes = [i for i in nodes() if tree.get_children(i)]
        open_items = {i for i in all_nodes if tree.item(i, "open")} if all_nodes else None
        selected = tree.selection()
        tree.delete(*tree.get_children(""))
        for key, value in current.items():
            parent = ""
            names = key.split(".")
            for depth in range(1, len(names)):
                node = ".".join(names[:depth])
                if not tree.exists(node):
                    tree.insert(parent, "end", iid=node, text=names[depth - 1],
                                open=open_items is None or node in open_items)  # fmt: skip
                parent = node
            if key in locked:
                text, src, tag = locked[key], "Pipeline tab", "pipeline"
            elif key in state["overrides"]:
                text, src, tag = format_value(value), "changed here", "override"
            elif from_file[key] != defaults_flat[key]:
                text, src, tag = format_value(value), "YAML file", "file"
            else:
                text, src, tag = format_value(value), "default", ""
            tree.insert(parent, "end", iid=key, text=names[-1], values=(text, src), tags=(tag,))
        if selected and tree.exists(selected[0]):
            tree.selection_set(selected[0])
            tree.see(selected[0])
        else:
            tree.yview_moveto(0)

    def on_select(_event=None):
        sel = tree.selection()
        if sel:
            desc.configure(text=f"{sel[0]}: {state['comments'].get(sel[0], '')}".rstrip(": "))

    def edit_cell(event):
        key = tree.identify_row(event.y)
        if not key or tree.get_children(key):
            return
        if "pipeline" in tree.item(key, "tags"):
            messagebox.showinfo("WorldSynth", f"{key} is set on the Pipeline tab; change it there.")
            return
        x, y, w, h = tree.bbox(key, "value")
        entry = ttk.Entry(tree)
        entry.insert(0, tree.set(key, "value"))
        entry.place(x=x, y=y, width=w, height=h)
        entry.focus_set()
        entry.select_range(0, "end")
        done = {"flag": False}

        def commit(_e=None):
            if done["flag"]:
                return
            done["flag"] = True
            text = entry.get().strip()
            entry.destroy()
            try:
                trial = dict(state["overrides"], **{key: text})
                value = config_values(base_config_path(settings()), trial)[key]
                base_value = config_values(base_config_path(settings()))[key]
            except Exception as e:  # noqa: BLE001  (shown to the user)
                messagebox.showerror("WorldSynth", f"{key}: {e}")
                return
            if value == base_value:
                state["overrides"].pop(key, None)
            else:
                state["overrides"][key] = text
            refresh_config()

        def cancel(_e=None):
            done["flag"] = True
            entry.destroy()

        entry.bind("<Return>", commit)
        entry.bind("<KP_Enter>", commit)
        entry.bind("<Escape>", cancel)
        entry.bind("<FocusOut>", commit)

    def revert_selected():
        for key in tree.selection():
            state["overrides"].pop(key, None)
        refresh_config()

    def load_yaml():
        f = filedialog.askopenfilename(
            initialdir=str(ROOT / "configs"), filetypes=[("YAML", "*.yaml *.yml"), ("All files", "*")]
        )
        if not f:
            return
        try:
            config_values(Path(f))
        except Exception as e:  # noqa: BLE001  (shown to the user)
            messagebox.showerror("WorldSynth", f"Cannot use {f}: {e}")
            return
        v["config_file"].set(f)
        if state["overrides"] and not messagebox.askyesno("WorldSynth", "Keep the values changed here on top of it?"):
            state["overrides"].clear()
        refresh_config()

    def save_yaml():
        f = filedialog.asksaveasfilename(
            initialdir=str(ROOT / "configs"), defaultextension=".yaml", filetypes=[("YAML", "*.yaml")]
        )
        if not f:
            return
        try:
            write_run_config(settings(), Path(f))
        except Exception as e:  # noqa: BLE001  (shown to the user)
            messagebox.showerror("WorldSynth", str(e))
            return
        v["config_file"].set(f)
        state["overrides"].clear()
        refresh_config()
        write(f"saved configuration to {f} (now the base config)")

    def reset_config():
        v["config_file"].set("")
        state["overrides"].clear()
        refresh_config()

    for text, cmd in [("Load YAML...", load_yaml), ("Save as YAML...", save_yaml), ("Revert selected", revert_selected),
                      ("Reset to defaults", reset_config)]:  # fmt: skip
        ttk.Button(cfg_btns, text=text, command=cmd).pack(side="left", padx=(0, 6))
    tree.bind("<<TreeviewSelect>>", on_select)
    tree.bind("<Double-1>", edit_cell)

    # --- prompts tab -------------------------------------------------------------------------------------------------
    prompt_tab = ttk.Frame(notebook, padding=6)
    notebook.add(prompt_tab, text="Prompts")
    ttk.Label(prompt_tab, wraplength=530, justify="left", foreground="#444", text=(
        "Edit the Cosmos prompts (JSON). {parts} becomes your class names. Edits are used by the next run even if not "
        "saved; Save writes them to the file. Which looks a run uses: Pipeline tab > Cosmos > Looks used."
    )).pack(fill="x")  # fmt: skip
    first_look = next((o for o in look_options if o.startswith("configs/cosmos/prompts/")), None)
    look_pick_var = tk.StringVar(value=first_look or (look_options[0] if look_options else ""))
    prompt_widgets = {}

    def prompt_editor(title, pick_var, options, height):
        box = ttk.LabelFrame(prompt_tab, text=title, padding=6)
        box.pack(fill="both", expand=True, pady=4)
        head = ttk.Frame(box)
        head.pack(fill="x")
        pick = ttk.Combobox(head, textvariable=pick_var, values=options, state="readonly", width=46)
        pick.pack(side="left")
        status_lbl = ttk.Label(head, text="", foreground="#0b5cad")
        status_lbl.pack(side="left", padx=6)
        text = scrolledtext.ScrolledText(box, height=height, font=("TkFixedFont", 9), wrap="word", undo=True)
        text.pack(fill="both", expand=True, pady=4)
        btns = ttk.Frame(box)
        btns.pack(fill="x")
        prompt_widgets[text] = (pick_var, status_lbl)
        pick.bind("<<ComboboxSelected>>", lambda _e: show_prompt(text))
        text.bind("<<Modified>>", lambda _e: on_prompt_modified(text))
        for label, fn in [("Revert", lambda: revert_prompt(text)), ("Save", lambda: save_prompt(text, False)),
                          ("Save as...", lambda: save_prompt(text, True))]:  # fmt: skip
            ttk.Button(btns, text=label, command=fn).pack(side="left", padx=(0, 6))
        return text

    def stash_prompt(text):
        """Keep the text box content as an edit of the file it shows (or drop the edit if unchanged)."""
        rel = state["prompt_shown"].get(text)
        if not rel:
            return
        content = text.get("1.0", "end-1c")
        if content != (ROOT / rel).read_text():
            state["prompt_edits"][rel] = content
        else:
            state["prompt_edits"].pop(rel, None)

    def show_prompt(text):
        stash_prompt(text)
        pick_var, status_lbl = prompt_widgets[text]
        rel = pick_var.get()
        if not rel:
            return
        content = state["prompt_edits"].get(rel) or (ROOT / rel).read_text()
        text.delete("1.0", "end")
        text.insert("1.0", content)
        text.edit_modified(False)
        text.edit_reset()
        state["prompt_shown"][text] = rel
        status_lbl.configure(text="edited, not saved" if rel in state["prompt_edits"] else "")

    def on_prompt_modified(text):
        if text.edit_modified():
            prompt_widgets[text][1].configure(text="edited, not saved")
            text.edit_modified(False)

    def revert_prompt(text):
        rel = state["prompt_shown"].get(text)
        if rel:
            state["prompt_edits"].pop(rel, None)
            state["prompt_shown"].pop(text, None)
            show_prompt(text)

    def save_prompt(text, as_new):
        rel = state["prompt_shown"].get(text)
        content = text.get("1.0", "end-1c")
        try:
            json.loads(content)
        except json.JSONDecodeError as e:
            messagebox.showerror("WorldSynth", f"Not valid JSON: {e}")
            return
        target = ROOT / rel
        if as_new:
            f = filedialog.asksaveasfilename(initialdir=str(target.parent), defaultextension=".json",
                                             filetypes=[("JSON", "*.json")])  # fmt: skip
            if not f:
                return
            target = Path(f)
        target.write_text(content)
        state["prompt_edits"].pop(rel, None)
        if target.is_relative_to(ROOT / "configs" / "cosmos"):
            new_rel = str(target.relative_to(ROOT))
            for combo_values in (look_options, negatives):
                if ("negative" in target.name) == (combo_values is negatives) and new_rel not in combo_values:
                    combo_values.append(new_rel)
            prompt_widgets[text][0].set(new_rel)
        state["prompt_shown"].pop(text, None)
        show_prompt(text)
        write(f"saved prompt {target}")

    look_text = prompt_editor("Look prompt (positive)", look_pick_var, look_options, 14)
    neg_text = prompt_editor("Negative prompt", v["negative"], negatives, 9)

    def all_prompt_edits() -> dict[str, str]:
        for text in (look_text, neg_text):
            stash_prompt(text)
        return dict(state["prompt_edits"])

    def on_tab(_e=None):
        tab = notebook.index("current")
        if tab == 1:
            refresh_config()
        elif tab == 2:
            if v["prompt"].get() != ALL_LOOKS and v["prompt"].get() != look_pick_var.get():
                look_pick_var.set(v["prompt"].get())
            for text in (look_text, neg_text):
                if state["prompt_shown"].get(text) != prompt_widgets[text][0].get():
                    show_prompt(text)

    # --- videos tab ------------------------------------------------------------------------------------------------
    from . import videos as vid

    vid_tab = ttk.Frame(notebook, padding=6)
    notebook.add(vid_tab, text="Videos")
    vf = ttk.Frame(vid_tab)
    vf.pack(fill="x")
    vf.columnconfigure(1, weight=1)
    vid_data = tk.StringVar()
    vid_cosmos = tk.StringVar()
    for r, (label, var) in enumerate([("Dataset", vid_data), ("Cosmos folder", vid_cosmos)]):
        ttk.Label(vf, text=label).grid(row=r, column=0, sticky="w", padx=(0, 6), pady=2)
        ttk.Entry(vf, textvariable=var, width=34).grid(row=r, column=1, sticky="ew", pady=2)
        ttk.Button(vf, text="Browse...", command=lambda var=var: (browse_dir(var), refresh_videos())).grid(
            row=r, column=2, padx=(4, 0)
        )
    vb = ttk.Frame(vid_tab)
    vb.pack(fill="x", pady=4)
    ttk.Button(vb, text="Use the run's folders", command=lambda: use_run_folders()).pack(side="left")
    ttk.Button(vb, text="Refresh", command=lambda: refresh_videos()).pack(side="left", padx=6)
    vid_box = ttk.Frame(vid_tab)
    vid_box.pack(fill="both", expand=True)
    vid_tree = ttk.Treeview(vid_box, show="tree", height=18)
    vid_scroll = ttk.Scrollbar(vid_box, orient="vertical", command=vid_tree.yview)
    vid_tree.configure(yscrollcommand=vid_scroll.set)
    vid_tree.pack(side="left", fill="both", expand=True)
    vid_scroll.pack(side="right", fill="y")
    vo = ttk.Frame(vid_tab)
    vo.pack(fill="x", pady=(6, 2))
    ttk.Label(vo, text="View").pack(side="left")
    vid_view = tk.StringVar(value="video")
    view_box = ttk.Combobox(vo, textvariable=vid_view, state="readonly", width=14)
    view_box.pack(side="left", padx=(4, 12))
    ttk.Label(vo, text="Speed").pack(side="left")
    vid_speed = tk.StringVar(value="1x")
    ttk.Combobox(vo, textvariable=vid_speed, values=["0.25x", "0.5x", "1x", "2x"], state="readonly",
                 width=6).pack(side="left", padx=4)  # fmt: skip
    vc = ttk.Frame(vid_tab)
    vc.pack(fill="x", pady=2)
    ttk.Button(vc, text="<", width=3, command=lambda: step_video(-1)).pack(side="left")
    play_btn = ttk.Button(vc, text="Play", width=6, command=lambda: toggle_play())
    play_btn.pack(side="left", padx=4)
    ttk.Button(vc, text=">", width=3, command=lambda: step_video(1)).pack(side="left")
    vid_pos = tk.IntVar(value=0)
    vid_slider = ttk.Scale(vc, from_=0, to=0, orient="horizontal", command=lambda x: seek_video(int(float(x))))
    vid_slider.pack(side="left", fill="x", expand=True, padx=6)
    vid_label = ttk.Label(vc, text="0 / 0", width=10)
    vid_label.pack(side="left")
    ttk.Label(vid_tab, foreground="#555", wraplength=530, justify="left", text=(
        "Isaac Sim clips play from their frames: view them as rgb, with labels, depth, segmentation or instance "
        "masks. The video plays in the large area on the right."
    )).pack(fill="x", pady=(4, 0))  # fmt: skip
    player = {"sources": {}, "src": None, "n": 0, "playing": False, "after": None, "cache": {}}

    def use_run_folders():
        s = settings()
        vid_data.set(str(dataset_path(s)))
        vid_cosmos.set(str(cosmos_path(s)))
        refresh_videos()

    def refresh_videos():
        stop_video()
        vid_tree.delete(*vid_tree.get_children(""))
        data = Path(vid_data.get()) if vid_data.get() else None
        cos = Path(vid_cosmos.get()) if vid_cosmos.get() else None
        sources = vid.find_sources(data, cos)
        player["sources"] = {src.key: src for src in sources}
        for src in sources:
            if not vid_tree.exists(src.group):
                vid_tree.insert("", "end", iid=src.group, text=src.group, open=True)
            vid_tree.insert(src.group, "end", iid=src.key, text=src.name)
        if not sources:
            vid_tree.insert("", "end", iid="none", text="no videos found in these folders (yet)")
        else:  # show the first frame of the first video
            vid_tree.selection_set(sources[0].key)

    def select_video(_e=None):
        sel = vid_tree.selection()
        if not sel or sel[0] not in player["sources"]:
            return
        stop_video()
        src = player["sources"][sel[0]]
        player.update(src=src, cache={}, n=vid.frame_count(src))
        view_box.configure(values=list(src.views))
        if vid_view.get() not in src.views:
            vid_view.set("rgb + labels" if "rgb + labels" in src.views else src.views[0])
        vid_slider.configure(to=max(0, player["n"] - 1))
        seek_video(0)

    def seek_video(i: int):
        src = player["src"]
        if src is None or player["n"] == 0:
            return
        i = max(0, min(player["n"] - 1, i))
        vid_pos.set(i)
        key = (vid_view.get(), i)
        if key not in player["cache"]:
            img = vid.read_frame(src, i, vid_view.get())
            if img is None:
                return
            player["cache"] = player["cache"] if len(player["cache"]) < 400 else {}
            player["cache"][key] = fit_width(img)
        show_array(player["cache"][key], f"{src.group}: {src.name}   ({vid_view.get()})", fitted=True)
        vid_label.configure(text=f"{i + 1} / {player['n']}")
        if abs(vid_slider.get() - i) >= 1:
            vid_slider.set(i)

    def step_video(d: int):
        stop_video()
        seek_video(vid_pos.get() + d)

    def tick():
        if not player["playing"]:
            return
        seek_video((vid_pos.get() + 1) % max(1, player["n"]))
        fps = player["src"].fps * float(vid_speed.get().rstrip("x"))
        player["after"] = app.after(max(10, int(1000 / fps)), tick)

    def toggle_play():
        if player["playing"]:
            stop_video()
        elif player["src"] is not None:
            player["playing"] = True
            play_btn.configure(text="Pause")
            tick()

    def stop_video():
        player["playing"] = False
        play_btn.configure(text="Play")
        if player["after"]:
            app.after_cancel(player["after"])
            player["after"] = None

    vid_tree.bind("<<TreeviewSelect>>", select_video)
    view_box.bind("<<ComboboxSelected>>", lambda _e: seek_video(vid_pos.get()))

    def on_tab_all(_e=None):
        on_tab()
        if notebook.index("current") == 3 and not vid_data.get():
            use_run_folders()

    notebook.bind("<<NotebookTabChanged>>", on_tab_all)

    # --- right side: buttons, image, log ----------------------------------------------------------------------------
    btns = ttk.Frame(right)
    btns.pack(fill="x")
    preview_btn = ttk.Button(btns, text="Preview frame", command=lambda: run_preview())
    run_btn = ttk.Button(btns, text="Run all steps", command=lambda: run_stages(run_all_stages(settings())))
    stop_btn = ttk.Button(btns, text="Stop", command=lambda: stop(), state="disabled")
    for b in (preview_btn, run_btn, stop_btn):
        b.pack(side="left", padx=(0, 6))
    ttk.Button(btns, text="Open dataset", command=lambda: open_dir(dataset_path(settings()))).pack(side="left", padx=6)
    ttk.Button(btns, text="Open Cosmos folder", command=lambda: open_dir(cosmos_path(settings()))).pack(side="left")
    steps_row = ttk.Frame(right, padding=(0, 4, 0, 6))
    steps_row.pack(fill="x")
    ttk.Label(steps_row, text="Steps:").pack(side="left", padx=(0, 6))
    step_btns = []
    for label, stage in [("1 Generate (Isaac Sim)", "generate"), ("2 Make Cosmos videos", "prepare"),
                         ("3 Send to Cosmos", "send"), ("4 Fetch results", "fetch"),
                         ("5 Convert for training", "convert")]:  # fmt: skip
        b = ttk.Button(steps_row, text=label, command=lambda st=stage: run_stages((st,)))
        b.pack(side="left", padx=(0, 6))
        step_btns.append(b)
    image_label = ttk.Label(right, text="Preview appears here", anchor="center", relief="groove")
    image_label.pack(fill="x", ipady=4)
    caption = ttk.Label(right, text="", foreground="#555")
    caption.pack(fill="x")
    log = scrolledtext.ScrolledText(right, height=20, font=("TkFixedFont", 9), state="disabled", wrap="word")
    log.pack(fill="both", expand=True, pady=(6, 0))
    status = ttk.Label(app, text="ready", relief="sunken", anchor="w")
    status.place(relx=0, rely=1, anchor="sw", relwidth=1)

    def write(line: str):
        log.configure(state="normal")
        log.insert("end", line if line.endswith("\n") else line + "\n")
        log.see("end")
        log.configure(state="disabled")

    def fit_width(img):
        """Scale a BGR image to the width of the image area (and at most 62 % of the window height)."""
        import cv2

        width = max(400, right.winfo_width() - 20)
        height = round(img.shape[0] * width / img.shape[1])
        max_h = max(240, int(app.winfo_height() * 0.62))
        if height > max_h:
            width, height = round(width * max_h / height), max_h
        return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

    def show_array(img, text: str, fitted: bool = False):
        img = img if fitted else fit_width(img)
        try:
            from PIL import Image, ImageTk

            photo = ImageTk.PhotoImage(Image.fromarray(img[..., ::-1]))
        except ImportError:
            import cv2

            ok, png = cv2.imencode(".png", img)
            photo = tk.PhotoImage(data=base64.b64encode(png.tobytes()).decode())
        image_label.configure(image=photo, text="")
        image_label.image = photo
        caption.configure(text=text)

    def show_image(path: Path, label_file: Path | None, text: str):
        import cv2

        from .viz import draw

        img = cv2.imread(str(path))
        if img is None:
            write(f"image not found: {path}")
            return
        if label_file is not None and label_file.exists():
            img = draw(img, json.loads(label_file.read_text()))
        show_array(img, text)

    # --- processes ---------------------------------------------------------------------------------------------------
    def set_busy(busy: bool, what: str = ""):
        for b in [preview_btn, run_btn, *step_btns]:
            b.configure(state="disabled" if busy else "normal")
        if not busy and v["source"].get() != "demos":
            step_btns[4].configure(state="disabled")  # convert is for recorded demos only
        stop_btn.configure(state="normal" if busy else "disabled")
        status.configure(text=what if busy else "ready")

    noise = re.compile(r"^\[worldsynth\]|Error|Traceback|^\s+File |Exception|^\S+Error")

    def start(cmd: list[str], env: dict, what: str, after=None, quiet: bool = False):
        """Run ``cmd`` and stream its output to the log; ``quiet`` keeps only WorldSynth progress and errors."""
        write(f"\n$ {' '.join(shlex.quote(c) for c in cmd)}")
        proc = subprocess.Popen(
            cmd, cwd=str(ROOT), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
            start_new_session=True,
        )  # fmt: skip
        state["proc"], state["after_run"] = proc, after
        set_busy(True, what)

        def pump():
            for line in proc.stdout:
                if not quiet or noise.search(line):
                    state["queue"].put(("log", line, None))
            state["queue"].put(("done", proc.wait(), None))

        threading.Thread(target=pump, daemon=True).start()

    def prepare(s: GuiSettings, stages=STAGES, for_preview=False) -> Path | None:
        """Check the settings and write the merged config; None if something is wrong."""
        errors = validate(s, stages, for_preview)
        if errors:
            messagebox.showerror("WorldSynth", "\n".join(errors))
            return None
        save_settings(s)
        config = write_run_config(s)
        if s.config_overrides:
            write(f"config: {base_config_path(s)} + {len(s.config_overrides)} change(s) -> {config}")
        return config

    def preview_demo(s: GuiSettings):
        """First frame of the first demo with its segmentation labels, straight from the HDF5 (no Isaac Sim)."""
        import h5py

        from . import demos as dm
        from .viz import draw

        cam = s.demo_cameras.split(",")[0].strip()
        with h5py.File(s.demos_hdf5, "r") as f:
            name = sorted(f["data"].keys(), key=lambda n: (len(n), n))[0]
            obs = f["data"][name]["obs"]
            if cam not in obs:
                messagebox.showerror("WorldSynth", f"{name} has no observation '{cam}'. It has: {', '.join(obs)}")
                return
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
        show_array(img, f"{name} / {cam}, frame 0 ({n_demos} demos, {n_frames} frames in this one; {extra})")

    def run_preview():
        s = settings()
        if s.source == "demos":
            errors = validate(s, for_preview=True)
            if errors:
                messagebox.showerror("WorldSynth", "\n".join(errors))
                return
            preview_demo(s)
            return
        config = prepare(s, for_preview=True)
        if config is None:
            return
        out = ROOT / "cache" / "gui_preview"
        env = dict(os.environ)
        env["ISAACLAB_DIR"] = env.get("ISAACLAB_DIR", str(ROOT.parent / "IsaacLab"))

        def done(code):
            clip = out / "clips" / "clip_0000"
            if code == 0 and (clip / "rgb" / "000000.png").exists():
                show_image(clip / "rgb" / "000000.png", clip / "labels" / "000000.json", "preview: frame 0 with labels")
            else:
                write("preview failed, see the log above")

        cmd = preview_command(s, config, out, isaaclab_python())
        start(cmd, env, "rendering preview (Isaac Sim starts, ~30 s)...", done, True)

    def run_stages(stages: tuple[str, ...]):
        s = settings()
        config = prepare(s, stages)
        if config is None:
            return
        try:
            looks, negative = write_prompts(s, all_prompt_edits())
        except ValueError as e:
            messagebox.showerror("WorldSynth", str(e))
            return
        if state["prompt_edits"] and "prepare" in stages:
            write(f"prompts with your edits: {', '.join(sorted(state['prompt_edits']))}")
        data, batch = dataset_path(s), cosmos_path(s)

        def done(code):
            if code != 0:
                write("stopped with an error, see the log above")
                return
            clips = sorted((data / "clips").glob("*/rgb")) if (data / "clips").is_dir() else []
            clip = clips[0].parent if clips else data / "clips" / "clip_0000"
            img_dir = "rgb_cosmos" if "fetch" in stages else "rgb"
            frames = sorted((clip / img_dir).glob("*.png"))
            if frames and ("generate" in stages or "fetch" in stages):
                mid = frames[len(frames) // 2]
                show_image(mid, clip / "labels" / f"{mid.stem}.json", f"{s.name}: {img_dir}/{mid.name} with labels")
            next_step = {"generate": "2 Make Cosmos videos", "prepare": "3 Send to Cosmos", "send": "4 Fetch results"}
            if s.source == "demos":
                next_step["fetch"] = "5 Convert for training"
            last = stages[-1]
            if last == "convert":
                write(f"training data: {Path(s.convert_out) if s.convert_out else data / 'training_data'}")
            write(f"done ({', '.join(stages)}). dataset: {data}  Cosmos folder: {batch}"
                  + (f"  next: {next_step[last]}" if last in next_step else ""))  # fmt: skip

        what = "running all steps" if stages == run_all_stages(s) else f"running step: {', '.join(stages)}"
        cmd = pipeline_command(s, config, stages=stages, prompts=looks, negative=negative)
        start(cmd, ssh_env(s), f"{what} ({s.name})...", done)

    def test_connection():
        s = settings()
        if not (s.spark_host and s.spark_user):
            messagebox.showerror("WorldSynth", "Enter the Spark host and user.")
            return
        remote = (
            "echo connected to $(hostname); "
            "test -x ~/cosmos-work/run_transfer.sh && echo 'Cosmos runner found (~/cosmos-work/run_transfer.sh)' "
            "|| echo 'WARNING: ~/cosmos-work/run_transfer.sh is missing (copy scripts/cosmos/run_transfer.sh there)'; "
            'free -g | awk \'/Mem:/{print "free memory: " $7 " GB"}\'; '
            "pgrep -f '[c]osmos_framework.scripts.inference' > /dev/null && echo 'a Cosmos job is running now' || true"
        )
        cmd = ["ssh", "-o", "ConnectTimeout=15", "-o", "StrictHostKeyChecking=accept-new",
               "-o", "ControlMaster=auto", "-o", f"ControlPath={Path.home() / '.ssh' / 'ws-%C'}",
               "-o", "ControlPersist=4h", f"{s.spark_user}@{s.spark_host}", remote]  # fmt: skip
        start(cmd, ssh_env(s), "testing SSH connection...")

    def stop():
        proc = state["proc"]
        if proc and proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)
            write("stopped. A Cosmos job already started on the Spark keeps running; use '4 Fetch results' later "
                  "to wait for it and copy the results back.")  # fmt: skip

    def open_dir(path: Path):
        subprocess.Popen(["xdg-open", str(path if path.exists() else path.parent)])

    def restore_session():
        if not SETTINGS_FILE.exists():
            messagebox.showinfo("WorldSynth", "No saved session yet (settings are saved when you run).")
            return
        apply_settings(load_settings())
        write(f"restored the settings of the last session from {SETTINGS_FILE}")

    def poll():
        try:
            while True:
                kind, a, _ = state["queue"].get_nowait()
                if kind == "log":
                    write(a.rstrip("\n"))
                elif kind == "done":
                    write(f"[exit code {a}]")
                    set_busy(False)
                    if state["after_run"]:
                        state["after_run"](a)
                elif kind == "scene":
                    show_scene(a)
                elif kind == "scene_error":
                    scene_info_label.configure(text=f"could not read scene: {a}")
        except queue.Empty:
            pass
        app.after(100, poll)

    def on_close():
        if state["proc"] and state["proc"].poll() is None:
            if not messagebox.askyesno("WorldSynth", "A run is in progress. Stop it and quit?"):
                return
            stop()
        app.destroy()

    update_source_state()
    refresh_config()
    write("WorldSynth GUI: all settings start at their defaults ('Restore last session' brings back the last run's).")
    write("The password is never saved.")
    app.protocol("WM_DELETE_WINDOW", on_close)
    app.after(100, poll)
    if "--restore" in sys.argv[1:]:
        app.after(200, restore_session)
    if "--preview" in sys.argv[1:]:  # render a preview of the restored setup right away
        app.after(1500, run_preview)
    if "--config-tab" in sys.argv[1:]:
        app.after(500, lambda: notebook.select(1))
    if "--prompts-tab" in sys.argv[1:]:
        app.after(500, lambda: notebook.select(2))
    if "--videos-tab" in sys.argv[1:]:
        app.after(500, lambda: notebook.select(3))
    app.mainloop()


if __name__ == "__main__":
    main()
