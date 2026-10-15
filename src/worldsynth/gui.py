"""WorldSynth desktop GUI: pick the CAD folder, an optional USD scene with its camera and surface, the dataset and
Cosmos settings, the output folders and the Spark login; adjust any config value on the Configuration tab (load a
YAML, override values); then preview a frame or run the whole pipeline (``scripts/run_pipeline.sh``).

Start it with the Isaac Lab Python (it needs ``pxr`` to read scenes): ``scripts/gui.sh`` (``--preview`` renders a
preview of the saved setup right after start).

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

import yaml

from .config import load_config

ROOT = Path(__file__).resolve().parents[2]
SETTINGS_FILE = Path(os.environ.get("WORLDSYNTH_GUI_SETTINGS", Path.home() / ".config" / "worldsynth" / "gui.json"))
DEFAULT_CONFIG = ROOT / "configs" / "default.yaml"
RUN_CONFIG = ROOT / "cache" / "gui_run_config.yaml"
"""Merged configuration (base YAML + Configuration tab overrides) handed to each preview and run."""
ASKPASS = Path.home() / ".cache" / "worldsynth" / "askpass.sh"
ALL_LOOKS = "all looks in configs/cosmos/prompts"
RANDOM_CAMERA = "(random cameras around the parts)"
NO_SCENE_PARTS = "(none: parts only from the CAD folder)"


@dataclass
class GuiSettings:
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
    render_only: bool = False
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


def validate(s: GuiSettings, for_preview: bool = False) -> list[str]:
    """Problems that block a run, as messages for the user."""
    errors = []
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
        if not s.render_only and not (s.spark_host and s.spark_user):
            errors.append("Enter the Spark host and user (or tick 'Render only').")
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


def pipeline_command(s: GuiSettings, config: Path, root: Path = ROOT) -> list[str]:
    """Command line of ``run_pipeline.sh`` for these settings (without secrets)."""
    cmd = [str(root / "scripts" / "run_pipeline.sh"), "--name", s.name, "--config", str(config)]
    if s.cad_dir:
        cmd += ["--cad", s.cad_dir]
    cmd += ["--dataset-dir", str(dataset_path(s, root)), "--cosmos-dir", str(cosmos_path(s, root))]
    cmd += ["--clips", str(s.clips), "--frames", str(s.frames), "--res", s.resolution, "--steps", str(s.steps)]
    cmd += ["--looks", str(s.looks), "--units", s.units, "--class-from", s.class_from]
    cmd += ["--spark", f"{s.spark_user}@{s.spark_host}", "--negative", str(root / s.negative)]
    if s.prompt != ALL_LOOKS:
        cmd += ["--prompts", str(root / s.prompt)]
    if not s.clutter:
        cmd.append("--no-clutter")
    if s.render_only:
        cmd.append("--skip-cosmos")
    cmd += scene_args(s)
    for item in common_overrides(s):
        cmd += ["--set", item]
    return cmd


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

    s0 = load_settings()
    app = tk.Tk()
    app.title("WorldSynth")
    app.geometry("1500x940")
    style = ttk.Style(app)
    if "clam" in style.theme_names():
        style.theme_use("clam")

    v: dict[str, tk.Variable] = {}
    for f in dataclasses.fields(GuiSettings):
        val = getattr(s0, f.name)
        if isinstance(val, dict):
            continue
        var = (
            tk.BooleanVar(value=val)
            if isinstance(val, bool)
            else tk.IntVar(value=val)
            if isinstance(val, int)
            else tk.DoubleVar(value=val)
            if isinstance(val, float)
            else tk.StringVar(value=val)
        )
        v[f.name] = var

    state = {
        "proc": None,
        "queue": queue.Queue(),
        "after_run": None,
        "overrides": dict(s0.config_overrides),
        "comments": config_comments(),
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

    # parts
    fr = section("Parts (CAD / STL folder)")
    row(fr, 0, "Folder", ttk.Entry(fr, textvariable=v["cad_dir"], width=30),
        ttk.Button(fr, text="Browse...", command=lambda: browse_dir(v["cad_dir"])))  # fmt: skip
    row(fr, 1, "Units", ttk.Combobox(fr, textvariable=v["units"], values=["auto", "mm", "m", "cm", "inch"],
                                     state="readonly", width=10))  # fmt: skip
    row(fr, 2, "Class per", ttk.Combobox(fr, textvariable=v["class_from"], values=["file", "folder"],
                                         state="readonly", width=10))  # fmt: skip
    hint(fr, 3, "optional when the USD scene provides the parts (Scene: Parts in scene)")

    # scene
    fr = section("Scene")
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

    def load_scene(select_saved: bool = False):
        path = v["scene_usd"].get()
        if not path or not Path(path).is_file():
            return
        scene_info_label.configure(text="reading scene...")

        def work():
            try:
                info = scene_info.inspect(path)
                state["queue"].put(("scene", info, select_saved))
            except Exception as e:  # noqa: BLE001  (shown to the user)
                state["queue"].put(("scene_error", str(e), None))

        threading.Thread(target=work, daemon=True).start()

    def show_scene(info, select_saved):
        cams = [RANDOM_CAMERA] + [f"{c.path}   (at {c.position}, f={c.focal_length:g})" for c in info.cameras]
        surfs = [f"{s.path}   (top {s.top_z:.3f} m, {s.size[0]:.2f} × {s.size[1]:.2f} m)" for s in info.surfaces]
        groups = [NO_SCENE_PARTS] + [
            f"{g.path}   ({len(g.children)}: {', '.join(g.children[:4])}{', ...' if len(g.children) > 4 else ''})"
            for g in info.groups
        ]
        camera_box.configure(values=cams)
        surface_box.configure(values=surfs)
        parts_box.configure(values=groups)
        saved_parts = s0.scene_parts if select_saved else ""
        v["scene_parts"].set(next((g for g in groups if saved_parts and g.split("   ")[0] == saved_parts), groups[0]))
        saved_cam, saved_surf = s0.scene_camera if select_saved else "", s0.scene_surface if select_saved else ""
        match_cam = next((c for c in cams if saved_cam and c.split("   ")[0] == saved_cam), None)
        v["scene_camera"].set(match_cam or (cams[1] if len(cams) > 1 else cams[0]))
        match_surf = next((s for s in surfs if saved_surf and s.split("   ")[0] == saved_surf), None)
        # default surface: the largest one that is not the floor-sized first entry, if there is a choice
        v["scene_surface"].set(match_surf or (surfs[1] if len(surfs) > 1 else surfs[0] if surfs else ""))
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
    hint(fr, 4, "batch sent to the Spark; results copied back to <folder>/results; empty: output/<run name>_cosmos")

    # dataset
    fr = section("Dataset")
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
    looks = [ALL_LOOKS] + [
        str(p.relative_to(ROOT))
        for p in sorted((ROOT / "configs" / "cosmos").rglob("*.json"))
        if "negative" not in p.name
    ]
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
    row(fr, 2, "Look prompt", ttk.Combobox(fr, textvariable=v["prompt"], values=looks, state="readonly", width=30))
    row(fr, 3, "Negative", ttk.Combobox(fr, textvariable=v["negative"], values=negatives, state="readonly", width=30))
    ttk.Checkbutton(fr, text="Render only (skip Cosmos)", variable=v["render_only"]).grid(row=4, column=1, sticky="w")

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
    tree = ttk.Treeview(cfg_tab, columns=("value", "source"), height=24)
    tree.heading("#0", text="Setting")
    tree.heading("value", text="Value (double-click to edit)")
    tree.heading("source", text="From")
    tree.column("#0", width=230)
    tree.column("value", width=200)
    tree.column("source", width=100)
    tree.tag_configure("override", foreground="#0b5cad", font=("TkDefaultFont", 9, "bold"))
    tree.tag_configure("file", foreground="#1b7a1b")
    tree.tag_configure("pipeline", foreground="#888")
    tree_box = ttk.Frame(cfg_tab)
    tree_box.pack(fill="both", expand=True)
    tree_scroll = ttk.Scrollbar(cfg_tab, orient="vertical", command=tree.yview)
    tree.configure(yscrollcommand=tree_scroll.set)
    tree.pack(in_=tree_box, side="left", fill="both", expand=True)
    tree_scroll.pack(in_=tree_box, side="right", fill="y")
    desc = ttk.Label(cfg_tab, text="", foreground="#444", wraplength=520, justify="left")
    desc.pack(fill="x", pady=(4, 0))
    legend = ("blue: changed here   green: from the loaded YAML   grey: set on the Pipeline tab (it wins)   "
              "black: default")  # fmt: skip
    ttk.Label(cfg_tab, text=legend, foreground="#666", wraplength=520).pack(fill="x")

    def refresh_config():
        s = settings()
        base = base_config_path(s)
        base_label.configure(text=str(base.relative_to(ROOT)) if base.is_relative_to(ROOT) else str(base))
        try:
            defaults = config_values(None)
            from_file = config_values(base)
            current = config_values(base, state["overrides"])
        except Exception as e:  # noqa: BLE001  (shown to the user)
            messagebox.showerror("WorldSynth", f"Configuration error: {e}")
            return
        locked = pipeline_keys(s)
        open_items = {i for i in tree.get_children("") if tree.item(i, "open")} or None
        selected = tree.selection()
        tree.delete(*tree.get_children(""))
        for key, value in current.items():
            parent = ""
            parts_ = key.split(".")
            for depth in range(1, len(parts_)):
                node = ".".join(parts_[:depth])
                if not tree.exists(node):
                    tree.insert(parent, "end", iid=node, text=parts_[depth - 1],
                                open=open_items is None or node in open_items)  # fmt: skip
                parent = node
            if key in locked:
                text, src, tag = locked[key], "Pipeline tab", "pipeline"
            elif key in state["overrides"]:
                text, src, tag = format_value(value), "changed here", "override"
            elif from_file[key] != defaults[key]:
                text, src, tag = format_value(value), "YAML file", "file"
            else:
                text, src, tag = format_value(value), "default", ""
            tree.insert(parent, "end", iid=key, text=parts_[-1], values=(text, src), tags=(tag,))
        if selected and tree.exists(selected[0]):
            tree.selection_set(selected[0])
            tree.see(selected[0])
        else:
            tree.yview_moveto(0)

    def on_select(_event=None):
        sel = tree.selection()
        if sel and sel[0] in state["comments"]:
            desc.configure(text=f"{sel[0]}: {state['comments'][sel[0]]}")
        elif sel:
            desc.configure(text=sel[0])

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

        def commit(_e=None):
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

        entry.bind("<Return>", commit)
        entry.bind("<KP_Enter>", commit)
        entry.bind("<Escape>", lambda _e: entry.destroy())
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
        if messagebox.askyesno("WorldSynth", "Use configs/default.yaml and drop all values changed here?"):
            v["config_file"].set("")
            state["overrides"].clear()
            refresh_config()

    for text, cmd in [("Load YAML...", load_yaml), ("Save as YAML...", save_yaml), ("Revert selected", revert_selected),
                      ("Reset to defaults", reset_config)]:  # fmt: skip
        ttk.Button(cfg_btns, text=text, command=cmd).pack(side="left", padx=(0, 6))
    tree.bind("<<TreeviewSelect>>", on_select)
    tree.bind("<Double-1>", edit_cell)
    notebook.bind("<<NotebookTabChanged>>", lambda _e: refresh_config() if notebook.index("current") == 1 else None)

    # --- right side: buttons, image, log ----------------------------------------------------------------------------
    btns = ttk.Frame(right, padding=(0, 0, 0, 6))
    btns.pack(fill="x")
    preview_btn = ttk.Button(btns, text="Preview frame", command=lambda: run_preview())
    run_btn = ttk.Button(btns, text="Run pipeline", command=lambda: run_pipeline())
    stop_btn = ttk.Button(btns, text="Stop", command=lambda: stop(), state="disabled")
    open_btn = ttk.Button(btns, text="Open output", command=lambda: open_output())
    for b in (preview_btn, run_btn, stop_btn, open_btn):
        b.pack(side="left", padx=(0, 6))
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

    def show_image(path: Path, label_file: Path | None, text: str):
        import cv2

        from .viz import draw

        img = cv2.imread(str(path))
        if img is None:
            write(f"image not found: {path}")
            return
        if label_file is not None and label_file.exists():
            img = draw(img, json.loads(label_file.read_text()))
        width = max(400, right.winfo_width() - 20)
        img = cv2.resize(img, (width, round(img.shape[0] * width / img.shape[1])), interpolation=cv2.INTER_AREA)
        ok, png = cv2.imencode(".png", img)
        photo = tk.PhotoImage(data=base64.b64encode(png.tobytes()).decode())
        image_label.configure(image=photo, text="")
        image_label.image = photo
        caption.configure(text=text)

    # --- processes ---------------------------------------------------------------------------------------------------
    def set_busy(busy: bool, what: str = ""):
        for b in (preview_btn, run_btn):
            b.configure(state="disabled" if busy else "normal")
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

    def prepare(s: GuiSettings, for_preview=False) -> Path | None:
        """Check the settings and write the merged config; None if something is wrong."""
        errors = validate(s, for_preview)
        if errors:
            messagebox.showerror("WorldSynth", "\n".join(errors))
            return None
        save_settings(s)
        config = write_run_config(s)
        if s.config_overrides:
            write(f"config: {base_config_path(s)} + {len(s.config_overrides)} change(s) -> {config}")
        return config

    def run_preview():
        s = settings()
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

    def run_pipeline():
        s = settings()
        config = prepare(s)
        if config is None:
            return
        out = dataset_path(s)

        def done(code):
            if code != 0:
                write("pipeline stopped with an error, see the log above")
                return
            clip = out / "clips" / "clip_0000"
            img_dir = "rgb" if s.render_only else "rgb_cosmos"
            frame = sorted((clip / img_dir).glob("*.png"))
            if frame:
                mid = frame[len(frame) // 2]
                show_image(mid, clip / "labels" / f"{mid.stem}.json", f"{s.name}: {img_dir}/{mid.name} with labels")
            write(f"done: dataset {out}" + ("" if s.render_only else f", Cosmos results {cosmos_path(s) / 'results'}"))

        start(pipeline_command(s, config), ssh_env(s), f"running pipeline {s.name}...", done)

    def test_connection():
        s = settings()
        if not (s.spark_host and s.spark_user):
            messagebox.showerror("WorldSynth", "Enter the Spark host and user.")
            return
        remote = (
            "echo connected to $(hostname); "
            "test -x ~/cosmos-work/run_transfer.sh && echo 'Cosmos runner found (~/cosmos-work/run_transfer.sh)' "
            "|| echo 'WARNING: ~/cosmos-work/run_transfer.sh is missing'; "
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
            write("stopped. A Cosmos job already started on the Spark keeps running; run again with the same name "
                  "to wait for it and copy the results back.")  # fmt: skip

    def open_output():
        path = dataset_path(settings())
        subprocess.Popen(["xdg-open", str(path if path.exists() else path.parent)])

    def poll():
        try:
            while True:
                kind, a, b = state["queue"].get_nowait()
                if kind == "log":
                    write(a.rstrip("\n"))
                elif kind == "done":
                    write(f"[exit code {a}]")
                    set_busy(False)
                    if state["after_run"]:
                        state["after_run"](a)
                elif kind == "scene":
                    show_scene(a, b)
                elif kind == "scene_error":
                    scene_info_label.configure(text=f"could not read scene: {a}")
        except queue.Empty:
            pass
        app.after(100, poll)

    def on_close():
        save_settings(settings())
        if state["proc"] and state["proc"].poll() is None:
            if not messagebox.askyesno("WorldSynth", "A run is in progress. Stop it and quit?"):
                return
            stop()
        app.destroy()

    update_scene_state()
    if s0.use_scene and s0.scene_usd:
        load_scene(select_saved=True)
    write("WorldSynth GUI. Settings are saved in " + str(SETTINGS_FILE) + " (the password is never saved).")
    app.protocol("WM_DELETE_WINDOW", on_close)
    app.after(100, poll)
    if "--preview" in sys.argv[1:]:  # render a preview of the saved setup right away
        app.after(1500, run_preview)
    if "--config-tab" in sys.argv[1:]:
        app.after(500, lambda: notebook.select(1))
    app.mainloop()


if __name__ == "__main__":
    main()
