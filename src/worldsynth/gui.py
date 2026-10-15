"""WorldSynth desktop GUI: pick the CAD folder, an optional USD scene with its camera and surface, the dataset and
Cosmos settings and the Spark login, then preview a frame or run the whole pipeline (``scripts/run_pipeline.sh``).

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

ROOT = Path(__file__).resolve().parents[2]
SETTINGS_FILE = Path(os.environ.get("WORLDSYNTH_GUI_SETTINGS", Path.home() / ".config" / "worldsynth" / "gui.json"))
ASKPASS = Path.home() / ".cache" / "worldsynth" / "askpass.sh"
ALL_LOOKS = "all looks in configs/cosmos/prompts"
RANDOM_CAMERA = "(random cameras around the parts)"


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
    scene_lights_only: bool = True
    """A predesigned scene is usually lit already; added random lights tend to wash it out."""
    camera_azimuth: float = 10.0
    """± degrees the scene camera moves around its target over a clip."""
    camera_elevation: float = 3.0
    name: str = "run_001"
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
    extra: str = ""
    """Extra config overrides, one ``key=value`` per line."""
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


def validate(s: GuiSettings, for_preview: bool = False) -> list[str]:
    """Problems that block a run, as messages for the user."""
    errors = []
    if not s.cad_dir or not Path(s.cad_dir).is_dir():
        errors.append("Choose the folder with the CAD/STL files.")
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
    return errors


def common_overrides(s: GuiSettings) -> list[str]:
    """``--set`` style config overrides shared by the preview and the pipeline."""
    sets = [f"scene.parts_per_clip=[{s.parts_min},{s.parts_max}]"]
    if s.clutter:
        sets.append(f"randomization.clutter.count=[{s.clutter_min},{s.clutter_max}]")
    if s.use_scene and s.scene_camera:
        a, e = abs(s.camera_azimuth), abs(s.camera_elevation)
        sets += [
            f"camera.azimuth_sweep=[{-a},{a}]",
            f"camera.elevation_sweep=[{-e},{e}]",
            "camera.distance_sweep=[0,0]",
        ]
    sets += [line.strip() for line in s.extra.splitlines() if line.strip() and not line.strip().startswith("#")]
    return sets


def scene_args(s: GuiSettings) -> list[str]:
    if not s.use_scene:
        return []
    args = ["--scene", s.scene_usd, "--scene-surface", s.scene_surface]
    if s.scene_camera:
        args += ["--scene-camera", s.scene_camera]
    if s.scene_lights_only:
        args.append("--scene-lights-only")
    return args


def pipeline_command(s: GuiSettings, root: Path = ROOT) -> list[str]:
    """Command line of ``run_pipeline.sh`` for these settings (without secrets)."""
    cmd = [str(root / "scripts" / "run_pipeline.sh"), "--cad", s.cad_dir, "--name", s.name]
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


def preview_command(s: GuiSettings, out_dir: Path, isaaclab_python: str, root: Path = ROOT) -> list[str]:
    """One-frame render with ``generate.py`` to check the camera, surface and part scale."""
    width = 1200 if s.resolution == "256" else 1248
    cmd = [isaaclab_python, str(root / "scripts" / "generate.py"), "--config", str(root / "configs" / "default.yaml")]
    cmd += [
        f"cad.dir={s.cad_dir}",
        f"cad.cache_dir={root / 'cache' / 'usd'}",
        f"cad.units={s.units}",
        f"cad.class_from={s.class_from}",
        f"output.dir={out_dir}",
        "output.overwrite=true",
        "dataset.num_clips=1",
        "dataset.frames_per_clip=1",
        f"dataset.resolution=[{width},720]",
        f"randomization.clutter.enabled={str(s.clutter).lower()}",
    ]
    if s.use_scene:
        cmd += [f"scene.scene_usd={s.scene_usd}", f"scene.scene_surface={s.scene_surface}"]
        if s.scene_camera:
            cmd.append(f"scene.scene_camera={s.scene_camera}")
        cmd.append(f"scene.scene_lights_only={str(s.scene_lights_only).lower()}")
    return cmd + common_overrides(s)


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

    state = {"proc": None, "queue": queue.Queue(), "scene": None, "after_run": None}

    def settings() -> GuiSettings:
        data = {}
        for f in dataclasses.fields(GuiSettings):
            if f.name == "extra":
                data[f.name] = extra_text.get("1.0", "end").strip()
            else:
                try:
                    data[f.name] = v[f.name].get()
                except tk.TclError:
                    data[f.name] = getattr(GuiSettings(), f.name)
        cam = data["scene_camera"]
        data["scene_camera"] = "" if cam == RANDOM_CAMERA else cam.split("   ")[0]
        data["scene_surface"] = data["scene_surface"].split("   ")[0]
        return GuiSettings(**data)

    # --- layout ------------------------------------------------------------------------------------------------------
    left = ttk.Frame(app, padding=8)
    left.pack(side="left", fill="y")
    right = ttk.Frame(app, padding=8)
    right.pack(side="left", fill="both", expand=True)

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

    # parts
    fr = section("Parts (CAD / STL folder)")
    row(fr, 0, "Folder", ttk.Entry(fr, textvariable=v["cad_dir"], width=46),
        ttk.Button(fr, text="Browse...", command=lambda: browse_dir(v["cad_dir"])))  # fmt: skip
    row(fr, 1, "Units", ttk.Combobox(fr, textvariable=v["units"], values=["auto", "mm", "m", "cm", "inch"],
                                     state="readonly", width=10))  # fmt: skip
    row(fr, 2, "Class per", ttk.Combobox(fr, textvariable=v["class_from"], values=["file", "folder"],
                                         state="readonly", width=10))  # fmt: skip

    # scene
    fr = section("Scene")
    mode = ttk.Frame(fr)
    ttk.Radiobutton(mode, text="Built-in workcell", variable=v["use_scene"], value=False,
                    command=lambda: update_scene_state()).pack(side="left")  # fmt: skip
    ttk.Radiobutton(mode, text="USD scene", variable=v["use_scene"], value=True,
                    command=lambda: update_scene_state()).pack(side="left", padx=8)  # fmt: skip
    mode.grid(row=0, column=0, columnspan=3, sticky="w")
    scene_entry = ttk.Entry(fr, textvariable=v["scene_usd"], width=46)

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
    camera_box = ttk.Combobox(fr, textvariable=v["scene_camera"], state="readonly", width=44)
    row(fr, 3, "Camera", camera_box)
    surface_box = ttk.Combobox(fr, textvariable=v["scene_surface"], state="readonly", width=44)
    row(fr, 4, "Place parts on", surface_box)
    motion = ttk.Frame(fr)
    ttk.Label(motion, text="± azimuth").pack(side="left")
    ttk.Spinbox(motion, from_=0, to=90, increment=1, textvariable=v["camera_azimuth"], width=5).pack(side="left")
    ttk.Label(motion, text="°   ± elevation").pack(side="left")
    ttk.Spinbox(motion, from_=0, to=45, increment=1, textvariable=v["camera_elevation"], width=5).pack(side="left")
    ttk.Label(motion, text="°").pack(side="left")
    row(fr, 5, "Camera motion", motion)
    lights_cb = ttk.Checkbutton(fr, text="Use only the scene's own lights", variable=v["scene_lights_only"])
    lights_cb.grid(row=6, column=1, sticky="w")
    scene_widgets = [scene_entry, scene_btn, camera_box, surface_box, lights_cb] + list(motion.winfo_children())

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
        state["scene"] = info
        cams = [RANDOM_CAMERA] + [f"{c.path}   (at {c.position}, f={c.focal_length:g})" for c in info.cameras]
        surfs = [f"{s.path}   (top {s.top_z:.3f} m, {s.size[0]:.2f} × {s.size[1]:.2f} m)" for s in info.surfaces]
        camera_box.configure(values=cams)
        surface_box.configure(values=surfs)
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

    # dataset
    fr = section("Dataset")
    row(fr, 0, "Run name", ttk.Entry(fr, textvariable=v["name"], width=20))
    nums = ttk.Frame(fr)
    for label, key, hi in [("clips", "clips", 10000), ("frames/clip", "frames", 241)]:
        ttk.Spinbox(nums, from_=1, to=hi, textvariable=v[key], width=6).pack(side="left")
        ttk.Label(nums, text=f" {label}   ").pack(side="left")
    row(fr, 1, "Size", nums)
    parts = ttk.Frame(fr)
    ttk.Spinbox(parts, from_=1, to=50, textvariable=v["parts_min"], width=4).pack(side="left")
    ttk.Label(parts, text=" to ").pack(side="left")
    ttk.Spinbox(parts, from_=1, to=50, textvariable=v["parts_max"], width=4).pack(side="left")
    row(fr, 2, "Parts per clip", parts)
    clut = ttk.Frame(fr)
    ttk.Checkbutton(clut, text="on,", variable=v["clutter"]).pack(side="left")
    ttk.Spinbox(clut, from_=0, to=50, textvariable=v["clutter_min"], width=4).pack(side="left")
    ttk.Label(clut, text=" to ").pack(side="left")
    ttk.Spinbox(clut, from_=0, to=50, textvariable=v["clutter_max"], width=4).pack(side="left")
    ttk.Label(clut, text=" items").pack(side="left")
    row(fr, 3, "Clutter", clut)
    ttk.Label(fr, text="Extra settings (key=value per line)").grid(row=4, column=0, columnspan=2, sticky="w")
    extra_text = tk.Text(fr, height=3, width=52, font=("TkFixedFont", 9))
    extra_text.insert("1.0", s0.extra)
    extra_text.grid(row=5, column=0, columnspan=3, sticky="ew")

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
    row(fr, 2, "Look prompt", ttk.Combobox(fr, textvariable=v["prompt"], values=looks, state="readonly", width=44))
    row(fr, 3, "Negative", ttk.Combobox(fr, textvariable=v["negative"], values=negatives, state="readonly", width=44))
    ttk.Checkbutton(fr, text="Render only (skip Cosmos)", variable=v["render_only"]).grid(row=4, column=1, sticky="w")

    # spark
    fr = section("Spark (runs Cosmos)")
    row(fr, 0, "Host / IP", ttk.Entry(fr, textvariable=v["spark_host"], width=24))
    row(fr, 1, "User", ttk.Entry(fr, textvariable=v["spark_user"], width=24))
    row(fr, 2, "Password", ttk.Entry(fr, textvariable=v["password"], show="*", width=24),
        ttk.Button(fr, text="Test connection", command=lambda: test_connection()))  # fmt: skip
    ttk.Label(fr, text="not saved; leave empty for SSH keys", foreground="#555").grid(row=3, column=1, sticky="w")

    # buttons, at the top of the right side so they are always visible
    btns = ttk.Frame(right, padding=(0, 0, 0, 6))
    btns.pack(fill="x")
    preview_btn = ttk.Button(btns, text="Preview frame", command=lambda: run_preview())
    run_btn = ttk.Button(btns, text="Run pipeline", command=lambda: run_pipeline())
    stop_btn = ttk.Button(btns, text="Stop", command=lambda: stop(), state="disabled")
    open_btn = ttk.Button(btns, text="Open output", command=lambda: open_output())
    for b in (preview_btn, run_btn, stop_btn, open_btn):
        b.pack(side="left", padx=(0, 6))

    # right side: buttons (above), image, log
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

    def check(s: GuiSettings, for_preview=False) -> bool:
        errors = validate(s, for_preview)
        if errors:
            messagebox.showerror("WorldSynth", "\n".join(errors))
            return False
        save_settings(s)
        return True

    def run_preview():
        s = settings()
        if not check(s, for_preview=True):
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

        start(
            preview_command(s, out, isaaclab_python()),
            env,
            "rendering preview (Isaac Sim starts, ~30 s)...",
            done,
            True,
        )

    def run_pipeline():
        s = settings()
        if not check(s):
            return
        out = ROOT / "output" / s.name

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
            write(f"done: {out}")

        start(pipeline_command(s), ssh_env(s), f"running pipeline {s.name}...", done)

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
        path = ROOT / "output" / settings().name
        subprocess.Popen(["xdg-open", str(path if path.exists() else ROOT / "output")])

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
    app.mainloop()


if __name__ == "__main__":
    main()
