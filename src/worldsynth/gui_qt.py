"""The WorldSynth window (Qt, PySide6). The settings, checks and commands live in :mod:`worldsynth.gui`; this module
only builds the window and runs the pipeline processes.

Layout: tabs on the left (Pipeline, Configuration, Prompts, Videos), the run buttons, image area and log on the right.
"""

from __future__ import annotations

import json
import os
import shlex
import signal
import subprocess
import threading
from collections.abc import Callable
from pathlib import Path
from string import Template

import numpy as np
from PySide6.QtCore import QObject, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QColor, QDesktopServices, QFont, QFontDatabase, QImage, QPainter
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from . import gui as g
from .gui import GuiSettings

THEMES = {
    # text colors keep at least ~7:1 contrast on the window background of their theme
    "dark": {
        "hint": "#b8bdc6", "info": "#8ab4ff", "changed": "#8ab4ff", "file": "#6ee7a0", "locked": "#9aa0a6",
        "button": "#3b4049", "button_hover": "#474d57", "border": "#5a616b", "disabled_text": "#7d838c",
        "accent": "#3b82f6", "accent_hover": "#2563eb", "accent_disabled": "#2c3e5c",
        "step": "#1f3354", "step_hover": "#284273", "step_text": "#d6e4ff", "step_border": "#3e64a6",
        "danger": "#dc2626", "danger_hover": "#b91c1c",
    },
    "light": {
        "hint": "#4b5563", "info": "#1d4ed8", "changed": "#1d4ed8", "file": "#15803d", "locked": "#6b7280",
        "button": "#f3f4f6", "button_hover": "#e5e7eb", "border": "#c4c9d0", "disabled_text": "#9ca3af",
        "accent": "#2563eb", "accent_hover": "#1d4ed8", "accent_disabled": "#a9c1f5",
        "step": "#e8f0fe", "step_hover": "#d7e4fd", "step_text": "#1e3a8a", "step_border": "#9db7ef",
        "danger": "#dc2626", "danger_hover": "#b91c1c",
    },
}  # fmt: skip
STYLE = Template("""
QGroupBox { font-weight: 600; border: 1px solid $border; border-radius: 6px; margin-top: 14px;
            padding: 10px 8px 8px 8px; }
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }
QLabel[role="hint"] { color: $hint; }
QLabel[role="info"] { color: $info; }
QPushButton { background: $button; border: 1px solid $border; border-radius: 5px; padding: 5px 12px; }
QPushButton:hover { background: $button_hover; }
QPushButton:disabled { color: $disabled_text; }
QPushButton#primary { background: $accent; color: white; border: 1px solid $accent; font-weight: 600; }
QPushButton#primary:hover { background: $accent_hover; }
QPushButton#primary:disabled { background: $accent_disabled; border-color: $accent_disabled; color: $disabled_text; }
QPushButton#danger:enabled { background: $danger; color: white; border-color: $danger; font-weight: 600; }
QPushButton#danger:enabled:hover { background: $danger_hover; }
QPushButton#step { background: $step; color: $step_text; border: 1px solid $step_border; }
QPushButton#step:hover { background: $step_hover; border-color: $accent; }
QPushButton#step:disabled { background: $button; color: $disabled_text; border-color: $border; }
QLineEdit:focus, QPlainTextEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {
    border: 1px solid $accent; }
""")


def colors() -> dict[str, str]:
    """The colors of the light or dark theme the desktop gives the application."""
    dark = QApplication.palette().window().color().lightness() < 128
    return THEMES["dark" if dark else "light"]


HDF5 = "HDF5 (*.hdf5 *.h5);;All files (*)"
TAB_PIPELINE, TAB_CONFIG, TAB_PROMPTS, TAB_VIDEOS = range(4)
STEP_LABELS = [
    ("1 Generate (Isaac Sim)", "generate"),
    ("2 Make Cosmos videos", "prepare"),
    ("3 Send to Cosmos", "send"),
    ("4 Fetch results", "fetch"),
    ("5 Convert for training", "convert"),
]
DEMOS_GUIDE = (
    "<b>Steps</b> (buttons on the right):<br>"
    "<b>1 Import demos</b>: copies the RGB, depth and segmentation of every demo and camera into the dataset "
    "folder.<br>"
    "<b>2 Make Cosmos videos</b>, <b>3 Send to Cosmos</b>: builds the Cosmos jobs and starts them on the Spark.<br>"
    "<b>4 Fetch results</b>: copies the Cosmos videos back and writes the Cosmos demos HDF5: the same demos (actions, "
    "states) with the camera images replaced.<br>"
    "<b>5 Convert for training</b>: recorded demos + Cosmos demos → one merged training dataset (e.g. LeRobot for "
    "GR00T)."
)
CONVERTER_HINT = (
    "Converter: from the robot project, e.g. kuka/scripts/gr00t/convert_to_gr00t_lerobot.py with "
    "~/venvs/convert/bin/python (LeRobot for GR00T); empty Python = Isaac Lab's. {inputs} = the HDF5 files, "
    "{output} = the training dataset folder."
)
PROMPTS_HINT = (
    "Edit the Cosmos prompts (JSON). {parts} becomes your class names. Edits are used by the next run even if not "
    "saved; Save writes them to the file. Which looks a run uses: Pipeline tab › Cosmos Transfer › Looks used."
)
VIDEOS_HINT = (
    "Isaac Sim clips play from their frames: view them as rgb, with labels, depth, segmentation or instance masks. "
    "The video plays in the large area on the right."
)


def label(text: str, role: str = "hint") -> QLabel:
    lbl = QLabel(text)
    lbl.setWordWrap(True)
    lbl.setMinimumWidth(300)  # else a form row reserves the height of a much narrower, many-line label
    lbl.setProperty("role", role)
    return lbl


def hbox(*widgets, stretch: bool = True) -> QWidget:
    box = QWidget()
    lay = QHBoxLayout(box)
    lay.setContentsMargins(0, 0, 0, 0)
    for w in widgets:
        lay.addWidget(QLabel(w) if isinstance(w, str) else w)
    if stretch:
        lay.addStretch(1)
    return box


def spin(lo: int, hi: int, width: int = 70) -> QSpinBox:
    s = QSpinBox()
    s.setRange(lo, hi)
    s.setFixedWidth(width)
    return s


class ImageView(QWidget):
    """Shows a BGR image scaled to fit, keeping its aspect ratio."""

    def __init__(self):
        super().__init__()
        self._img: QImage | None = None
        self.setMinimumHeight(240)

    def set_bgr(self, img: np.ndarray):
        rgb = np.ascontiguousarray(img[..., ::-1])
        h, w = rgb.shape[:2]
        self._img = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
        self.update()

    def paintEvent(self, _event):  # noqa: N802  (Qt API)
        p = QPainter(self)
        p.fillRect(self.rect(), self.palette().base())
        if self._img is None:
            p.setPen(self.palette().placeholderText().color())
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Preview appears here")
            return
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        img = self._img.scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio,
                               Qt.TransformationMode.SmoothTransformation)  # fmt: skip
        p.drawImage((self.width() - img.width()) // 2, (self.height() - img.height()) // 2, img)


class Runner(QObject):
    """Runs one command at a time and streams its output (lines come from a reader thread)."""

    line = Signal(str)
    finished = Signal(int)

    def __init__(self):
        super().__init__()
        self.proc: subprocess.Popen | None = None

    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def start(self, cmd: list[str], env: dict, quiet_filter=None):
        self.proc = proc = subprocess.Popen(
            cmd, cwd=str(g.ROOT), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
            start_new_session=True,
        )  # fmt: skip

        def pump():
            for line in proc.stdout:
                if quiet_filter is None or quiet_filter(line):
                    self.line.emit(line.rstrip("\n"))
            self.finished.emit(proc.wait())

        threading.Thread(target=pump, daemon=True).start()

    def stop(self):
        if self.running():
            os.killpg(self.proc.pid, signal.SIGTERM)


class PromptEditor(QGroupBox):
    """A prompt file picker with an editable JSON text; edits are kept per file until saved or reverted."""

    def __init__(self, window: MainWindow, title: str, options: list[str]):
        super().__init__(title)
        self.w, self.shown = window, None
        lay = QVBoxLayout(self)
        self.pick = QComboBox()
        self.pick.addItems(options)
        self.status = label("", "info")
        lay.addWidget(hbox(self.pick, self.status))
        self.text = QPlainTextEdit()
        self.text.setFont(QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont))
        lay.addWidget(self.text)
        btns = [QPushButton(t) for t in ("Revert", "Save", "Save as...")]
        lay.addWidget(hbox(*btns))
        btns[0].clicked.connect(self.revert)
        btns[1].clicked.connect(lambda: self.save(False))
        btns[2].clicked.connect(lambda: self.save(True))
        self.pick.currentTextChanged.connect(lambda _t: self.show_file())
        self.text.textChanged.connect(self._changed)
        self._loading = False

    def _changed(self):
        if not self._loading:
            self.status.setText("edited, not saved")

    def stash(self):
        """Keep the text as an edit of the file it shows (or drop the edit when unchanged)."""
        if not self.shown:
            return
        content = self.text.toPlainText()
        if content != (g.ROOT / self.shown).read_text():
            self.w.prompt_edits[self.shown] = content
        else:
            self.w.prompt_edits.pop(self.shown, None)

    def show_file(self):
        self.stash()
        rel = self.pick.currentText()
        if not rel:
            return
        self._loading = True
        self.text.setPlainText(self.w.prompt_edits.get(rel) or (g.ROOT / rel).read_text())
        self._loading = False
        self.shown = rel
        self.status.setText("edited, not saved" if rel in self.w.prompt_edits else "")

    def revert(self):
        if self.shown:
            self.w.prompt_edits.pop(self.shown, None)
            self.shown = None
            self.show_file()

    def save(self, as_new: bool):
        content = self.text.toPlainText()
        try:
            json.loads(content)
        except json.JSONDecodeError as e:
            self.w.error(f"Not valid JSON: {e}")
            return
        target = g.ROOT / self.shown
        if as_new:
            f, _ = QFileDialog.getSaveFileName(self, "Save prompt as", str(target.parent), "JSON (*.json)")
            if not f:
                return
            target = Path(f).with_suffix(".json")
        target.write_text(content)
        self.w.prompt_edits.pop(self.shown, None)
        if target.is_relative_to(g.ROOT / "configs" / "cosmos"):
            rel = str(target.relative_to(g.ROOT))
            self.w.add_prompt_option(rel)
            self.shown = None
            self.pick.setCurrentText(rel)
        self.shown = None
        self.show_file()
        self.w.write(f"saved prompt {target}")


class MainWindow(QMainWindow):
    scene_loaded = Signal(object)
    scene_failed = Signal(str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("WorldSynth")
        self.resize(1560, 980)
        self.fields: dict[str, tuple[Callable, Callable]] = {}
        self._config_file = ""  # base YAML, shown on the Configuration tab
        self.fields["config_file"] = (lambda: self._config_file, lambda v: setattr(self, "_config_file", v))
        self.overrides: dict[str, str] = {}
        self.prompt_edits: dict[str, str] = {}
        self.comments = g.config_comments()
        self.select = GuiSettings()  # camera/surface/parts to select once a scene is read
        self.after_run: Callable[[int], None] | None = None
        self.runner = Runner()
        self.runner.line.connect(self.write)
        self.runner.finished.connect(self._run_finished)
        self.scene_loaded.connect(self.show_scene)
        self.scene_failed.connect(lambda e: self.scene_info.setText(f"could not read scene: {e}"))

        cosmos_dir = g.ROOT / "configs" / "cosmos"
        self.look_options = [str(p.relative_to(g.ROOT)) for p in sorted(cosmos_dir.rglob("*.json"))
                             if "negative" not in p.name]  # fmt: skip
        self.negatives = [str(p.relative_to(g.ROOT)) for p in sorted(cosmos_dir.rglob("negative*.json"))]

        split = QSplitter(Qt.Orientation.Horizontal)
        self.tabs = QTabWidget()
        self.tabs.setMinimumWidth(560)
        split.addWidget(self.tabs)
        split.addWidget(self._build_right())
        split.setStretchFactor(1, 1)
        split.setSizes([620, 940])
        self.setCentralWidget(split)
        self.tabs.addTab(self._build_pipeline(), "Pipeline")
        self.tabs.addTab(self._build_config(), "Configuration")
        self.tabs.addTab(self._build_prompts(), "Prompts")
        self.tabs.addTab(self._build_videos(), "Videos")
        self.tabs.currentChanged.connect(self._on_tab)
        self.statusBar().showMessage("ready")

        self.apply_settings(GuiSettings())
        self.write(
            "WorldSynth: all settings start at their defaults ('Restore last session' brings back the last run's)."
        )
        self.write("The password is never saved.")

    # --- settings <-> widgets -----------------------------------------------------------------------------------------
    def bind(self, name: str, widget):
        """Connect a GuiSettings field to its widget."""
        if isinstance(widget, QLineEdit):
            self.fields[name] = (widget.text, lambda v: widget.setText(str(v)))
        elif isinstance(widget, QSpinBox | QDoubleSpinBox):
            self.fields[name] = (widget.value, widget.setValue)
        elif isinstance(widget, QCheckBox):
            self.fields[name] = (widget.isChecked, widget.setChecked)
        elif isinstance(widget, QComboBox):

            def set_combo(v, c=widget):
                if c.findText(str(v)) < 0:
                    c.addItem(str(v))
                c.setCurrentText(str(v))

            self.fields[name] = (widget.currentText, set_combo)
        elif isinstance(widget, dict):  # radio buttons: value -> button
            buttons = widget
            self.fields[name] = (lambda: next(v for v, b in buttons.items() if b.isChecked()),
                                 lambda v: buttons[v].setChecked(True) if v in buttons else None)  # fmt: skip
        return widget

    def radios(self, name: str, options: list[tuple[object, str]], on_change=None) -> QWidget:
        group, buttons = QButtonGroup(self), {}
        for value, text in options:
            b = QRadioButton(text)
            group.addButton(b)
            buttons[value] = b
        next(iter(buttons.values())).setChecked(True)  # never none selected
        if on_change:
            for b in buttons.values():
                b.toggled.connect(lambda on: on and on_change())
        self.bind(name, buttons)
        return hbox(*buttons.values())

    def settings(self) -> GuiSettings:
        data = {name: get() for name, (get, _) in self.fields.items()}
        data.update(config_overrides=dict(self.overrides), password=self.password.text())
        for name, box in (("scene_camera", self.camera_box), ("scene_surface", self.surface_box),
                          ("scene_parts", self.parts_box)):  # fmt: skip
            data[name] = box.currentData() or ""
        return GuiSettings(**data)

    def apply_settings(self, s: GuiSettings):
        for name, (_, set_) in self.fields.items():
            set_(getattr(s, name))
        self.overrides = dict(s.config_overrides)
        self.select = s
        self.update_source_state()
        if s.use_scene and s.scene_usd:
            self.load_scene()
        self.refresh_config()

    # --- pipeline tab -------------------------------------------------------------------------------------------------
    def section(self, title: str) -> QFormLayout:
        box = QGroupBox(title)
        form = QFormLayout(box)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        self.pipe_layout.addWidget(box)
        self.sections[title] = box
        return form

    def path_field(self, name: str, kind: str, filt: str = "All files (*)") -> QWidget:
        """A text field with a Browse button; kind: dir, file or save."""
        edit = self.bind(name, QLineEdit())
        btn = QPushButton("Browse...")

        def browse():
            start = edit.text() or str(g.ROOT)
            start = start if kind == "dir" or not edit.text() else str(Path(start).parent)
            if kind == "dir":
                f = QFileDialog.getExistingDirectory(self, "Choose folder", start)
            elif kind == "save":
                f, _ = QFileDialog.getSaveFileName(self, "Choose file", start, filt)
            else:
                f, _ = QFileDialog.getOpenFileName(self, "Choose file", start, filt)
            if f:
                edit.setText(f)

        btn.clicked.connect(browse)
        box = hbox(edit, btn, stretch=False)
        return box

    def _build_pipeline(self) -> QWidget:
        page = QWidget()
        self.pipe_layout = QVBoxLayout(page)
        self.sections: dict[str, QGroupBox] = {}
        restore, defaults = QPushButton("Restore last session"), QPushButton("Defaults")
        restore.clicked.connect(self.restore_session)
        defaults.clicked.connect(lambda: self.apply_settings(GuiSettings()))
        self.pipe_layout.addWidget(hbox(restore, defaults, label("(starts with defaults)")))

        f = self.section("Source")
        f.addRow(self.radios("source", [("cad", "Render CAD parts in Isaac Sim"), ("demos", "Recorded demos (HDF5)")],
                             self.update_source_state))  # fmt: skip
        self.demo_box = QWidget()
        d = QFormLayout(self.demo_box)
        d.setContentsMargins(0, 0, 0, 0)
        d.addRow("Recorded demos", self.path_field("demos_hdf5", "file", HDF5))
        d.addRow("", label("the Mimic or teleop HDF5 file; it is only read, never changed"))
        d.addRow("Cameras", self.bind("demo_cameras", QLineEdit()))
        d.addRow("Max demos", self.bind("max_demos", spin(0, 100000, 90)))
        d.addRow("Seg classes", self.bind("seg_names", QLineEdit()))
        d.addRow("", label("Max demos 0 = all. Seg classes name the ids of &lt;camera&gt;_seg (for labels), e.g. "
                           "1=robot,2=table,3=ground,4-5=bin,10-15=soup_can"))  # fmt: skip
        guide = label(DEMOS_GUIDE, "info")
        guide.setTextFormat(Qt.TextFormat.RichText)
        d.addRow(guide)
        f.addRow(self.demo_box)

        f = self.section("Results (demos)")
        f.addRow("Cosmos demos", self.path_field("demos_out", "save", HDF5))
        f.addRow("", self.bind("keep_missing", QCheckBox("Also write demos with only some cameras done")))
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        f.addRow(line)
        f.addRow("Converter", self.path_field("convert_script", "file", "Python (*.py);;All files (*)"))
        f.addRow("Converter Python", self.path_field("convert_python", "file"))
        f.addRow("Arguments", self.bind("convert_args", QLineEdit()))
        f.addRow("Training dataset", self.path_field("convert_out", "dir"))
        f.addRow("", hbox(self.bind("cosmos_only", QCheckBox("Cosmos demos only (no recorded demos)")),
                          self.bind("convert_overwrite", QCheckBox("Overwrite"))))  # fmt: skip
        f.addRow("", label(CONVERTER_HINT))
        self.result_paths = label("", "info")
        f.addRow(self.result_paths)

        f = self.section("Parts (CAD / STL folder)")
        f.addRow("Folder", self.path_field("cad_dir", "dir"))
        units, class_from = QComboBox(), QComboBox()
        units.addItems(["auto", "mm", "m", "cm", "inch"])
        class_from.addItems(["file", "folder"])
        f.addRow("Units", self.bind("units", units))
        f.addRow("Class per", self.bind("class_from", class_from))
        f.addRow("", label("optional when the USD scene provides the parts (Scene: Parts in scene)"))

        f = self.section("Scene")
        f.addRow(self.radios("use_scene", [(False, "Built-in workcell"), (True, "USD scene")], self.update_scene_state))
        self.scene_edit = self.bind("scene_usd", QLineEdit())
        open_btn = QPushButton("Open...")
        open_btn.clicked.connect(self.browse_scene)
        self.scene_edit.editingFinished.connect(self.load_scene)
        f.addRow("USD file", hbox(self.scene_edit, open_btn, stretch=False))
        self.scene_info = label("")
        f.addRow("", self.scene_info)
        self.camera_box, self.surface_box, self.parts_box = QComboBox(), QComboBox(), QComboBox()
        f.addRow("Camera", self.camera_box)
        f.addRow("Place parts on", self.surface_box)
        f.addRow("Parts in scene", self.parts_box)
        parts_mode = self.radios("scene_parts_mode", [("place", "place randomly"), ("keep", "keep as in scene")])
        f.addRow("Scene parts", parts_mode)
        az, el = QDoubleSpinBox(), QDoubleSpinBox()
        for box, hi in ((az, 90), (el, 45)):
            box.setRange(0, hi)
            box.setDecimals(1)
            box.setSuffix(" °")
            box.setFixedWidth(90)
        motion = hbox("± azimuth", self.bind("camera_azimuth", az), "± elevation", self.bind("camera_elevation", el))
        f.addRow("Camera motion", motion)
        lights = self.bind("scene_lights_only", QCheckBox("Use only the scene's own lights"))
        f.addRow("", lights)
        self.scene_widgets = [self.scene_edit, open_btn, self.camera_box, self.surface_box, self.parts_box,
                              parts_mode, motion, lights]  # fmt: skip

        f = self.section("Output")
        f.addRow("Run name", self.bind("name", QLineEdit()))
        f.addRow("Dataset folder", self.path_field("dataset_dir", "dir"))
        f.addRow("", label("Isaac Sim output or imported demos; empty: output/&lt;run name&gt;"))
        f.addRow("Cosmos folder", self.path_field("cosmos_dir", "dir"))
        f.addRow("", label("Cosmos input videos and batch; results copied back to &lt;folder&gt;/results; "
                           "empty: output/&lt;run name&gt;_cosmos"))  # fmt: skip

        f = self.section("Dataset")
        f.addRow("Size", hbox(self.bind("clips", spin(1, 10000)), "clips", self.bind("frames", spin(1, 241)),
                              "frames per clip"))  # fmt: skip
        f.addRow("Parts per clip", hbox(self.bind("parts_min", spin(1, 50)), "to", self.bind("parts_max", spin(1, 50))))
        f.addRow("Clutter", hbox(self.bind("clutter", QCheckBox("on,")), self.bind("clutter_min", spin(0, 50)), "to",
                                 self.bind("clutter_max", spin(0, 50)), "items"))  # fmt: skip
        f.addRow("", label("all other settings: Configuration tab"))

        f = self.section("Cosmos Transfer")
        f.addRow("Resolution", self.radios("resolution", [("256", "256 (fast, ~3 min/clip)"),
                                                          ("480", "480 (sharp, ~36 min/clip)")]))  # fmt: skip
        f.addRow("Sampling", hbox(self.bind("steps", spin(10, 100)), "steps", self.bind("looks", spin(1, 10)),
                                  "looks per clip"))  # fmt: skip
        self.look_box, self.negative_box = QComboBox(), QComboBox()
        self.look_box.addItems([g.ALL_LOOKS, *self.look_options])
        self.negative_box.addItems(self.negatives)
        f.addRow("Looks used", self.bind("prompt", self.look_box))
        f.addRow("Negative", self.bind("negative", self.negative_box))
        f.addRow("", label("edit the prompt texts on the Prompts tab"))

        f = self.section("Spark (runs Cosmos)")
        f.addRow("Host / IP", self.bind("spark_host", QLineEdit()))
        f.addRow("User", self.bind("spark_user", QLineEdit()))
        self.password = QLineEdit()
        self.password.setEchoMode(QLineEdit.EchoMode.Password)
        test = QPushButton("Test connection")
        test.clicked.connect(self.test_connection)
        f.addRow("Password", hbox(self.password, test, stretch=False))
        f.addRow("", label("not saved; leave empty for SSH keys"))
        self.pipe_layout.addStretch(1)

        for name in ("name", "dataset_dir", "demos_out", "convert_out", "convert_script"):
            self.fields_widget(name).textChanged.connect(self.update_result_paths)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidget(page)
        return scroll

    def fields_widget(self, name: str) -> QLineEdit:
        """The QLineEdit of a text field (for change notifications)."""
        getter = self.fields[name][0]
        return getter.__self__

    def update_scene_state(self):
        on = self.fields["use_scene"][0]()
        for w in self.scene_widgets:
            w.setEnabled(on)

    def update_source_state(self):
        demos = self.fields["source"][0]() == "demos"
        for title, box in self.sections.items():
            box.setVisible(title not in (("Parts (CAD / STL folder)", "Scene", "Dataset") if demos
                                         else ("Results (demos)",)))  # fmt: skip
        self.demo_box.setVisible(demos)
        self.step_btns[0].setText("1 Import demos" if demos else "1 Generate (Isaac Sim)")
        self.step_btns[4].setEnabled(demos and not self.runner.running())
        if demos:
            self.update_result_paths()
        else:
            self.update_scene_state()

    def update_result_paths(self):
        s = self.settings()
        self.result_paths.setText(
            f"Step 4 writes: {g.demos_export_path(s)}\n"
            + (f"Step 5 writes: {g.training_path(s)}" if s.convert_script else "Step 5: choose a converter first")
        )

    def browse_scene(self):
        start = str(Path(self.scene_edit.text()).parent) if self.scene_edit.text() else str(g.ROOT)
        f, _ = QFileDialog.getOpenFileName(self, "Open USD scene", start, "USD (*.usd *.usda *.usdc *.usdz);;All (*)")
        if f:
            self.scene_edit.setText(f)
            self.select = GuiSettings()
            self.load_scene()

    def load_scene(self):
        path = self.scene_edit.text()
        if not path or not Path(path).is_file():
            return
        self.scene_info.setText("reading scene...")

        def work():
            from . import scene_info

            try:
                self.scene_loaded.emit(scene_info.inspect(path))
            except Exception as e:  # noqa: BLE001  (shown to the user)
                self.scene_failed.emit(str(e))

        threading.Thread(target=work, daemon=True).start()

    def show_scene(self, info):
        sel = self.select
        cams = [(g.RANDOM_CAMERA, "")] + [(f"{c.path}   (at {c.position}, f={c.focal_length:g})", c.path)
                                          for c in info.cameras]  # fmt: skip
        surfs = [(f"{s.path}   (top {s.top_z:.3f} m, {s.size[0]:.2f} × {s.size[1]:.2f} m)", s.path)
                 for s in info.surfaces]  # fmt: skip
        groups = [(g.NO_SCENE_PARTS, "")] + [
            (f"{p.path}   ({len(p.children)}: {', '.join(p.children[:4])}{', ...' if len(p.children) > 4 else ''})",
             p.path) for p in info.groups
        ]  # fmt: skip
        # defaults: the first scene camera; the second largest surface (the largest is usually the floor)
        for box, items, saved, default in ((self.camera_box, cams, sel.scene_camera, 1),
                                           (self.surface_box, surfs, sel.scene_surface, 1),
                                           (self.parts_box, groups, sel.scene_parts, 0)):  # fmt: skip
            box.clear()
            for text, data in items:
                box.addItem(text, data)
            idx = next((i for i, (_, data) in enumerate(items) if saved and data == saved), None)
            box.setCurrentIndex(idx if idx is not None else min(default, len(items) - 1))
        self.scene_info.setText(f"{len(info.cameras)} cameras, {len(info.surfaces)} surfaces; file units "
                                f"{info.meters_per_unit:g} m, {info.up_axis}-up (converted to m, z-up)")  # fmt: skip

    # --- configuration tab --------------------------------------------------------------------------------------------
    def _build_config(self) -> QWidget:
        page = QWidget()
        lay = QVBoxLayout(page)
        self.base_label = QLabel()
        lay.addWidget(hbox("Base config:", self.base_label))
        btns = [QPushButton(t) for t in ("Load YAML...", "Save as YAML...", "Revert selected", "Reset to defaults")]
        for b, fn in zip(btns, (self.load_yaml, self.save_yaml, self.revert_selected, self.reset_config), strict=True):
            b.clicked.connect(fn)
        lay.addWidget(hbox(*btns))
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Setting", "Value (double-click to edit)", "From"])
        self.tree.setColumnWidth(0, 240)
        self.tree.setColumnWidth(1, 220)
        self.tree.setAlternatingRowColors(True)
        self.tree.setSelectionMode(QTreeWidget.SelectionMode.ExtendedSelection)
        self.tree.setEditTriggers(QTreeWidget.EditTrigger.NoEditTriggers)
        self.tree.itemDoubleClicked.connect(self._edit_config)
        self.tree.itemChanged.connect(self._config_changed)
        self.tree.currentItemChanged.connect(self._config_selected)
        lay.addWidget(self.tree, 1)
        self.config_desc = label("Select a setting to see what it does.", "")
        lay.addWidget(self.config_desc)
        c = colors()
        lay.addWidget(label(f"<span style='color:{c['changed']}'><b>blue</b>: changed here</span> &nbsp; "
                            f"<span style='color:{c['file']}'>green: from the loaded YAML</span> &nbsp; "
                            f"<span style='color:{c['locked']}'>grey: set on the Pipeline tab (it wins)</span> &nbsp; "
                            "plain: default"))  # fmt: skip
        self._refreshing = False
        return page

    def refresh_config(self):
        s = self.settings()
        base = g.base_config_path(s)
        self.base_label.setText(str(base.relative_to(g.ROOT)) if base.is_relative_to(g.ROOT) else str(base))
        try:
            defaults_flat, from_file = g.config_values(None), g.config_values(base)
            current = g.config_values(base, self.overrides)
        except Exception as e:  # noqa: BLE001  (shown to the user)
            self.error(f"Configuration error: {e}")
            return
        locked, c = g.pipeline_keys(s), colors()
        expanded = {k for k, it in getattr(self, "_config_nodes", {}).items() if it.isExpanded()}
        first = not getattr(self, "_config_nodes", None)
        cur = self.tree.currentItem()
        selected = cur.data(0, Qt.ItemDataRole.UserRole) if cur else None
        self._refreshing = True
        self.tree.clear()
        self._config_nodes, items = {}, {}
        bold = QFont()
        bold.setBold(True)
        for key, value in current.items():
            parent = self.tree.invisibleRootItem()
            names = key.split(".")
            for depth in range(1, len(names)):
                node = ".".join(names[:depth])
                if node not in self._config_nodes:
                    it = QTreeWidgetItem(parent, [names[depth - 1]])
                    it.setExpanded(first or node in expanded)
                    self._config_nodes[node] = it
                parent = self._config_nodes[node]
            if key in locked:
                text, src, color = locked[key], "Pipeline tab", QColor(c["locked"])
            elif key in self.overrides:
                text, src, color = g.format_value(value), "changed here", QColor(c["changed"])
            elif from_file[key] != defaults_flat[key]:
                text, src, color = g.format_value(value), "YAML file", QColor(c["file"])
            else:
                text, src, color = g.format_value(value), "default", None
            it = QTreeWidgetItem(parent, [names[-1], text, src])
            it.setData(0, Qt.ItemDataRole.UserRole, key)
            it.setFlags(it.flags() | Qt.ItemFlag.ItemIsEditable)
            if color is not None:
                for col in range(3):
                    it.setForeground(col, color)
            if src == "changed here":
                it.setFont(1, bold)
            items[key] = it
        self._refreshing = False
        if selected in items:
            self.tree.setCurrentItem(items[selected])
            self.tree.scrollToItem(items[selected])

    def _config_key(self, item) -> str | None:
        return item.data(0, Qt.ItemDataRole.UserRole) if item is not None else None

    def _config_selected(self, item, _prev=None):
        key = self._config_key(item)
        if key:
            self.config_desc.setText(f"{key}: {self.comments.get(key, '')}".rstrip(": "))

    def _edit_config(self, item, _col):
        key = self._config_key(item)
        if not key:
            return
        if key in g.pipeline_keys(self.settings()):
            QMessageBox.information(self, "WorldSynth", f"{key} is set on the Pipeline tab; change it there.")
            return
        self.tree.editItem(item, 1)

    def _config_changed(self, item, col):
        key = self._config_key(item)
        if self._refreshing or col != 1 or not key:
            return
        text = item.text(1).strip()
        try:
            base = g.base_config_path(self.settings())
            value = g.config_values(base, dict(self.overrides, **{key: text}))[key]
            base_value = g.config_values(base)[key]
        except Exception as e:  # noqa: BLE001  (shown to the user)
            self.error(f"{key}: {e}")
            QTimer.singleShot(0, self.refresh_config)
            return
        if value == base_value:
            self.overrides.pop(key, None)
        else:
            self.overrides[key] = text
        QTimer.singleShot(0, self.refresh_config)  # not while the item's editor is still closing

    def revert_selected(self):
        for item in self.tree.selectedItems():
            self.overrides.pop(self._config_key(item) or "", None)
        self.refresh_config()

    def load_yaml(self):
        f, _ = QFileDialog.getOpenFileName(self, "Load YAML", str(g.ROOT / "configs"), "YAML (*.yaml *.yml);;All (*)")
        if not f:
            return
        try:
            g.config_values(Path(f))
        except Exception as e:  # noqa: BLE001  (shown to the user)
            self.error(f"Cannot use {f}: {e}")
            return
        self.fields["config_file"][1](f)
        if self.overrides and not self.ask("Keep the values changed here on top of it?"):
            self.overrides.clear()
        self.refresh_config()

    def save_yaml(self):
        f, _ = QFileDialog.getSaveFileName(self, "Save configuration", str(g.ROOT / "configs"), "YAML (*.yaml)")
        if not f:
            return
        f = str(Path(f).with_suffix(".yaml"))
        try:
            g.write_run_config(self.settings(), Path(f))
        except Exception as e:  # noqa: BLE001  (shown to the user)
            self.error(str(e))
            return
        self.fields["config_file"][1](f)
        self.overrides.clear()
        self.refresh_config()
        self.write(f"saved configuration to {f} (now the base config)")

    def reset_config(self):
        self.fields["config_file"][1]("")
        self.overrides.clear()
        self.refresh_config()

    # --- prompts tab --------------------------------------------------------------------------------------------------
    def _build_prompts(self) -> QWidget:
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.addWidget(label(PROMPTS_HINT))
        self.look_editor = PromptEditor(self, "Look prompt (positive)", self.look_options)
        self.neg_editor = PromptEditor(self, "Negative prompt", self.negatives)
        first = next((o for o in self.look_options if o.startswith("configs/cosmos/prompts/")), None)
        if first:
            self.look_editor.pick.setCurrentText(first)
        # the negative prompt shown here is the one the run uses
        self.neg_editor.pick.currentTextChanged.connect(self.negative_box.setCurrentText)
        lay.addWidget(self.look_editor, 3)
        lay.addWidget(self.neg_editor, 2)
        return page

    def add_prompt_option(self, rel: str):
        """A prompt file saved under configs/cosmos becomes selectable."""
        negative = "negative" in Path(rel).name
        options = self.negatives if negative else self.look_options
        if rel not in options:
            options.append(rel)
            for box in (
                (self.negative_box, self.neg_editor.pick) if negative else (self.look_box, self.look_editor.pick)
            ):
                box.addItem(rel)

    def all_prompt_edits(self) -> dict[str, str]:
        for ed in (self.look_editor, self.neg_editor):
            ed.stash()
        return dict(self.prompt_edits)

    # --- videos tab ---------------------------------------------------------------------------------------------------
    def _build_videos(self) -> QWidget:
        page = QWidget()
        lay = QVBoxLayout(page)
        form = QFormLayout()
        self.vid_data, self.vid_cosmos = QLineEdit(), QLineEdit()
        for text, edit in (("Dataset", self.vid_data), ("Cosmos folder", self.vid_cosmos)):
            btn = QPushButton("Browse...")
            btn.clicked.connect(lambda _=False, e=edit: self._browse_video_dir(e))
            form.addRow(text, hbox(edit, btn, stretch=False))
        lay.addLayout(form)
        use, refresh = QPushButton("Use the run's folders"), QPushButton("Refresh")
        use.clicked.connect(self.use_run_folders)
        refresh.clicked.connect(self.refresh_videos)
        lay.addWidget(hbox(use, refresh))
        self.vid_tree = QTreeWidget()
        self.vid_tree.setHeaderHidden(True)
        self.vid_tree.currentItemChanged.connect(lambda it, _p: self.select_video(it))
        lay.addWidget(self.vid_tree, 1)
        self.view_box, self.speed_box = QComboBox(), QComboBox()
        self.speed_box.addItems(["0.25x", "0.5x", "1x", "2x"])
        self.speed_box.setCurrentText("1x")
        self.view_box.currentTextChanged.connect(lambda _t: self.seek_video(self.vid_pos))
        lay.addWidget(hbox("View", self.view_box, "Speed", self.speed_box))
        prev_b, self.play_btn, next_b = QPushButton("◀"), QPushButton("Play"), QPushButton("▶")
        for b in (prev_b, next_b):
            b.setFixedWidth(40)
        prev_b.clicked.connect(lambda: self.step_video(-1))
        next_b.clicked.connect(lambda: self.step_video(1))
        self.play_btn.clicked.connect(self.toggle_play)
        self.vid_slider = QSlider(Qt.Orientation.Horizontal)
        self.vid_slider.valueChanged.connect(lambda i: i != self.vid_pos and self.seek_video(i))
        self.vid_label = QLabel("0 / 0")
        lay.addWidget(hbox(prev_b, self.play_btn, next_b, self.vid_slider, self.vid_label, stretch=False))
        lay.addWidget(label(VIDEOS_HINT))
        self.vid_sources, self.vid_src, self.vid_n, self.vid_pos, self.vid_cache = {}, None, 0, 0, {}
        self.vid_timer = QTimer(self)
        self.vid_timer.timeout.connect(lambda: self.seek_video((self.vid_pos + 1) % max(1, self.vid_n)))
        return page

    def _browse_video_dir(self, edit: QLineEdit):
        d = QFileDialog.getExistingDirectory(self, "Choose folder", edit.text() or str(g.ROOT))
        if d:
            edit.setText(d)
            self.refresh_videos()

    def use_run_folders(self):
        s = self.settings()
        self.vid_data.setText(str(g.dataset_path(s)))
        self.vid_cosmos.setText(str(g.cosmos_path(s)))
        self.refresh_videos()

    def refresh_videos(self):
        from . import videos as vid

        self.stop_video()
        self.vid_tree.clear()
        data = Path(self.vid_data.text()) if self.vid_data.text() else None
        cos = Path(self.vid_cosmos.text()) if self.vid_cosmos.text() else None
        sources = vid.find_sources(data, cos)
        self.vid_sources = {src.key: src for src in sources}
        groups, first = {}, None
        for src in sources:
            if src.group not in groups:
                groups[src.group] = QTreeWidgetItem(self.vid_tree, [src.group])
                groups[src.group].setExpanded(True)
            it = QTreeWidgetItem(groups[src.group], [src.name])
            it.setData(0, Qt.ItemDataRole.UserRole, src.key)
            first = first or it
        if first is None:
            QTreeWidgetItem(self.vid_tree, ["no videos found in these folders (yet)"])
        else:  # show the first frame of the first video
            self.vid_tree.setCurrentItem(first)

    def select_video(self, item):
        from . import videos as vid

        key = item.data(0, Qt.ItemDataRole.UserRole) if item is not None else None
        if key not in self.vid_sources:
            return
        self.stop_video()
        src = self.vid_src = self.vid_sources[key]
        self.vid_cache, self.vid_n = {}, vid.frame_count(src)
        view = self.view_box.currentText()
        self.view_box.blockSignals(True)
        self.view_box.clear()
        self.view_box.addItems(list(src.views))
        self.view_box.setCurrentText(view if view in src.views else
                                     "rgb + labels" if "rgb + labels" in src.views else src.views[0])  # fmt: skip
        self.view_box.blockSignals(False)
        self.vid_slider.setRange(0, max(0, self.vid_n - 1))
        self.seek_video(0)

    def seek_video(self, i: int):
        from . import videos as vid

        src = self.vid_src
        if src is None or self.vid_n == 0:
            return
        i = max(0, min(self.vid_n - 1, i))
        self.vid_pos = i
        key = (self.view_box.currentText(), i)
        if key not in self.vid_cache:
            img = vid.read_frame(src, i, key[0])
            if img is None:
                return
            if len(self.vid_cache) > 150:
                self.vid_cache = {}
            self.vid_cache[key] = img
        self.show_array(self.vid_cache[key], f"{src.group}: {src.name}   ({key[0]})")
        self.vid_label.setText(f"{i + 1} / {self.vid_n}")
        self.vid_slider.blockSignals(True)
        self.vid_slider.setValue(i)
        self.vid_slider.blockSignals(False)

    def step_video(self, d: int):
        self.stop_video()
        self.seek_video(self.vid_pos + d)

    def toggle_play(self):
        if self.vid_timer.isActive():
            self.stop_video()
        elif self.vid_src is not None:
            fps = self.vid_src.fps * float(self.speed_box.currentText().rstrip("x"))
            self.vid_timer.start(max(10, int(1000 / fps)))
            self.play_btn.setText("Pause")

    def stop_video(self):
        self.vid_timer.stop()
        self.play_btn.setText("Play")

    # --- right side ---------------------------------------------------------------------------------------------------
    def _build_right(self) -> QWidget:
        page = QWidget()
        lay = QVBoxLayout(page)
        self.preview_btn, self.run_btn, self.stop_btn = (QPushButton("Preview frame"), QPushButton("Run all steps"),
                                                         QPushButton("Stop"))  # fmt: skip
        self.run_btn.setObjectName("primary")
        self.stop_btn.setObjectName("danger")
        self.stop_btn.setEnabled(False)
        self.preview_btn.clicked.connect(self.run_preview)
        self.run_btn.clicked.connect(lambda: self.run_stages(g.run_all_stages(self.settings())))
        self.stop_btn.clicked.connect(self.stop)
        open_data, open_cosmos = QPushButton("Open dataset"), QPushButton("Open Cosmos folder")
        open_data.clicked.connect(lambda: self.open_dir(g.dataset_path(self.settings())))
        open_cosmos.clicked.connect(lambda: self.open_dir(g.cosmos_path(self.settings())))
        lay.addWidget(hbox(self.preview_btn, self.run_btn, self.stop_btn, open_data, open_cosmos))
        self.step_btns = []
        for text, stage in STEP_LABELS:
            b = QPushButton(text)
            b.setObjectName("step")
            b.clicked.connect(lambda _=False, st=stage: self.run_stages((st,)))
            self.step_btns.append(b)
        lay.addWidget(hbox("Steps:", *self.step_btns))
        split = QSplitter(Qt.Orientation.Vertical)
        top = QWidget()
        tl = QVBoxLayout(top)
        tl.setContentsMargins(0, 0, 0, 0)
        self.image = ImageView()
        self.caption = label("")
        tl.addWidget(self.image, 1)
        tl.addWidget(self.caption)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(20000)
        self.log.setFont(QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont))
        split.addWidget(top)
        split.addWidget(self.log)
        split.setSizes([600, 320])
        lay.addWidget(split, 1)
        return page

    def write(self, line: str):
        self.log.appendPlainText(line.rstrip("\n"))

    def show_array(self, img: np.ndarray, text: str):
        self.image.set_bgr(img)
        self.caption.setText(text)

    def show_image(self, path: Path, label_file: Path | None, text: str):
        import cv2

        from .viz import draw

        img = cv2.imread(str(path))
        if img is None:
            self.write(f"image not found: {path}")
            return
        if label_file is not None and label_file.exists():
            img = draw(img, json.loads(label_file.read_text()))
        self.show_array(img, text)

    def error(self, text: str):
        QMessageBox.critical(self, "WorldSynth", text)

    def ask(self, text: str) -> bool:
        return QMessageBox.question(self, "WorldSynth", text) == QMessageBox.StandardButton.Yes

    # --- processes ----------------------------------------------------------------------------------------------------
    def set_busy(self, busy: bool, what: str = ""):
        for b in [self.preview_btn, self.run_btn, *self.step_btns]:
            b.setEnabled(not busy)
        if not busy and self.fields["source"][0]() != "demos":
            self.step_btns[4].setEnabled(False)  # convert is for recorded demos only
        self.stop_btn.setEnabled(busy)
        self.statusBar().showMessage(what if busy else "ready")

    def start(self, cmd: list[str], env: dict, what: str, after=None, quiet: bool = False):
        """Run ``cmd`` and stream its output to the log; ``quiet`` keeps only WorldSynth progress and errors."""
        import re

        noise = re.compile(r"^\[worldsynth\]|Error|Traceback|^\s+File |Exception|^\S+Error")
        self.write(f"\n$ {' '.join(shlex.quote(c) for c in cmd)}")
        self.after_run = after
        self.set_busy(True, what)
        self.runner.start(cmd, env, noise.search if quiet else None)

    def _run_finished(self, code: int):
        self.write(f"[exit code {code}]")
        self.set_busy(False)
        if self.after_run:
            self.after_run(code)

    def prepare(self, s: GuiSettings, stages=g.STAGES, for_preview=False) -> Path | None:
        """Check the settings and write the merged config; None if something is wrong."""
        errors = g.validate(s, stages, for_preview)
        if errors:
            self.error("\n".join(errors))
            return None
        g.save_settings(s)
        config = g.write_run_config(s)
        if s.config_overrides:
            self.write(f"config: {g.base_config_path(s)} + {len(s.config_overrides)} change(s) -> {config}")
        return config

    def run_preview(self):
        s = self.settings()
        if s.source == "demos":
            errors = g.validate(s, for_preview=True)
            if errors:
                self.error("\n".join(errors))
                return
            try:
                self.show_array(*g.demo_preview(s))
            except (ValueError, OSError) as e:
                self.error(str(e))
            return
        config = self.prepare(s, for_preview=True)
        if config is None:
            return
        out = g.ROOT / "cache" / "gui_preview"
        env = dict(os.environ)
        env["ISAACLAB_DIR"] = env.get("ISAACLAB_DIR", str(g.ROOT.parent / "IsaacLab"))

        def done(code):
            clip = out / "clips" / "clip_0000"
            if code == 0 and (clip / "rgb" / "000000.png").exists():
                self.show_image(clip / "rgb" / "000000.png", clip / "labels" / "000000.json",
                                "preview: frame 0 with labels")  # fmt: skip
            else:
                self.write("preview failed, see the log above")

        self.start(g.preview_command(s, config, out, g.isaaclab_python()), env,
                   "rendering preview (Isaac Sim starts, ~30 s)...", done, quiet=True)  # fmt: skip

    def run_stages(self, stages: tuple[str, ...]):
        s = self.settings()
        config = self.prepare(s, stages)
        if config is None:
            return
        try:
            looks, negative = g.write_prompts(s, self.all_prompt_edits())
        except ValueError as e:
            self.error(str(e))
            return
        if self.prompt_edits and "prepare" in stages:
            self.write(f"prompts with your edits: {', '.join(sorted(self.prompt_edits))}")
        data, batch = g.dataset_path(s), g.cosmos_path(s)

        def done(code):
            if code != 0:
                self.write("stopped with an error, see the log above")
                return
            clips = sorted((data / "clips").glob("*/rgb")) if (data / "clips").is_dir() else []
            clip = clips[0].parent if clips else data / "clips" / "clip_0000"
            img_dir = "rgb_cosmos" if "fetch" in stages else "rgb"
            frames = sorted((clip / img_dir).glob("*.png"))
            if frames and ("generate" in stages or "fetch" in stages):
                mid = frames[len(frames) // 2]
                self.show_image(
                    mid, clip / "labels" / f"{mid.stem}.json", f"{s.name}: {img_dir}/{mid.name} with labels"
                )
            next_step = {"generate": "2 Make Cosmos videos", "prepare": "3 Send to Cosmos", "send": "4 Fetch results"}
            if s.source == "demos":
                next_step["fetch"] = "5 Convert for training"
            last = stages[-1]
            if last == "convert":
                self.write(f"training data: {g.training_path(s)}")
            self.write(f"done ({', '.join(stages)}). dataset: {data}  Cosmos folder: {batch}"
                       + (f"  next: {next_step[last]}" if last in next_step else ""))  # fmt: skip

        what = "running all steps" if stages == g.run_all_stages(s) else f"running step: {', '.join(stages)}"
        cmd = g.pipeline_command(s, config, stages=stages, prompts=looks, negative=negative)
        self.start(cmd, g.ssh_env(s), f"{what} ({s.name})...", done)

    def test_connection(self):
        s = self.settings()
        if not (s.spark_host and s.spark_user):
            self.error("Enter the Spark host and user.")
            return
        self.start(g.connection_check_command(s), g.ssh_env(s), "testing SSH connection...")

    def stop(self):
        if self.runner.running():
            self.runner.stop()
            self.write("stopped. A Cosmos job already started on the Spark keeps running; use '4 Fetch results' later "
                       "to wait for it and copy the results back.")  # fmt: skip

    def open_dir(self, path: Path):
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path if path.exists() else path.parent)))

    def restore_session(self):
        if not g.SETTINGS_FILE.exists():
            QMessageBox.information(self, "WorldSynth", "No saved session yet (settings are saved when you run).")
            return
        self.apply_settings(g.load_settings())
        self.write(f"restored the settings of the last session from {g.SETTINGS_FILE}")

    def _on_tab(self, index: int):
        if index == TAB_CONFIG:
            self.refresh_config()
        elif index == TAB_PROMPTS:
            prompt = self.look_box.currentText()
            if prompt != g.ALL_LOOKS:
                self.look_editor.pick.setCurrentText(prompt)
            self.neg_editor.pick.setCurrentText(self.negative_box.currentText())
            for ed in (self.look_editor, self.neg_editor):
                if ed.shown != ed.pick.currentText():
                    ed.show_file()
        elif index == TAB_VIDEOS and not self.vid_data.text():
            self.use_run_folders()

    def closeEvent(self, event):  # noqa: N802  (Qt API)
        if self.runner.running():
            if not self.ask("A run is in progress. Stop it and quit?"):
                event.ignore()
                return
            self.stop()
        event.accept()


def create_window(argv: list[str]) -> tuple[QApplication, MainWindow]:
    app = QApplication.instance() or QApplication(argv)
    app.setApplicationName("WorldSynth")
    app.setStyle("Fusion")
    app.setStyleSheet(STYLE.substitute(colors()))
    win = MainWindow()
    args = argv[1:]
    if "--restore" in args:
        QTimer.singleShot(200, win.restore_session)
    if "--preview" in args:  # render a preview of the restored setup right away
        QTimer.singleShot(1500, win.run_preview)
    for flag, tab in (("--config-tab", TAB_CONFIG), ("--prompts-tab", TAB_PROMPTS), ("--videos-tab", TAB_VIDEOS)):
        if flag in args:
            QTimer.singleShot(400, lambda t=tab: win.tabs.setCurrentIndex(t))
    return app, win


def run(argv: list[str]) -> int:
    app, win = create_window(argv)
    win.show()
    return app.exec()
