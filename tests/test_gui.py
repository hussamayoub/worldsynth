import json

from worldsynth import gui
from worldsynth.gui import GuiSettings


def _settings(tmp_path, **kw):
    cad = tmp_path / "cad"
    cad.mkdir(exist_ok=True)
    scene = tmp_path / "cell.usda"
    scene.write_text("#usda 1.0\n")
    base = dict(cad_dir=str(cad), name="run_x", spark_host="10.0.0.5", spark_user="u", password="secret")
    base.update(kw)
    s = GuiSettings(**base)
    if s.use_scene and not s.scene_usd:
        s.scene_usd = str(scene)
    return s


def test_password_is_never_saved(tmp_path):
    s = _settings(tmp_path)
    path = tmp_path / "gui.json"
    gui.save_settings(s, path)
    assert "secret" not in path.read_text() and "password" not in json.loads(path.read_text())
    loaded = gui.load_settings(path)
    assert loaded.password == "" and loaded.spark_host == "10.0.0.5"
    assert "secret" not in " ".join(gui.pipeline_command(s, tmp_path))


def test_pipeline_command_with_scene_camera(tmp_path):
    s = _settings(
        tmp_path, use_scene=True, scene_surface="/Cell/Workbench/Top", scene_camera="/Cell/Cameras/BenchCam",
        camera_azimuth=12, camera_elevation=4, clutter_min=2, clutter_max=5, extra="scene.min_spacing=0.005\n# note",
    )  # fmt: skip
    cmd = gui.pipeline_command(s, tmp_path)
    joined = " ".join(cmd)
    assert cmd[0].endswith("run_pipeline.sh")
    assert "--spark u@10.0.0.5" in joined and "--name run_x" in joined
    assert f"--scene {s.scene_usd} --scene-surface /Cell/Workbench/Top --scene-camera /Cell/Cameras/BenchCam" in joined
    sets = [cmd[i + 1] for i, c in enumerate(cmd) if c == "--set"]
    assert "camera.azimuth_sweep=[-12,12]" in sets and "camera.elevation_sweep=[-4,4]" in sets
    assert "randomization.clutter.count=[2,5]" in sets and "scene.min_spacing=0.005" in sets
    assert not any(x.startswith("#") for x in sets)


def test_builtin_workcell_has_no_scene_args(tmp_path):
    s = _settings(tmp_path, clutter=False, render_only=True)
    cmd = gui.pipeline_command(s, tmp_path)
    assert "--scene" not in cmd and "--no-clutter" in cmd and "--skip-cosmos" in cmd
    assert not any(c.startswith("camera.") for c in cmd)


def test_validate(tmp_path):
    assert gui.validate(_settings(tmp_path)) == []
    assert gui.validate(_settings(tmp_path, use_scene=True))  # no surface chosen
    assert gui.validate(_settings(tmp_path, name="bad name"))
    assert gui.validate(_settings(tmp_path, spark_host="")) and not gui.validate(
        _settings(tmp_path, spark_host="", render_only=True)
    )
    assert gui.validate(_settings(tmp_path, name="bad name"), for_preview=True) == []


def test_ssh_env_passes_password_only_through_askpass(tmp_path, monkeypatch):
    monkeypatch.setattr(gui, "ASKPASS", tmp_path / "askpass.sh")
    env = gui.ssh_env(_settings(tmp_path), base={})
    assert env["WS_SSH_PASSWORD"] == "secret" and env["SSH_ASKPASS_REQUIRE"] == "force"
    assert "secret" not in (tmp_path / "askpass.sh").read_text()
    assert "WS_SSH_PASSWORD" not in gui.ssh_env(_settings(tmp_path, password=""), base={})


def test_preview_command(tmp_path):
    s = _settings(tmp_path, use_scene=True, scene_surface="/Cell/Workbench/Top", resolution="480")
    cmd = gui.preview_command(s, tmp_path / "prev", "/py", tmp_path)
    assert cmd[0] == "/py" and "dataset.frames_per_clip=1" in cmd and "dataset.resolution=[1248,720]" in cmd
    assert "scene.scene_surface=/Cell/Workbench/Top" in cmd and not any("scene_camera" in c for c in cmd)
