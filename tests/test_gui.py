import json

import yaml

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


def _sets(cmd):
    return [cmd[i + 1] for i, c in enumerate(cmd) if c == "--set"]


def test_password_is_never_saved(tmp_path):
    s = _settings(tmp_path, config_overrides={"dataset.fps": "10"})
    path = tmp_path / "gui.json"
    gui.save_settings(s, path)
    assert "secret" not in path.read_text() and "password" not in json.loads(path.read_text())
    loaded = gui.load_settings(path)
    assert (
        loaded.password == "" and loaded.spark_host == "10.0.0.5" and loaded.config_overrides == {"dataset.fps": "10"}
    )
    assert "secret" not in " ".join(gui.pipeline_command(s, tmp_path / "c.yaml", tmp_path))


def test_pipeline_command_with_scene_camera_and_folders(tmp_path):
    s = _settings(
        tmp_path, use_scene=True, scene_surface="/Cell/Workbench/Top", scene_camera="/Cell/Cameras/BenchCam",
        camera_azimuth=12, camera_elevation=4, clutter_min=2, clutter_max=5,
        dataset_dir=str(tmp_path / "data"), cosmos_dir=str(tmp_path / "cos"),
    )  # fmt: skip
    cmd = gui.pipeline_command(s, tmp_path / "run.yaml", tmp_path)
    joined = " ".join(cmd)
    assert cmd[0].endswith("run_pipeline.sh")
    assert "--spark u@10.0.0.5" in joined and "--name run_x" in joined and f"--config {tmp_path / 'run.yaml'}" in joined
    assert f"--dataset-dir {tmp_path / 'data'} --cosmos-dir {tmp_path / 'cos'}" in joined
    assert f"--scene {s.scene_usd} --scene-surface /Cell/Workbench/Top --scene-camera /Cell/Cameras/BenchCam" in joined
    sets = _sets(cmd)
    assert "camera.azimuth_sweep=[-12, 12]" in sets and "camera.elevation_sweep=[-4, 4]" in sets
    assert "randomization.clutter.count=[2, 5]" in sets


def test_default_folders_follow_run_name(tmp_path):
    s = _settings(tmp_path, clutter=False)
    assert gui.dataset_path(s, tmp_path) == tmp_path / "output" / "run_x"
    assert gui.cosmos_path(s, tmp_path) == tmp_path / "output" / "run_x_cosmos"
    cmd = gui.pipeline_command(s, tmp_path / "c.yaml", tmp_path, stages=("generate",))
    assert "--scene" not in cmd and "--no-clutter" in cmd and cmd[cmd.index("--stages") + 1] == "generate"
    assert not any(c.startswith("camera.") for c in _sets(cmd))
    assert gui.pipeline_command(s, tmp_path / "c.yaml", tmp_path)[5:7] == ["--stages", "generate,prepare,send,fetch"]


def test_run_config_merges_yaml_and_overrides(tmp_path):
    base = tmp_path / "base.yaml"
    base.write_text("dataset:\n  fps: 15\ncamera:\n  distance: [0.3, 0.5]\n")
    s = _settings(tmp_path, config_file=str(base), config_overrides={"camera.distance": "[0.2, 0.4]"})
    out = gui.write_run_config(s, tmp_path / "run.yaml")
    data = yaml.safe_load(out.read_text())
    assert data["dataset"]["fps"] == 15  # from the base YAML
    assert data["camera"]["distance"] == [0.2, 0.4]  # changed on the Configuration tab
    assert data["dataset"]["frames_per_clip"] == 61  # default
    assert gui.validate(s) == []
    assert gui.validate(_settings(tmp_path, config_overrides={"dataset.nope": "1"}))  # unknown key


def test_pipeline_keys_and_config_comments(tmp_path):
    keys = gui.pipeline_keys(_settings(tmp_path, resolution="480", clips=7), tmp_path)
    assert keys["dataset.resolution"] == "[1248, 720]" and keys["dataset.num_clips"] == "7"
    assert keys["output.dir"] == str(tmp_path / "output" / "run_x")
    comments = gui.config_comments()
    assert "seed" in comments["output.seed"] and "clutter" in comments["randomization.clutter.enabled"]
    flat = gui.config_values(None)
    assert set(comments) <= set(flat) | {k.rsplit(".", 1)[0] for k in flat}  # comments refer to real keys
    assert gui.format_value((0.25, 0.5)) == "[0.25, 0.5]" and gui.format_value(None) == "null"


def test_validate(tmp_path):
    assert gui.validate(_settings(tmp_path)) == []
    assert gui.validate(_settings(tmp_path, use_scene=True))  # no surface chosen
    assert gui.validate(_settings(tmp_path, name="bad name"))
    no_spark = _settings(tmp_path, spark_host="")
    assert gui.validate(no_spark) and not gui.validate(no_spark, ("generate", "prepare"))
    assert gui.validate(no_spark, ("fetch",))
    # later steps do not need the CAD folder any more
    assert gui.validate(_settings(tmp_path, cad_dir="/no/such/dir"), ("send", "fetch")) == []
    assert gui.validate(_settings(tmp_path, name="bad name"), for_preview=True) == []


def test_ssh_env_passes_password_only_through_askpass(tmp_path, monkeypatch):
    monkeypatch.setattr(gui, "ASKPASS", tmp_path / "askpass.sh")
    env = gui.ssh_env(_settings(tmp_path), base={})
    assert env["WS_SSH_PASSWORD"] == "secret" and env["SSH_ASKPASS_REQUIRE"] == "force"
    assert "secret" not in (tmp_path / "askpass.sh").read_text()
    assert "WS_SSH_PASSWORD" not in gui.ssh_env(_settings(tmp_path, password=""), base={})


def test_preview_command(tmp_path):
    s = _settings(tmp_path, use_scene=True, scene_surface="/Cell/Workbench/Top", resolution="480")
    cmd = gui.preview_command(s, tmp_path / "run.yaml", tmp_path / "prev", "/py", tmp_path)
    assert cmd[0] == "/py" and cmd[cmd.index("--config") + 1] == str(tmp_path / "run.yaml")
    assert "dataset.frames_per_clip=1" in cmd and "dataset.resolution=[1248, 720]" in cmd
    assert f"output.dir={tmp_path / 'prev'}" in cmd
    assert "scene.scene_surface=/Cell/Workbench/Top" in cmd and not any("scene_camera" in c for c in cmd)


def test_scene_parts_make_cad_folder_optional(tmp_path):
    s = _settings(tmp_path, cad_dir="", use_scene=True, scene_surface="/Cell/Workbench/Top",
                  scene_parts="/Cell/Parts", scene_parts_mode="keep")  # fmt: skip
    assert gui.validate(s) == []
    cmd = gui.pipeline_command(s, tmp_path / "c.yaml", tmp_path)
    assert "--cad" not in cmd and " ".join(cmd).count("--scene-parts /Cell/Parts --scene-parts-mode keep") == 1
    keys = gui.pipeline_keys(s, tmp_path)
    assert keys["cad.dir"] == "null" and keys["scene.scene_parts_mode"] == "keep"
    assert gui.validate(_settings(tmp_path, cad_dir=""))  # no CAD folder and no scene parts


def test_write_prompts_uses_edits_and_checks_json(tmp_path):
    root = tmp_path
    (root / "configs" / "cosmos" / "prompts").mkdir(parents=True)
    (root / "configs" / "cosmos" / "prompts" / "a.json").write_text('{"look": "a"}')
    (root / "configs" / "cosmos" / "prompts" / "b.json").write_text('{"look": "b"}')
    (root / "configs" / "cosmos" / "negative_prompt.json").write_text('{"avoid": "x"}')
    s = _settings(tmp_path)  # all looks
    edits = {"configs/cosmos/prompts/b.json": '{"look": "B edited"}'}
    looks, negative = gui.write_prompts(s, edits, tmp_path / "out", root)
    assert [p.name for p in looks] == ["a.json", "b.json"]
    assert json.loads(looks[1].read_text()) == {"look": "B edited"}
    assert json.loads((root / "configs/cosmos/prompts/b.json").read_text()) == {"look": "b"}  # file untouched
    assert json.loads(negative.read_text()) == {"avoid": "x"}
    one = _settings(tmp_path, prompt="configs/cosmos/prompts/a.json")
    assert [p.name for p in gui.write_prompts(one, {}, tmp_path / "out", root)[0]] == ["a.json"]
    import pytest

    with pytest.raises(ValueError, match="not valid JSON"):
        gui.write_prompts(s, {"configs/cosmos/negative_prompt.json": "{broken"}, tmp_path / "out", root)
    cmd = gui.pipeline_command(s, tmp_path / "c.yaml", root, prompts=looks, negative=negative)
    assert cmd.count("--prompts") == 2 and cmd[cmd.index("--negative") + 1] == str(negative)


def test_demos_source_command(tmp_path):
    h5 = tmp_path / "demos.hdf5"
    h5.write_bytes(b"")
    s = _settings(tmp_path, source="demos", demos_hdf5=str(h5), demo_cameras="scene_cam, wrist_cam", max_demos=5,
                  seg_names="10-15=can", cad_dir="")  # fmt: skip
    assert gui.validate(s) == []
    cmd = gui.pipeline_command(s, tmp_path / "c.yaml", tmp_path)
    joined = " ".join(cmd)
    assert f"--demos {h5} --demo-cameras scene_cam,wrist_cam --max-demos 5 --seg-names 10-15=can" in joined
    assert "--scene" not in cmd and not _sets(cmd)
    assert gui.validate(_settings(tmp_path, source="demos", demos_hdf5=str(tmp_path / "missing.hdf5")))
    assert gui.run_all_stages(s) == gui.STAGES and "--convert-script" not in cmd
    assert gui.validate(s, ("convert",))  # no converter chosen
    conv = tmp_path / "conv.py"
    conv.write_text("")
    s.convert_script, s.convert_out, s.cosmos_only, s.keep_missing = str(conv), str(tmp_path / "lr"), True, True
    assert gui.validate(s, ("convert",)) == [] and gui.run_all_stages(s)[-1] == "convert"
    joined = " ".join(gui.pipeline_command(s, tmp_path / "c.yaml", tmp_path, stages=gui.run_all_stages(s)))
    assert f"--convert-script {conv}" in joined and f"--convert-out {tmp_path / 'lr'}" in joined
    assert (
        "--cosmos-only" in joined
        and "--keep-missing" in joined
        and "--stages generate,prepare,send,fetch,convert" in joined
    )
    assert gui.demos_export_path(s, tmp_path) == tmp_path / "output" / "run_x" / "demos_cosmos.hdf5"
    assert gui.training_path(s, tmp_path) == tmp_path / "lr"
    assert gui.validate(_settings(tmp_path, convert_script=str(conv)), ("convert",))  # CAD source
    assert gui.validate(_settings(tmp_path, source="demos", convert_script=str(conv), convert_args="--x"), ("convert",))


def test_qt_window_round_trips_settings(tmp_path, monkeypatch):
    import dataclasses

    import pytest

    pytest.importorskip("PySide6")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from worldsynth import gui_qt

    _app, win = gui_qt.create_window(["worldsynth"])
    s = GuiSettings(
        source="demos", demos_hdf5="/d.hdf5", max_demos=3, seg_names="10-15=can", keep_missing=True, cosmos_only=True,
        convert_script="/c.py", convert_args="--in {inputs} --out {output}", cad_dir="/cad", units="mm",
        class_from="folder", scene_parts_mode="keep", scene_lights_only=False, camera_azimuth=7.5, name="r2",
        clips=4, frames=31, clutter=False, resolution="480", steps=30, looks=2,
        prompt="configs/cosmos/prompts/machine_shop.json", config_overrides={"dataset.fps": "15"},
    )  # fmt: skip
    win.apply_settings(s)
    assert win.settings() == s  # every field reaches a widget and comes back unchanged
    assert win.step_btns[4].isEnabled() and not win.sections["Scene"].isVisibleTo(win)
    win.apply_settings(dataclasses.replace(s, source="cad"))
    assert not win.step_btns[4].isEnabled() and win.sections["Scene"].isVisibleTo(win)
    win.close()
