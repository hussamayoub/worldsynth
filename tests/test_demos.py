import json

import cv2
import h5py
import numpy as np
import pytest

from worldsynth import cosmos, demos

T, H, W = 6, 32, 32


def _hdf5(path):
    rng = np.random.default_rng(0)
    with h5py.File(path, "w") as f:
        data = f.create_group("data")
        data.attrs["env_args"] = '{"env_name": "Test"}'
        data.attrs["total"] = 2 * T
        for d, success in (("demo_0", True), ("demo_1", True), ("demo_2", False)):
            g = data.create_group(d)
            g.attrs["num_samples"] = T
            g.attrs["success"] = success
            g.create_dataset("actions", data=rng.normal(size=(T, 8)).astype(np.float32))
            g.create_dataset("states/articulation/robot/joint_position", data=np.ones((T, 13), np.float32))
            obs = g.create_group("obs")
            obs.create_dataset("eef_pos", data=np.zeros((T, 3), np.float32))
            for cam in ("scene_cam", "wrist_cam"):
                obs.create_dataset(cam, data=rng.integers(0, 255, (T, H, W, 3), dtype=np.uint8), compression="gzip")
                obs.create_dataset(f"{cam}_depth", data=np.full((T, H, W, 1), 700, np.int16))
                seg = np.zeros((T, H, W, 1), np.uint8)
                seg[:, 5:15, 5:20] = 10
                seg[:, 20:30, 2:30] = 4
                obs.create_dataset(f"{cam}_seg", data=seg)
    return path


def test_import_prepare_export_roundtrip(tmp_path):
    src = _hdf5(tmp_path / "demos.hdf5")
    names = demos.parse_seg_names("4-5=bin,10-15=soup_can")
    meta = demos.import_demos(src, tmp_path / "ds", ["scene_cam", "wrist_cam"], seg_names=names)
    assert meta["source"]["demos"] == ["demo_0", "demo_1"]  # the failed demo is skipped
    assert [c["name"] for c in meta["categories"]] == ["bin", "soup_can"]
    clip = tmp_path / "ds" / "clips" / "demo_0__wrist_cam"
    assert len(list((clip / "rgb").glob("*.png"))) == T
    assert cv2.imread(str(clip / "depth" / "000000.png"), cv2.IMREAD_UNCHANGED)[0, 0] == 700
    label = json.loads((clip / "labels" / "000003.json").read_text())
    can = next(o for o in label["objects"] if o["class"] == "soup_can")
    assert can["bbox_2d"] == [5, 5, 20, 15] and can["pixel_area"] == 150
    assert (tmp_path / "ds" / "annotations" / "coco_rgb.json").exists()

    # Cosmos batch: square images -> 1:1, both cameras of a demo get the same look
    looks = []
    for name in ("look_a", "look_b"):
        (tmp_path / f"{name}.json").write_text(json.dumps({"subjects": [{"description": "{parts}"}]}))
        looks.append(tmp_path / f"{name}.json")
    manifest = cosmos.prepare(tmp_path / "ds", tmp_path / "batch", looks, None, spec_overrides={
        "resolution": "256", "aspect_ratio": "auto"})  # fmt: skip
    by_clip = {j["clip"]: j for j in manifest["jobs"]}
    assert by_clip["demo_0__scene_cam"]["prompt"] == by_clip["demo_0__wrist_cam"]["prompt"]
    assert by_clip["demo_0__scene_cam"]["prompt"] != by_clip["demo_1__scene_cam"]["prompt"]
    spec = json.loads((tmp_path / "batch" / "specs" / f"{by_clip['demo_0__scene_cam']['name']}.json").read_text())
    assert spec["aspect_ratio"] == "1,1" and "depth" in spec and "seg" in spec

    # pretend Cosmos ran for demo_0 only: its frames are a constant gray
    for cam in ("scene_cam", "wrist_cam"):
        out = tmp_path / "ds" / "clips" / f"demo_0__{cam}" / "rgb_cosmos"
        out.mkdir()
        for t in range(T):
            cv2.imwrite(str(out / f"{t:06d}.png"), np.full((H, W, 3), 128, np.uint8))
    written = demos.export_demos(tmp_path / "ds", tmp_path / "cosmos.hdf5")
    assert written == ["demo_0"]  # demo_1 has no Cosmos frames
    # only one camera of demo_1 through Cosmos: left out by default, kept with keep_missing
    out1 = tmp_path / "ds" / "clips" / "demo_1__scene_cam" / "rgb_cosmos"
    out1.mkdir()
    for t in range(T):
        cv2.imwrite(str(out1 / f"{t:06d}.png"), np.full((H, W, 3), 64, np.uint8))
    assert demos.export_demos(tmp_path / "ds", tmp_path / "c2.hdf5") == ["demo_0"]
    assert demos.export_demos(tmp_path / "ds", tmp_path / "c3.hdf5", keep_missing=True) == ["demo_0", "demo_1"]
    with h5py.File(src) as a, h5py.File(tmp_path / "c3.hdf5") as b:
        assert (b["data/demo_1/obs/scene_cam"][()] == 64).all()
        assert np.array_equal(b["data/demo_1/obs/wrist_cam"][()], a["data/demo_1/obs/wrist_cam"][()])
    with h5py.File(src) as a, h5py.File(tmp_path / "cosmos.hdf5") as b:
        assert list(b["data"].keys()) == ["demo_0"] and b["data"].attrs["total"] == T
        assert b["data"].attrs["env_args"] == a["data"].attrs["env_args"]
        new, old = b["data/demo_0"], a["data/demo_0"]
        assert new["obs/scene_cam"].shape == old["obs/scene_cam"].shape and (new["obs/scene_cam"][()] == 128).all()
        assert np.array_equal(new["actions"][()], old["actions"][()])
        assert np.array_equal(new["obs/scene_cam_seg"][()], old["obs/scene_cam_seg"][()])
        assert np.array_equal(new["states/articulation/robot/joint_position"][()],
                              old["states/articulation/robot/joint_position"][()])  # fmt: skip


def test_parse_seg_names(tmp_path):
    assert demos.parse_seg_names("1=robot, 10-12=can") == {1: "robot", 10: "can", 11: "can", 12: "can"}
    (tmp_path / "n.json").write_text('{"3": "ground"}')
    assert demos.parse_seg_names(str(tmp_path / "n.json")) == {3: "ground"}
    assert demos.parse_seg_names(None) == {}


def test_import_without_cameras_fails(tmp_path):
    src = _hdf5(tmp_path / "demos.hdf5")
    with pytest.raises(RuntimeError, match="No demo"):
        demos.import_demos(src, tmp_path / "ds", ["front_cam"])


def test_convert_runs_converter_with_inputs_and_output(tmp_path):
    import subprocess
    import sys
    from pathlib import Path

    assert demos.export_name(tmp_path / "out.hdf5", "rgb_cosmos") == tmp_path / "out.hdf5"
    assert demos.export_name(tmp_path / "out.hdf5", "rgb_cosmos_machine_shop") == tmp_path / "out_machine_shop.hdf5"
    # stand-in converter: records its arguments as a LeRobot-like dataset
    conv = tmp_path / "conv.py"
    conv.write_text(
        "import json, sys, pathlib\n"
        "out = pathlib.Path(sys.argv[sys.argv.index('--output_dir') + 1])\n"
        "(out / 'meta').mkdir(parents=True)\n"
        "(out / 'meta' / 'info.json').write_text(json.dumps(sys.argv[1:]))\n"
    )
    script = Path(__file__).resolve().parents[1] / "scripts" / "demos.py"
    out = tmp_path / "lerobot"
    run = [sys.executable, str(script), "convert", "--hdf5", "a.hdf5", "b.hdf5", "--out", str(out),
           "--script", str(conv), "--args", "--fps 15 --input_files {inputs} --output_dir {output}"]  # fmt: skip
    assert subprocess.run(run, capture_output=True).returncode == 0
    args = json.loads((out / "meta" / "info.json").read_text())
    assert args == ["--fps", "15", "--input_files", "a.hdf5", "b.hdf5", "--output_dir", str(out)]
    assert subprocess.run(run, capture_output=True).returncode != 0  # output exists
    assert subprocess.run(run + ["--overwrite"], capture_output=True).returncode == 0
    with pytest.raises(ValueError, match="inputs"):
        demos.convert_command(conv, ["a.hdf5"], out, args="--output_dir {output}")
