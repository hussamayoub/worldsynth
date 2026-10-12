import json

import cv2
import numpy as np
import pytest

from worldsynth import cosmos


def _scene(shift: int = 0):
    seg = np.zeros((120, 160), np.uint16)
    seg[30:70, 40:90] = 1
    seg[80:110, 100:150] = 2
    img = np.full((120, 160, 3), 90, np.uint8)
    img[np.roll(seg, shift, axis=1) == 1] = (200, 60, 60)
    img[np.roll(seg, shift, axis=1) == 2] = (40, 160, 220)
    return img, seg


def test_alignment_score_detects_moved_geometry():
    img, seg = _scene()
    moved, _ = _scene(shift=25)
    assert cosmos.alignment_score(img, seg) > 0.9
    assert cosmos.alignment_score(moved, seg) < 0.5


def test_depth_control_near_is_bright_and_consistent():
    near = np.full((4, 4), 500, np.uint16)
    far = np.full((4, 4), 2000, np.uint16)
    far[0, 0] = 0  # invalid
    a, b = cosmos.depth_to_control([near, far])
    assert a.shape == (4, 4, 3) and a[1, 1, 0] > b[1, 1, 0]
    assert b[0, 0, 0] == 0


def test_prepare_writes_specs(tmp_path):
    ds = tmp_path / "ds"
    clip = ds / "clips" / "clip_0000"
    for d in ("rgb", "depth", "segmentation"):
        (clip / d).mkdir(parents=True)
    img, seg = _scene()
    for i in range(3):
        cv2.imwrite(str(clip / "rgb" / f"{i:06d}.png"), img)
        cv2.imwrite(str(clip / "depth" / f"{i:06d}.png"), np.full((120, 160), 800, np.uint16))
        cv2.imwrite(str(clip / "segmentation" / f"{i:06d}.png"), seg)
    (clip / "clip.json").write_text(
        json.dumps({"clutter": [{"shape": "cube"}, {"shape": "cube"}, {"shape": "sphere"}]})
    )
    (ds / "dataset.json").write_text(
        json.dumps(
            {
                "config": {"dataset": {"fps": 20, "resolution": [1248, 720]}},
                "categories": [{"id": 1, "name": "flange_plate"}],
            }
        )
    )
    prompt = tmp_path / "look.json"
    prompt.write_text(json.dumps({"subjects": [{"description": "Parts ({parts}) on a bench"}]}))
    manifest = cosmos.prepare(ds, tmp_path / "batch", [prompt], None, remote_dir="/remote/batch")
    (job,) = manifest["jobs"]
    spec = json.loads((tmp_path / "batch" / "specs" / f"{job['name']}.json").read_text())
    assert spec["num_frames"] == 3 and spec["fps"] == 20
    assert spec["depth"]["control_path"] == "/remote/batch/inputs/clip_0000/depth.mp4"
    assert spec["vision_path"] == "/remote/batch/inputs/clip_0000/vision.mp4"
    written = json.loads((tmp_path / "batch" / "prompts" / f"{job['name']}.json").read_text())
    assert spec["prompt_path"] == f"/remote/batch/prompts/{job['name']}.json"
    assert written["subjects"][0]["description"] == "Parts (flange plate) on a bench"
    assert written["subjects"][1]["description"].endswith("2 small cardboard boxes, 1 rubber ball")
    assert len(cosmos.read_video(tmp_path / "batch" / "inputs" / "clip_0000" / "seg.mp4")) == 3


def test_clutter_subject_only_when_clutter_exists():
    prompt = {"subjects": [{"description": "parts"}]}
    assert cosmos.add_clutter_subject(prompt, []) == prompt
    assert len(cosmos.add_clutter_subject(prompt, [{"shape": "cone"}])["subjects"]) == 2
    assert len(prompt["subjects"]) == 1  # the template is not modified


def test_aspect_mismatch_is_refused():
    cosmos.check_aspect((1248, 720), (832, 480))
    cosmos.check_aspect((1200, 720), (320, 192))
    with pytest.raises(ValueError, match="1248x720"):
        cosmos.check_aspect((1280, 720), (832, 480))
