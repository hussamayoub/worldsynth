import json

import numpy as np
import pytest

from worldsynth import labels as L


def test_rle_matches_pycocotools_reference():
    # reference string from pycocotools.mask.encode for this mask
    m = np.zeros((6, 7), dtype=bool)
    m[1:4, 2:6] = True
    m[5, 0] = True
    m[0, 6] = True
    assert L.rle_encode(m) == {"size": [6, 7], "counts": "5172L00000ON3"}


@pytest.mark.parametrize("seed", range(5))
def test_rle_roundtrip(seed):
    rng = np.random.default_rng(seed)
    m = rng.random((37, 53)) < 0.3
    m[0, 0] = bool(seed % 2)  # masks starting with a foreground pixel need a leading zero run
    assert np.array_equal(L.rle_decode(L.rle_encode(m)), m)


def test_bbox_from_mask_is_max_exclusive():
    m = np.zeros((10, 10), dtype=bool)
    m[2:5, 3:8] = True
    assert L.bbox_from_mask(m) == [3, 2, 8, 5]
    assert L.bbox_from_mask(np.zeros((4, 4), dtype=bool)) is None


def test_project_center_and_depth_encoding():
    k = np.array([[500.0, 0, 320], [0, 500.0, 240], [0, 0, 1]])
    uv = L.project(k, np.array([[0.0, 0.0, 2.0], [0.2, -0.1, 1.0]]))
    np.testing.assert_allclose(uv, [[320, 240], [420, 190]])
    d = L.encode_depth_mm(np.array([[0.5, np.inf], [np.nan, 70.0]]))
    np.testing.assert_array_equal(d, [[500, 0], [0, 0]])


def _write_dataset(root, ignore_second: bool):
    (root / "clips" / "clip_0000" / "labels").mkdir(parents=True)
    (root / "clips" / "clip_0000" / "rgb").mkdir()
    (root / "dataset.json").write_text(json.dumps({"categories": [{"id": 1, "name": "bracket"}]}))
    mask = np.zeros((4, 6), dtype=bool)
    mask[1:3, 2:5] = True
    obj = {
        "instance_id": 1,
        "class_id": 1,
        "bbox_2d": [2, 1, 5, 3],
        "pixel_area": 6,
        "mask_rle": L.rle_encode(mask),
        "ignore": False,
        "visible_fraction": 1.0,
    }
    objs = [obj, {**obj, "instance_id": 2, "ignore": ignore_second}]
    label = {"width": 6, "height": 4, "objects": objs}
    (root / "clips" / "clip_0000" / "labels" / "000000.json").write_text(json.dumps(label))
    (root / "clips" / "clip_0000" / "rgb" / "000000.png").write_bytes(b"")


def test_build_coco(tmp_path):
    _write_dataset(tmp_path, ignore_second=True)
    coco = L.build_coco(tmp_path)
    assert [i["file_name"] for i in coco["images"]] == ["clips/clip_0000/rgb/000000.png"]
    a, b = coco["annotations"]
    assert a["bbox"] == [2, 1, 3, 2] and a["area"] == 6 and a["iscrowd"] == 0
    assert b["iscrowd"] == 1  # ignored objects become crowd regions
    assert (tmp_path / "annotations" / "coco_rgb.json").exists()
    # frames without an image in a requested folder are skipped
    assert L.build_coco(tmp_path, ["rgb", "rgb_cosmos"])["images"][0]["source"] == "rgb"
