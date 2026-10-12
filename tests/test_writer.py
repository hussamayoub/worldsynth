import json

import cv2
import numpy as np
import pytest

from worldsynth import labels as L
from worldsynth.randomize import USD_TO_CV, look_at
from worldsynth.writer import FrameWriter, SceneObject, rasterize_mesh, simplify_for_silhouette


def _box(size):
    import trimesh

    m = trimesh.creation.box(size)
    return np.asarray(m.vertices), np.asarray(m.faces)


K = np.array([[400.0, 0, 160], [0, 400.0, 120], [0, 0, 1]])


def test_rasterize_box_front_view_matches_projected_square():
    v, f = _box((0.1, 0.1, 0.1))
    t = np.eye(4)
    t[2, 3] = 1.0  # box centered 1 m in front of the camera
    total, inside, box = rasterize_mesh(K, t, v, f, 320, 240)
    # front face at z = 0.95 projects to 0.1 * 400 / 0.95 px wide
    side = 0.1 * 400 / 0.95
    assert total == inside
    assert total == pytest.approx(side**2, rel=0.08)
    assert box[0] == pytest.approx(160 - side / 2, abs=1.5)


def test_rasterize_counts_truncated_part():
    v, f = _box((0.1, 0.1, 0.1))
    t = np.eye(4)
    t[:3, 3] = [-0.4, 0.0, 1.0]  # center projects to x = 0, half of it is left of the image
    total, inside, _ = rasterize_mesh(K, t, v, f, 320, 240)
    assert inside / total == pytest.approx(0.5, abs=0.03)


def test_simplify_keeps_silhouette():
    import trimesh

    m = trimesh.creation.icosphere(subdivisions=5, radius=0.05)
    v, f = simplify_for_silhouette(np.asarray(m.vertices), np.asarray(m.faces), cells=32)
    assert len(f) < len(m.faces) / 2
    t = np.eye(4)
    t[2, 3] = 0.5
    full = rasterize_mesh(K, t, np.asarray(m.vertices), np.asarray(m.faces), 320, 240)[0]
    assert rasterize_mesh(K, t, v, f, 320, 240)[0] == pytest.approx(full, rel=0.05)


def test_frame_writer_outputs(tmp_path):
    h, w = 240, 320
    v, f = _box((0.1, 0.1, 0.1))
    t_world_obj = np.eye(4)
    t_world_obj[:3, 3] = [0.0, 0.0, 0.05]
    cam_usd = look_at([0.0, 0.0, 1.05], [0.0, 0.0, 0.0], up=(0.0, 1.0, 0.0))
    t_cam_obj = np.linalg.inv(cam_usd @ USD_TO_CV) @ t_world_obj
    # synthetic renderer output: the object's silhouette as renderer id 7, a surface as id 3
    _, _, box = rasterize_mesh(K, t_cam_obj, v, f, w, h)
    raw = np.full((h, w), 3, dtype=np.int64)
    raw[box[1] : box[3], box[0] : box[2]] = 7
    obj = SceneObject(1, "cube", 1, "/World/Parts/part_000", "cube", v, f, v.min(0), v.max(0), t_world_obj)
    writer = FrameWriter(num_workers=2)
    label = writer.write(
        tmp_path,
        0,
        np.zeros((h, w, 3), np.uint8),
        np.full((h, w), 1.0, np.float32),
        raw,
        {7: "/World/Parts/part_000/geom/asset/mesh", 3: "/World/Workcell/Table"},
        [obj],
        cam_usd,
        K,
        background_ids={"/World/Workcell/Table": 1001},
    )
    writer.close()
    (o,) = label["objects"]
    assert o["bbox_2d"] == box
    assert o["visible_fraction"] == pytest.approx(1.0, abs=0.02)
    np.testing.assert_allclose(o["pose"]["T_cam_obj"], t_cam_obj, atol=1e-9)
    # z of the object center in the camera frame = camera height - object center height
    assert o["bbox_3d"]["center_cam"][2] == pytest.approx(1.0)
    inst = cv2.imread(str(tmp_path / "instance" / "000000.png"), cv2.IMREAD_UNCHANGED)
    seg = cv2.imread(str(tmp_path / "segmentation" / "000000.png"), cv2.IMREAD_UNCHANGED)
    assert set(np.unique(inst)) == {0, 1}
    assert set(np.unique(seg)) == {1, 1001}
    assert (L.rle_decode(o["mask_rle"]) == (inst == 1)).all()
    assert json.loads((tmp_path / "labels" / "000000.json").read_text())["objects"][0]["instance_id"] == 1
    assert cv2.imread(str(tmp_path / "depth" / "000000.png"), cv2.IMREAD_UNCHANGED)[0, 0] == 1000
