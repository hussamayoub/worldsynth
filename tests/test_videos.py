import json

import cv2
import numpy as np

from worldsynth import videos as vid
from worldsynth.labels import rle_encode


def _dataset(root):
    clip = root / "data" / "clips" / "clip_0000"
    for d in ("rgb", "rgb_cosmos", "depth", "segmentation", "instance", "labels"):
        (clip / d).mkdir(parents=True)
    (root / "data" / "dataset.json").write_text(json.dumps({"config": {"dataset": {"fps": 10}}}))
    mask = np.zeros((40, 60), bool)
    mask[10:20, 10:30] = True
    for i in range(3):
        cv2.imwrite(str(clip / "rgb" / f"{i:06d}.png"), np.full((40, 60, 3), 100, np.uint8))
        cv2.imwrite(str(clip / "rgb_cosmos" / f"{i:06d}.png"), np.full((40, 60, 3), 50, np.uint8))
        cv2.imwrite(str(clip / "depth" / f"{i:06d}.png"), np.full((40, 60), 800 + 100 * i, np.uint16))
        cv2.imwrite(str(clip / "segmentation" / f"{i:06d}.png"), mask.astype(np.uint16) * 1001)
        cv2.imwrite(str(clip / "instance" / f"{i:06d}.png"), mask.astype(np.uint16))
        obj = {"instance_id": 1, "class": "bolt", "mask_rle": rle_encode(mask), "bbox_2d": [10, 10, 30, 20],
               "ignore": False, "visible_fraction": 1.0,
               "bbox_3d": {"corners_2d": [[10, 10]] * 8, "min_obj": [0, 0, 0], "max_obj": [0.1, 0.1, 0.1]},
               "pose": {"T_cam_obj": np.eye(4).tolist()}}  # fmt: skip
        label = {"camera": {"K": [[50, 0, 30], [0, 50, 20], [0, 0, 1]]}, "objects": [obj]}
        (clip / "labels" / f"{i:06d}.json").write_text(json.dumps(label))
    inputs = root / "cos" / "inputs" / "clip_0000"
    inputs.mkdir(parents=True)
    import imageio.v2 as imageio

    imageio.mimwrite(inputs / "vision.mp4", [np.full((32, 48, 3), 200, np.uint8)] * 4, fps=10, macro_block_size=1)
    return root / "data", root / "cos"


def test_find_sources_and_read_frames(tmp_path):
    data, cos = _dataset(tmp_path)
    sources = {s.key: s for s in vid.find_sources(data, cos)}
    assert set(sources) == {"Isaac Sim/clip_0000", "Cosmos (imported)/clip_0000 rgb_cosmos",
                            "Cosmos inputs/clip_0000 vision"}  # fmt: skip
    render = sources["Isaac Sim/clip_0000"]
    assert render.fps == 10 and vid.frame_count(render) == 3
    rgb = vid.read_frame(render, 0, "rgb")
    labelled = vid.read_frame(render, 0, "rgb + labels")
    assert rgb.shape == (40, 60, 3) and not np.array_equal(rgb, labelled)  # labels were drawn
    assert vid.read_frame(render, 0, "depth").shape == (40, 60, 3)
    seg = vid.read_frame(render, 1, "segmentation")
    assert seg[15, 15].any() and not seg[0, 0].any()  # segment colored, background black
    assert vid.read_frame(render, 5, "rgb") is None
    movie = sources["Cosmos inputs/clip_0000 vision"]
    assert vid.frame_count(movie) == 4 and vid.read_frame(movie, 3, "video").shape == (32, 48, 3)
    assert vid.find_sources(None, None) == []
