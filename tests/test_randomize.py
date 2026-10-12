import numpy as np
import pytest

from worldsynth import randomize as R
from worldsynth.config import CameraCfg, load_config


def test_look_at_points_usd_camera_at_target():
    eye, target = np.array([-1.0, 0.5, 1.5]), np.array([0.1, -0.2, 0.75])
    t = R.look_at(eye, target)
    forward = -t[:3, 2]  # USD cameras look along local -z
    np.testing.assert_allclose(forward, (target - eye) / np.linalg.norm(target - eye), atol=1e-12)
    assert t[2, 1] > 0  # image up points up in the world
    np.testing.assert_allclose(t[:3, :3].T @ t[:3, :3], np.eye(3), atol=1e-12)


def test_quaternion_roundtrip():
    rng = np.random.default_rng(1)
    for _ in range(20):
        r = R.random_rotation(rng)
        q = R.matrix_to_quat(r)
        assert q[0] >= 0
        np.testing.assert_allclose(R.quat_to_matrix(q), r, atol=1e-9)


def test_flat_rotations_put_each_axis_down():
    downs = {tuple(np.round(r.T @ [0, 0, -1]).astype(int)) for r in R.flat_rotations()}
    assert downs == {(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)}


def _box(size):
    import trimesh

    m = trimesh.creation.box(size)
    return np.asarray(m.vertices), np.asarray(m.faces)


def test_stable_rotations_of_flat_plate_prefer_large_faces():
    v, f = _box((0.1, 0.06, 0.01))
    rots, probs = R.stable_rotations(v, f)
    assert abs(sum(probs) - 1) < 1e-9
    best = rots[int(np.argmax(probs))]
    # the most likely resting pose has the thin axis vertical
    assert abs(abs((best @ [0, 0, 1])[2]) - 1) < 1e-6


def test_place_on_table_no_overlap_and_on_surface():
    rng = np.random.default_rng(0)
    v, _ = _box((0.05, 0.03, 0.02))
    fps = [R.footprint(v, R.flat_rotations()[i % 6]) for i in range(10)]
    occupied = []
    poses = R.place_on_table(rng, fps, ((-0.3, 0.3), (-0.2, 0.2)), 0.75, 0.01, occupied=occupied)
    placed = [(p, fp) for p, fp in zip(poses, fps, strict=True) if p is not None]
    assert len(placed) == len(occupied) >= 8
    for p, _ in placed:
        world = v @ p[:3, :3].T + p[:3, 3]
        assert world[:, 2].min() == pytest.approx(0.75)
    for i in range(len(occupied)):
        for j in range(i):
            (ci, ri), (cj, rj) = occupied[i], occupied[j]
            assert np.linalg.norm(ci - cj) >= ri + rj + 0.01 - 1e-9


def test_camera_trajectory_keeps_distance_and_target():
    rng = np.random.default_rng(3)
    cfg = CameraCfg(distance=(0.5, 0.5), distance_sweep=(0, 0), target_jitter=0.0)
    traj = R.sample_camera_trajectory(rng, cfg, [0, 0, 0.75], 5)
    for t in traj.transforms:
        assert np.linalg.norm(t[:3, 3] - [0, 0, 0.75]) == pytest.approx(0.5)


def test_load_config_overrides(tmp_path):
    cfg_file = tmp_path / "c.yaml"
    cfg_file.write_text("dataset:\n  num_clips: 3\nscene:\n  placement_area: [[-0.1, 0.1], [-0.2, 0.2]]\n")
    cfg = load_config(cfg_file, ["dataset.resolution=[640,360]", "camera.focal_length=[20,30]"])
    assert cfg.dataset.num_clips == 3
    assert cfg.dataset.resolution == (640, 360)
    assert cfg.scene.placement_area == ((-0.1, 0.1), (-0.2, 0.2))
    assert cfg.camera.focal_length == (20, 30)
    with pytest.raises(KeyError):
        load_config(None, ["dataset.num_clip=3"])


def test_default_yaml_matches_code_defaults():
    from pathlib import Path

    yaml_cfg = load_config(Path(__file__).parents[1] / "configs" / "default.yaml")
    assert yaml_cfg == load_config(None)
