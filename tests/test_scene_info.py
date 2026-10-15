import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pxr")

from worldsynth import scene_info as si  # noqa: E402

ROOT = Path(__file__).parents[1]


@pytest.fixture(scope="module")
def example_scene(tmp_path_factory):
    path = tmp_path_factory.mktemp("scene") / "cell.usda"
    subprocess.run([sys.executable, str(ROOT / "scripts" / "make_example_scene.py"), str(path)], check=True)
    return path


def test_inspect_converts_y_up_centimeters(example_scene):
    info = si.inspect(example_scene)
    assert info.root_prim == "/Cell" and info.up_axis == "Y" and info.meters_per_unit == pytest.approx(0.01)
    cams = {c.path: c for c in info.cameras}
    assert set(cams) == {"/Cell/Cameras/BenchCam", "/Cell/Cameras/Overview"}
    # scene (0, 150, 55) cm, y up -> (0, -0.55, 1.5) m, z up
    np.testing.assert_allclose(cams["/Cell/Cameras/BenchCam"].position, [0.0, -0.55, 1.5], atol=1e-6)
    surfaces = {s.path: s for s in info.surfaces}
    top = surfaces["/Cell/Workbench/Top"]
    assert top.top_z == pytest.approx(0.925) and top.size == pytest.approx([1.4, 0.8])
    assert "/Cell/Workbench" in surfaces  # typeless group prims are offered too
    assert info.surfaces[0].path == "/Cell/Floor"  # largest first


def test_stage_path_and_correction():
    assert si.stage_path("/Cell/Workbench/Top", "/Cell") == "/World/Scene/Workbench/Top"
    assert si.stage_path("/Cell", "/Cell") == "/World/Scene"
    with pytest.raises(ValueError):
        si.stage_path("/Other/Prim", "/Cell")
    t = si.correction(0.01, "Y")
    np.testing.assert_allclose(t[:3, :3] @ [0, 100, 0], [0, 0, 1.0], atol=1e-12)  # 1 m up
    rot = si.orthonormal(t)[:3, :3]
    np.testing.assert_allclose(rot.T @ rot, np.eye(3), atol=1e-12)


def test_scene_parts_groups_and_class_names(example_scene):
    groups = {g.path: g.children for g in si.inspect(example_scene).groups}
    assert groups["/Cell/Parts"] == ["Bolt_01", "Bolt_02", "Block_01", "Block_02"]
    assert "/Cell/Cameras" not in groups and "/Cell/Lights" not in groups  # cameras and lights are no parts
    names = ["Bolt_01", "bolt-2", "Gear.12", "Housing", "Plate 3", "007"]
    assert [si.part_class_name(n) for n in names] == ["Bolt", "bolt", "Gear", "Housing", "Plate", "007"]
