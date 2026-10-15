"""Configuration for the WorldSynth generation pipeline.

All settings live in one YAML file (see ``configs/default.yaml``). Ranges are ``[min, max]`` pairs that are sampled
uniformly per clip. Lengths are in meters, angles in degrees unless the field name says otherwise.
"""

from __future__ import annotations

import dataclasses
import types
import typing
from dataclasses import dataclass, field
from pathlib import Path

import yaml

Range = tuple[float, float]
IntRange = tuple[int, int]


@dataclass
class CadCfg:
    dir: str | None = "cad"
    """Folder that is searched (recursively) for CAD and mesh files. Optional when ``scene.scene_parts`` provides
    the parts."""
    cache_dir: str = "cache/usd"
    """Where converted USD files are stored. Conversions are reused while the source file is unchanged."""
    units: str = "auto"
    """Source length unit: ``auto`` (trust the converted stage metadata), ``m``, ``cm``, ``mm`` or ``inch``."""
    class_from: str = "file"
    """Category name per part: ``file`` (file stem) or ``folder`` (parent folder name, for several files per class)."""
    tessellation_lod: int = 2
    """HOOPS tessellation level for B-rep CAD (STEP, IGES, ...), 0 (coarse) to 4 (fine)."""


@dataclass
class OutputCfg:
    dir: str = "output/run_000"
    seed: int = 0
    overwrite: bool = False


@dataclass
class DatasetCfg:
    num_clips: int = 20
    """Number of randomized scenes. Each clip keeps the scene fixed and moves the camera, which gives Cosmos a
    temporally consistent video to transfer and keeps every output frame aligned with its labels."""
    frames_per_clip: int = 61
    fps: int = 20
    resolution: tuple[int, int] = (1248, 720)
    """Image width and height [px]. Must have the aspect ratio of the Cosmos output, because Cosmos crops to it:
    1248x720 matches 480p (832x480), 1200x720 matches 256 16:9 (320x192)."""
    renderer: str = "realtime"
    """``realtime`` (RTX real-time) or ``pathtracing`` (slower; better reflections on metal parts)."""
    antialiasing: str = "DLAA"
    """Real-time anti-aliasing: DLAA (native resolution), FXAA, TAA, DLSS (renders lower and upscales) or Off."""
    path_tracing_spp: int = 64
    """Samples per pixel when ``renderer`` is ``pathtracing``."""
    rt_subframes: int = 8
    """Render subframes per captured frame; more removes RTX denoiser ghosting after scene changes."""
    warmup_frames: int = 4
    """Frames rendered and discarded after building a clip so materials and textures are loaded."""
    annotations: list[str] = field(default_factory=lambda: ["depth", "segmentation"])
    """Optional outputs besides rgb, instance/semantic masks and the box/pose labels (always written):
    ``depth``, ``segmentation`` (whole scene, needed for the Cosmos seg control) and ``normals``."""
    min_visible_fraction: float = 0.05
    """Objects whose visible fraction (1 - occlusion) is below this are flagged ``ignore`` in the labels."""
    coco_segmentation: str = "rle"
    """COCO mask encoding: ``rle`` (exact, compressed RLE) or ``polygon``."""


@dataclass
class TableCfg:
    size: tuple[float, float, float] = (1.2, 0.8, 0.75)
    """Table top size x, y and height of the top surface [m]."""


@dataclass
class SceneCfg:
    scene_usd: str | None = None
    """Predesigned USD scene used instead of the built-in workcell (table, floor, walls). Its units and up axis are
    converted to meters and z-up. Parts are placed on ``scene_surface``."""
    scene_surface: str | None = None
    """Prim path (as in the scene file) of the surface the parts are placed on; the top of its bounding box, inset by
    ``surface_margin``, becomes the placement area. Required with ``scene_usd``."""
    surface_margin: float = 0.03
    """Distance kept between parts and the edge of ``scene_surface`` [m]."""
    scene_camera: str | None = None
    """Camera prim path (as in the scene file) to render from. Its pose, focal length and aperture are used; the
    ``camera.*_sweep`` ranges move it around the point it looks at. Without it, cameras are sampled as usual."""
    scene_parts: str | None = None
    """Prim path (as in the scene file) whose children are labelled parts of the scene. Each child is one part; its
    class is its semantic ``class`` label or its name without a trailing number (``Bolt_01`` -> ``Bolt``)."""
    scene_parts_mode: str = "place"
    """``place``: each clip shows a random subset of the scene parts at random stable poses on ``scene_surface``
    (the others are hidden), like CAD parts. ``keep``: all scene parts stay where (and how) the scene moves them,
    e.g. on an animated conveyor; they still get labels and random materials."""
    placement_in_view: bool = True
    """With a scene camera, place parts (and clutter) only on the part of the surface the camera sees."""
    scene_lights_only: bool = False
    """True: light the scene with its own lights only. False: also add the randomized dome and area lights."""
    table: TableCfg = field(default_factory=TableCfg)
    walls: bool = True
    parts_per_clip: IntRange = (3, 8)
    max_instances_per_class: int = 3
    placement_area: tuple[Range, Range] = ((-0.3, 0.3), (-0.2, 0.2))
    """x and y range on the table top where parts are placed [m]."""
    pose_mode: str = "stable"
    """``stable`` (resting poses from the convex hull), ``flat`` (one of the 6 box faces down) or ``upright``
    (CAD z axis up)."""
    min_spacing: float = 0.01
    """Extra clearance between part footprints [m]."""


@dataclass
class LightingCfg:
    dome_intensity: Range = (200.0, 1200.0)
    dome_color_temperature: Range = (4000.0, 7500.0)
    num_lights: IntRange = (1, 3)
    light_intensity: Range = (3.0e3, 2.5e4)
    light_color_temperature: Range = (3000.0, 6500.0)
    light_height: Range = (1.6, 2.6)
    """Height of the ceiling lights above the floor [m]."""
    light_size: Range = (0.2, 1.2)


@dataclass
class PartMaterialCfg:
    randomize_prob: float = 0.8
    """Probability that a part gets a sampled material instead of the material from the CAD file."""
    metal_prob: float = 0.6
    """Of the randomized parts, fraction rendered as bare metal (the rest as painted/anodized/plastic)."""
    metal_roughness: Range = (0.15, 0.55)
    paint_roughness: Range = (0.3, 0.9)


@dataclass
class ClutterCfg:
    enabled: bool = True
    """False renders the same scenes without clutter (clutter has its own random stream, so the seed still gives
    identical parts, lights and camera paths)."""
    count: IntRange = (0, 8)
    size: Range = (0.02, 0.12)
    """Edge length of distractor primitives [m]."""
    margin: float = 0.1
    """Clutter is placed in ``scene.placement_area`` grown by this margin [m] (kept on the table), so it stays near
    the parts and in the camera's view."""


@dataclass
class RandomizationCfg:
    lighting: LightingCfg = field(default_factory=LightingCfg)
    part_material: PartMaterialCfg = field(default_factory=PartMaterialCfg)
    clutter: ClutterCfg = field(default_factory=ClutterCfg)
    surface_roughness: Range = (0.3, 0.95)
    """Roughness of the table, floor and walls; their colors are sampled from industrial palettes."""


@dataclass
class CameraCfg:
    distance: Range = (0.45, 0.9)
    """Distance from the look-at target [m]."""
    elevation: Range = (35.0, 75.0)
    azimuth: Range = (-70.0, 70.0)
    """Start azimuth around the table, 0 = in front of the table looking along +x."""
    azimuth_sweep: Range = (-35.0, 35.0)
    """Azimuth change over one clip; 0 gives a static camera."""
    elevation_sweep: Range = (-10.0, 10.0)
    distance_sweep: Range = (-0.15, 0.15)
    focal_length: Range = (18.0, 35.0)
    """Focal length [mm] with a 20.955 mm horizontal aperture."""
    target_jitter: float = 0.08
    """Random offset of the look-at target around the placement area center [m]."""


@dataclass
class WorldSynthCfg:
    cad: CadCfg = field(default_factory=CadCfg)
    output: OutputCfg = field(default_factory=OutputCfg)
    dataset: DatasetCfg = field(default_factory=DatasetCfg)
    scene: SceneCfg = field(default_factory=SceneCfg)
    randomization: RandomizationCfg = field(default_factory=RandomizationCfg)
    camera: CameraCfg = field(default_factory=CameraCfg)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def _build(cls, data):
    """Build dataclass ``cls`` from a (possibly partial) nested dict, rejecting unknown keys."""
    if data is None:
        return cls()
    hints = typing.get_type_hints(cls)
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - names
    if unknown:
        raise KeyError(f"Unknown key(s) {sorted(unknown)} for {cls.__name__}; valid keys: {sorted(names)}")
    kwargs = {}
    for key, value in data.items():
        hint = hints[key]
        if dataclasses.is_dataclass(hint):
            kwargs[key] = _build(hint, value)
        elif _is_tuple(hint) and value is not None:
            kwargs[key] = _to_tuple(value)
        else:
            kwargs[key] = value
    return cls(**kwargs)


def _is_tuple(hint) -> bool:
    if typing.get_origin(hint) is tuple:
        return True
    if isinstance(hint, types.UnionType):
        return any(typing.get_origin(a) is tuple for a in typing.get_args(hint))
    return False


def _to_tuple(value):
    if isinstance(value, list | tuple):
        return tuple(_to_tuple(v) for v in value)
    return value


def load_config(path: str | Path | None, overrides: list[str] | None = None) -> WorldSynthCfg:
    """Load a YAML config and apply ``section.key=value`` overrides (values are parsed as YAML).

    Relative paths (``cad.dir``, ``cad.cache_dir``, ``output.dir``, ``scene.scene_usd``) are resolved against
    the current working directory.
    """
    data: dict = {}
    base = Path.cwd()
    if path is not None:
        data = yaml.safe_load(Path(path).read_text()) or {}
    for item in overrides or []:
        key, _, value = item.partition("=")
        node = data
        *parents, leaf = key.strip().split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = yaml.safe_load(value)
    cfg = _build(WorldSynthCfg, data)
    cfg.cad.dir = _resolve(base, cfg.cad.dir) if cfg.cad.dir else None
    cfg.cad.cache_dir = _resolve(base, cfg.cad.cache_dir)
    cfg.output.dir = _resolve(base, cfg.output.dir)
    if cfg.scene.scene_usd:
        cfg.scene.scene_usd = _resolve(base, cfg.scene.scene_usd)
    return cfg


def _resolve(base: Path, p: str) -> str:
    path = Path(p).expanduser()
    return str(path if path.is_absolute() else (base / path).resolve())
