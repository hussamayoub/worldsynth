"""Dataset generation loop. Needs a running Isaac Sim app with the Replicator orchestrator enabled
(``scripts/generate.py`` takes care of that)."""

from __future__ import annotations

import json
import logging
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import cad
from . import randomize as R
from .config import DatasetCfg, WorldSynthCfg
from .labels import build_coco
from .scene import CAMERA_PATH, HORIZONTAL_APERTURE, WorkcellScene, clutter_half_height
from .writer import FrameWriter, SceneObject, simplify_for_silhouette

logger = logging.getLogger("worldsynth")


@dataclass
class PartEntry:
    part: cad.Part
    class_id: int
    rotations: list[np.ndarray]
    probs: list[float]
    silhouette: tuple[np.ndarray, np.ndarray]
    """Simplified (vertices, faces) for the visibility computation."""


def load_parts(cfg: WorldSynthCfg) -> tuple[list[PartEntry], list[dict]]:
    """Discover, convert and load all CAD parts; return them with the category table."""
    files = cad.discover(cfg.cad.dir, cfg.cad.class_from)
    usd_files = cad.convert_all(files, cfg.cad.cache_dir, cfg.cad.tessellation_lod)
    loaded = []
    for f in files:
        if f.path not in usd_files:
            continue
        try:
            loaded.append(cad.load_part(f, usd_files[f.path], cfg.cad.units))
        except Exception as e:
            logger.error("Skipping %s: %s", f.path, e)
    if not loaded:
        raise RuntimeError("No CAD part could be loaded.")
    names = sorted({p.category for p in loaded})
    categories = [{"id": i + 1, "name": n} for i, n in enumerate(names)]
    class_ids = {c["name"]: c["id"] for c in categories}
    entries = []
    for p in loaded:
        mode = cfg.scene.pose_mode
        if mode == "stable":
            rots, probs = R.stable_rotations(p.vertices, p.faces)
        elif mode == "flat":
            rots = R.flat_rotations()
            probs = [1.0 / len(rots)] * len(rots)
        elif mode == "upright":
            rots, probs = [np.eye(3)], [1.0]
        else:
            raise ValueError(f"scene.pose_mode must be stable, flat or upright, got {mode!r}")
        sil = simplify_for_silhouette(p.vertices, p.faces)
        entries.append(PartEntry(part=p, class_id=class_ids[p.category], rotations=rots, probs=probs, silhouette=sil))
        logger.info(
            "Part %-24s class=%-16s extent=%s m, %d verts, %d resting poses",
            p.name, p.category, np.round(p.extent, 4).tolist(), len(p.vertices), len(rots),
        )  # fmt: skip
    return entries, categories


class Capture:
    """Render product on the scene camera with the annotators the dataset needs."""

    def __init__(self, ds: DatasetCfg):
        import omni.replicator.core as rep

        self.rep = rep
        if ds.renderer == "pathtracing":
            rep.settings.set_render_pathtraced(samples_per_pixel=ds.path_tracing_spp)
        elif ds.renderer == "realtime":
            rep.settings.set_render_rtx_realtime(antialiasing=ds.antialiasing)
        else:
            raise ValueError(f"dataset.renderer must be realtime or pathtracing, got {ds.renderer!r}")
        self.rp = rep.create.render_product(CAMERA_PATH, tuple(ds.resolution))
        names = {"rgb": "rgb", "instance": "instance_segmentation"}
        if "depth" in ds.annotations:
            names["depth"] = "distance_to_image_plane"
        if "normals" in ds.annotations:
            names["normals"] = "normals"
        self.annotators = {}
        for key, anno in names.items():
            kwargs = {"init_params": {"colorize": False}} if anno == "instance_segmentation" else {}
            a = rep.AnnotatorRegistry.get_annotator(anno, **kwargs)
            a.attach(self.rp)
            self.annotators[key] = a

    def step(self, rt_subframes: int):
        self.rep.orchestrator.step(rt_subframes=rt_subframes)

    def get(self) -> dict:
        out = {"rgb": np.asarray(self.annotators["rgb"].get_data())[..., :3]}
        inst = self.annotators["instance"].get_data()
        out["instance"] = np.asarray(inst["data"]).astype(np.int64)
        out["instance_paths"] = {
            int(k): v for k, v in inst["info"]["idToLabels"].items() if isinstance(v, str) and v.startswith("/")
        }
        out["depth"] = np.asarray(self.annotators["depth"].get_data()) if "depth" in self.annotators else None
        if "normals" in self.annotators:
            out["normals"] = np.asarray(self.annotators["normals"].get_data())[..., :3]
        else:
            out["normals"] = None
        return out


def sample_clip(
    rng: np.random.Generator, cfg: WorldSynthCfg, entries: list[PartEntry], scene: WorkcellScene, clip_idx: int
) -> dict:
    """Sample everything that defines one clip (pure data, applied to the stage by :func:`build_clip`)."""
    n = R.randint(rng, cfg.scene.parts_per_clip)
    per_class: dict[int, int] = {}
    chosen = []
    for idx in rng.permutation(np.repeat(np.arange(len(entries)), cfg.scene.max_instances_per_class)):
        e = entries[idx]
        if len(chosen) >= n or per_class.get(e.class_id, 0) >= cfg.scene.max_instances_per_class:
            continue
        per_class[e.class_id] = per_class.get(e.class_id, 0) + 1
        chosen.append(int(idx))
    footprints = []
    for idx in chosen:
        e = entries[idx]
        rot = e.rotations[rng.choice(len(e.rotations), p=e.probs)]
        footprints.append(R.footprint(e.part.vertices, rot))
    occupied: list[tuple[np.ndarray, float]] = []
    poses = R.place_on_table(
        rng, footprints, cfg.scene.placement_area, scene.table_z, cfg.scene.min_spacing, occupied=occupied
    )
    parts = [
        {"entry": idx, "pose": pose, "material": R.sample_part_material(rng, cfg.randomization.part_material)}
        for idx, pose in zip(chosen, poses, strict=True)
        if pose is not None
    ]

    # clutter goes around and between the parts, anywhere on the table top
    sx, sy, _ = cfg.scene.table.size
    table_area = ((-sx / 2 + 0.05, sx / 2 - 0.05), (-sy / 2 + 0.05, sy / 2 - 0.05))
    # own random stream: switching clutter off leaves the rest of the clip unchanged
    clutter_rng = np.random.default_rng([cfg.output.seed, clip_idx, 1])
    items = R.sample_clutter(clutter_rng, cfg.randomization.clutter) if cfg.randomization.clutter.enabled else []
    clutter_fps = []
    for it in items:
        s = np.asarray(it["scale"]) * (2.0 if it["shape"] == "sphere" else 1.0)
        box = R.footprint(np.array([-s / 2, s / 2]), np.eye(3))
        box.min_z = -clutter_half_height(it)
        clutter_fps.append(box)
    clutter_poses = R.place_on_table(clutter_rng, clutter_fps, table_area, scene.table_z, 0.005, occupied=occupied)
    clutter = [
        {**it, "pose": pose, "material": R.sample_surface_material(clutter_rng, (0.2, 0.9))}
        for it, pose in zip(items, clutter_poses, strict=True)
        if pose is not None
    ]
    return {
        "parts": parts,
        "clutter": clutter,
        "lights": R.sample_lights(rng, cfg.randomization.lighting, scene.placement_center()),
        "surfaces": {k: R.sample_surface_material(rng, cfg.randomization.surface_roughness) for k in scene.surfaces},
    }


def build_clip(scene: WorkcellScene, clip: dict, entries: list[PartEntry]) -> tuple[list[SceneObject], dict[str, int]]:
    """Apply a sampled clip to the stage.

    Returns:
        The labelled part instances and the segmentation ids of all other labelled prims (surfaces, clutter).
    """
    scene.clear_clip()
    scene.apply_lights(clip["lights"])
    scene.apply_surfaces(clip["surfaces"])
    objects = []
    for i, p in enumerate(clip["parts"]):
        e = entries[p["entry"]]
        path = scene.add_part(i, e.part, p["pose"], p["material"])
        objects.append(
            SceneObject(
                instance_id=i + 1,
                category=e.part.category,
                class_id=e.class_id,
                prim_path=path,
                part_name=e.part.name,
                vertices=e.silhouette[0],
                faces=e.silhouette[1],
                bbox_min=e.part.bbox_min,
                bbox_max=e.part.bbox_max,
                T_world_obj=p["pose"],
            )
        )
    background = scene.background_ids()
    for i, c in enumerate(clip["clutter"]):
        background[scene.add_clutter(i, c, c["pose"], c["material"])] = 2000 + i
    return objects, background


def _clip_record(clip: dict, entries: list[PartEntry], objects: list[SceneObject], camera: dict) -> dict:
    """JSON-friendly description of a clip's randomization, kept for ablations and failure analysis."""
    return {
        "parts": [
            {
                "instance_id": o.instance_id,
                "part": o.part_name,
                "class": o.category,
                "source": str(entries[p["entry"]].part.source),
                "T_world_obj": o.T_world_obj.tolist(),
                "material": p["material"],
            }
            for o, p in zip(objects, clip["parts"], strict=True)
        ],
        "clutter": [
            {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in c.items()} for c in clip["clutter"]
        ],
        "lights": clip["lights"],
        "surfaces": clip["surfaces"],
        "camera": camera,
    }


def run(cfg: WorldSynthCfg):
    import omni.usd

    out = Path(cfg.output.dir)
    if out.exists() and cfg.output.overwrite:
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    entries, categories = load_parts(cfg)
    meta = {
        "name": out.name,
        "generator": "worldsynth",
        "config": cfg.to_dict(),
        "categories": categories,
        "parts": [
            {
                "name": e.part.name,
                "class": e.part.category,
                "class_id": e.class_id,
                "source": str(e.part.source),
                "usd": str(e.part.usd_path),
                "scale_to_m": e.part.scale,
                "bbox_min": e.part.bbox_min.tolist(),
                "bbox_max": e.part.bbox_max.tolist(),
            }
            for e in entries
        ],
        "conventions": {
            "camera": "OpenCV (x right, y down, z forward); T_world_cam maps camera to world",
            "pose": "T_cam_obj maps the part frame (CAD file frame in meters) to the camera frame",
            "depth": "depth/*.png uint16, distance to the image plane in millimeters, 0 = invalid",
            "instance": "instance/*.png uint16, per-clip instance_id, 0 = background or clutter",
            "semantic": "semantic/*.png uint8, class_id from categories, 0 = background",
            "segmentation": "segmentation/*.png uint16, whole scene: parts = instance_id, surfaces 1000+, "
            "clutter 2000+",
            "normals": "normals/*.png RGB = (n + 1) / 2 * 255, camera frame",
            "bbox_2d": "[x_min, y_min, x_max, y_max] in pixels, max exclusive",
        },
        "coco_segmentation": cfg.dataset.coco_segmentation,
    }
    meta_path = out / "dataset.json"
    if meta_path.exists() and json.loads(meta_path.read_text())["categories"] != categories:
        raise RuntimeError(f"{out} holds a dataset with other categories; use a new output.dir or output.overwrite")
    meta_path.write_text(json.dumps(meta, indent=2))

    omni.usd.get_context().new_stage()
    stage = omni.usd.get_context().get_stage()
    scene = WorkcellScene(stage, cfg)
    width, height = cfg.dataset.resolution
    capture = Capture(cfg.dataset)
    writer = FrameWriter()
    ds = cfg.dataset
    t_start = time.time()
    generated = 0
    for clip_idx in range(ds.num_clips):
        clip_dir = out / "clips" / f"clip_{clip_idx:04d}"
        if (clip_dir / "clip.json").exists():
            logger.info("clip %d exists, skipping", clip_idx)
            continue
        if clip_dir.exists():
            shutil.rmtree(clip_dir)  # unfinished clip from an interrupted run
        rng = np.random.default_rng([cfg.output.seed, clip_idx])
        clip = sample_clip(rng, cfg, entries, scene, clip_idx)
        objects, background = build_clip(scene, clip, entries)
        traj = R.sample_camera_trajectory(rng, cfg.camera, scene.placement_center(), ds.frames_per_clip)
        k = R.intrinsics(width, height, traj.focal_length, HORIZONTAL_APERTURE)

        scene.set_camera(traj.transforms[0], traj.focal_length)
        for _ in range(ds.warmup_frames):
            capture.step(ds.rt_subframes)
        for f, t_cam in enumerate(traj.transforms):
            scene.set_camera(t_cam, traj.focal_length)
            capture.step(ds.rt_subframes)
            data = capture.get()
            writer.write(
                clip_dir,
                f,
                data["rgb"],
                data["depth"],
                data["instance"],
                data["instance_paths"],
                objects,
                t_cam,
                k,
                normals_world=data["normals"],
                background_ids=background if "segmentation" in ds.annotations else None,
                min_visible_fraction=ds.min_visible_fraction,
            )
        writer.flush()
        record = _clip_record(clip, entries, objects, {**traj.params, "K": k.tolist(), "fps": ds.fps})
        (clip_dir / "clip.json").write_text(json.dumps(record, indent=1))  # written last: marks the clip complete
        generated += 1
        rate = (time.time() - t_start) / generated
        logger.info(
            "clip %d/%d: %d parts, %d clutter, %.1f s/clip",
            clip_idx + 1,
            ds.num_clips,
            len(objects),
            len(clip["clutter"]),
            rate,
        )
    writer.close()
    coco = build_coco(out, "rgb")
    logger.info(
        "Done: %d images, %d annotations -> %s", len(coco["images"]), len(coco["annotations"]), out / "annotations"
    )
