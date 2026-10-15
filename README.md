# WorldSynth

Synthetic training data for robotic machine tending, from CAD files to labelled, photorealistic images.
Part of the KIEM HighTech project *WorldSynth: World Foundation Models for Photorealistic Synthetic Data Generation in
Robotic Machine Tending* (Fontys, Affix Engineering, Datory).

```
CAD folder ──► Isaac Sim (stage 1) ──────────────────────► labelled dataset (rgb + labels)
 STEP, IGES,     convert to USD, randomize scenes,            │
 SolidWorks,     render clips with a moving camera            │ depth + segmentation + rgb
 STL, OBJ, ...                                                ▼
                                       Cosmos Transfer (stage 2, DGX Spark)
                                       edge/depth/seg-controlled video2video
                                                              │
                                                              ▼
                            same labels, photorealistic images (rgb_cosmos)
```

**Stage 1** renders *clips*: one randomized scene (parts, lighting, materials, clutter) seen by a camera that moves
smoothly over N frames. Every frame gets RGB, depth, instance/semantic/whole-scene segmentation, 2D boxes (tight and
amodal), 3D boxes, 6D poses, visibility, camera intrinsics and extrinsics, plus a COCO file.

**Stage 2** sends each clip as a video to Cosmos Transfer, conditioned on edges, depth and segmentation of the render
so the geometry stays put while the appearance becomes realistic. The generated frames are imported back next to the
rendered ones and reuse their labels. An alignment check rejects clips where Cosmos moved or invented objects.

Clips (rather than single images) are used because Cosmos Transfer works on videos, and a camera move gives
temporally consistent views of one scene.

## Install

WorldSynth runs in the Isaac Lab environment (Isaac Sim 6.1). Install it editable into Isaac Lab's venv:

```bash
cd ~/research/isaaclab/IsaacLab
uv pip install --python .venv/bin/python --no-deps -e ../worldsynth
```

Its dependencies (numpy, opencv, imageio-ffmpeg, pyyaml, trimesh, networkx, scipy) are already in that venv.
Only stage 1 needs Isaac Sim; the Cosmos tools, `visualize.py` and the tests run in any Python 3.12 environment
with those packages.

## GUI

```bash
cd ~/research/isaaclab/worldsynth
scripts/gui.sh              # add --preview to render a preview of the saved setup right after start
```

One window for the whole pipeline:

- **Parts:** the CAD/STL folder, units and how classes are named.
- **Scene:** the built-in workcell, or a predesigned USD scene. After opening a scene, pick a **camera** from it
  (or random cameras around the parts) and the **surface** to place the parts on. Set how far the camera moves
  over a clip (± azimuth and elevation), and whether to use only the scene's own lights (the default; added random
  lights tend to wash a lit scene out, and Cosmos then invents objects in the blank areas).
- **Dataset:** run name, clips, frames per clip, parts per clip, clutter, and extra `key=value` config overrides.
- **Cosmos:** resolution (256 or 480), steps, looks per clip, look and negative prompt, or "Render only".
- **Spark:** host/IP, user and password, with **Test connection** (checks the login, the Cosmos runner and free
  memory). The password is only passed to `ssh` in the environment of the pipeline process; it is never saved
  (leave it empty if you use SSH keys).
- **Preview frame** renders one labelled frame (Isaac Sim starts, ~30 s) to check camera, surface and part size.
  **Run pipeline** runs `scripts/run_pipeline.sh` with these settings and streams its log; at the end it shows a
  labelled Cosmos frame. **Stop** ends the local run (a Cosmos job already on the Spark finishes; run again with
  the same name to collect it). **Open output** opens `output/<name>`.

Settings are saved in `~/.config/worldsynth/gui.json` (set `WORLDSYNTH_GUI_SETTINGS` to use another file).

## Using your own USD scene

Any USD scene works, in any units and with y or z up (converted to meters and z up). WorldSynth references it
instead of the built-in table and walls, places the parts on the chosen surface (the top of its bounding box, minus
`scene.surface_margin`), adds the clutter around them, and renders from the chosen scene camera with its focal
length and aperture. The camera moves by `camera.azimuth_sweep` / `elevation_sweep` / `distance_sweep` around the
point it looks at; the first frame is exactly the scene camera. The scene's top-level objects are labelled so the
whole-scene segmentation (Cosmos seg control) covers them. On the command line:

```bash
scripts/run_pipeline.sh --cad cad_lego --units mm --name scene_001 --clips 20 \
  --scene assets/scenes/example_cell.usda --scene-surface /Cell/Workbench/Top \
  --scene-camera /Cell/Cameras/BenchCam --scene-lights-only \
  --set "camera.azimuth_sweep=[-10,10]" --set "camera.elevation_sweep=[-3,3]" --set "camera.distance_sweep=[0,0]"
```

`assets/scenes/example_cell.usda` (made by `scripts/make_example_scene.py`, centimeters, y up) has a workbench, a
machine and two cameras. Prim paths are the paths inside the scene file, as the GUI lists them. The surface is
used as an axis-aligned box, so a strongly rotated table gives a placement area that overhangs it.

## One command: `scripts/run_pipeline.sh`

`run_pipeline.sh` runs everything from a CAD folder to an imported Cosmos dataset: render the clips in Isaac Sim,
build the Cosmos batch, upload it to the DGX Spark, run Cosmos there, copy the outputs back, and import them with
label previews. Run it from the `worldsynth` folder:

```bash
cd ~/research/isaaclab/worldsynth
scripts/run_pipeline.sh --cad /path/to/stl_folder --name run_001 --clips 20
```

Example: the LEGO pieces in this repo, one clip at Cosmos 480 (about 30 s in Isaac Sim, about 36 min on the Spark).
The `--set` lines adapt the scene to small plastic parts: a closer camera, a smaller placement area, glossy plastic
instead of metal, and smaller clutter.

```bash
scripts/run_pipeline.sh --cad cad_lego --name lego_480 --clips 1 --res 480 --units mm \
  --prompts configs/cosmos/lego/lego_bench.json --negative configs/cosmos/lego/negative_prompt.json \
  --set "camera.distance=[0.25,0.5]" --set "camera.distance_sweep=[-0.08,0.08]" --set "camera.target_jitter=0.04" \
  --set "scene.placement_area=[[-0.12,0.12],[-0.08,0.08]]" --set "scene.parts_per_clip=[5,10]" \
  --set "scene.max_instances_per_class=2" --set "scene.min_spacing=0.005" \
  --set "randomization.part_material.randomize_prob=1.0" --set "randomization.part_material.metal_prob=0.0" \
  --set "randomization.part_material.paint_roughness=[0.05,0.35]" --set "randomization.clutter.size=[0.01,0.04]" \
  --set "randomization.clutter.margin=0.05"
```

| Option | Meaning |
|---|---|
| `--cad DIR` | Folder with CAD/STL files |
| `--name NAME` | Dataset goes to `output/NAME`, the Cosmos batch to `output/NAME_cosmos` |
| `--clips N`, `--frames N` | Number of clips (default 20) and frames per clip (default 61) |
| `--res 256\|480` | Cosmos resolution; also sets the matching render size (256: 1200x720, ~3 min per clip; 480: 1248x720, ~36 min per clip) |
| `--steps N` | Cosmos sampling steps (default 50) |
| `--looks N` | Prompts (looks) per clip (default 1) |
| `--prompts FILE`, `--negative FILE` | Cosmos look prompt(s) and negative prompt (default: `configs/cosmos/`) |
| `--no-clutter` | Render without clutter |
| `--units mm\|m\|cm\|inch\|auto` | CAD units (default auto) |
| `--class-from file\|folder` | Class per file name (default) or per parent folder |
| `--scene USD`, `--scene-surface PRIM` | Predesigned scene and the prim to place parts on |
| `--scene-camera PRIM`, `--scene-lights-only` | Render from a scene camera; light with the scene's lights only |
| `--spark USER@HOST` | Cosmos machine (default `h@100.123.185.58`) |
| `--set KEY=VALUE` | Any config override, repeatable |
| `--skip-sim` | Reuse the rendered dataset (only the Cosmos part) |
| `--skip-cosmos` | Only render the dataset |

- **Resuming:** re-running with the same `--name` keeps finished clips, finished Cosmos jobs and copied outputs. If
  the workstation or the connection drops while Cosmos runs, Cosmos keeps running on the Spark; re-run the same
  command with `--skip-sim` to wait for it and copy the results back.
- **Background:** for long runs, start it detached so closing the terminal does not stop it:
  `nohup scripts/run_pipeline.sh ... > run_001.log 2>&1 &` and follow it with `tail -f run_001.log`.
- **SSH:** the Spark password is asked once per run (`ssh-copy-id h@100.123.185.58` removes the prompt).
  `SPARK` and `ISAACLAB_DIR` override the Spark address and the Isaac Lab checkout.
- **Results:** `output/NAME/annotations/` (COCO for rendered, Cosmos and both), `output/NAME/viz/` (label preview
  videos), `output/NAME/viz/side_by_side_rgb_cosmos/<clip>.mp4` (render | Cosmos | labels, every clip),
  `output/NAME_cosmos/results/` (raw Cosmos outputs).

## Viewing results and side-by-side videos

`scripts/visualize.py` draws the labels (masks, 2D boxes, 3D boxes, pose axes) on the images. Run it from the
`worldsynth` folder with the Isaac Lab Python (`PY=~/research/isaaclab/IsaacLab/.venv/bin/python`):

```bash
# side by side: render | Cosmos | Cosmos with labels, one mp4 per clip, all frames
$PY scripts/visualize.py output/NAME --image_dir rgb_cosmos --all_clips --side_by_side
#   -> output/NAME/viz/side_by_side_rgb_cosmos/clip_0000.mp4, clip_0001.mp4, ...

# only some clips
$PY scripts/visualize.py output/NAME --image_dir rgb_cosmos --clip 0 3 --side_by_side

# labels on the rendered images or the Cosmos images, as a video per clip
$PY scripts/visualize.py output/NAME --image_dir rgb --clip 0 --all --video          # -> viz/rgb/clip_0000.mp4
$PY scripts/visualize.py output/NAME --image_dir rgb_cosmos --clip 0 --all --video   # -> viz/rgb_cosmos/clip_0000.mp4

# single frames as PNG
$PY scripts/visualize.py output/NAME --image_dir rgb_cosmos --clip 0 --frames 0 30 60
```

`run_pipeline.sh` makes the side-by-side videos of every clip automatically after the Cosmos import. With several
looks per clip (`--looks`), the Cosmos images are in `rgb_cosmos_<look>/`; pass that folder as `--image_dir`.

## Quick start (step by step)

```bash
cd ~/research/isaaclab/IsaacLab
.venv/bin/python ../worldsynth/scripts/make_example_cad.py ../worldsynth/cad_examples     # or use your CAD folder

# stage 1: 20 clips x 61 frames at 1248x720 (~20 s per clip on an RTX PRO 5000 laptop GPU)
.venv/bin/python ../worldsynth/scripts/generate.py --config ../worldsynth/configs/default.yaml \
    cad.dir=../worldsynth/cad_examples output.dir=../worldsynth/output/run_000

# check the labels visually (writes PNGs to output/run_000/viz/rgb/)
.venv/bin/python ../worldsynth/scripts/visualize.py ../worldsynth/output/run_000 --clip 0 1 --frames 0 30 60
```

Arguments after the options override the YAML (`section.key=value`, values parsed as YAML), e.g.
`dataset.num_clips=200 camera.distance=[0.4,0.7] scene.pose_mode=flat`. Runs resume: finished clips in
`output.dir` are kept, so re-running with a larger `dataset.num_clips` adds clips. Clip `i` always uses seed
`(output.seed, i)`, so it is reproducible.

`generate.py --convert_only ...` only converts the CAD files and prints the parts, classes and sizes; run it first on
a new CAD folder to check units and classes.

## Included assets

- `cad_lego/`: 8 LEGO car pieces as STL (millimeters), the source files of the LEGO test runs.
- `assets/lego/`: the same pieces converted to USD, usable directly as `--cad assets/lego --units mm`.
- `assets/scenes/example_cell.usda`: example scene for the scene options and the GUI.

## CAD input

| Format | Converter |
|---|---|
| STEP, IGES, JT, Parasolid, SolidWorks, CATIA, Creo, Inventor, ACIS, 3DXML, Rhino | HOOPS (`omni.kit.converter.hoops_core`) |
| STL, OBJ, FBX, glTF/GLB | `omni.kit.asset_converter` |
| USD, USDA, USDC, USDZ | used directly |

- **Classes**: `cad.class_from: file` makes every file its own class (file name); `folder` uses the parent folder,
  so `cad/flange/a.step` and `cad/flange/b.step` are both `flange`.
- **Units**: `cad.units: auto` trusts the units in the file (STEP carries them). Unit-less meshes larger than 2 units
  are assumed to be millimeters (logged as a warning). Set `cad.units: mm` (or `m`, `cm`, `inch`) to be explicit.
- **Part frame**: poses refer to the CAD file's own origin and axes, scaled to meters, so a pose estimator can use the
  same CAD model.
- Converted USD files are cached in `cad.cache_dir` and reused while the source file is unchanged.

## Dataset layout

```
output/run_000/
  dataset.json                 config, categories, parts (source, scale, bbox), label conventions
  annotations/coco_rgb.json    COCO (boxes, RLE masks; occluded/tiny objects as iscrowd=1)
  clips/clip_0000/
    clip.json                  everything randomized in this clip (parts, poses, materials, lights, camera path)
    rgb/000000.png
    depth/000000.png           uint16, millimeters to the image plane, 0 = invalid
    instance/000000.png        uint16, per-clip instance_id of the parts, 0 = background
    semantic/000000.png        uint8, class_id, 0 = background
    segmentation/000000.png    uint16, whole scene (parts, surfaces 1000+, clutter 2000+); Cosmos seg control
    labels/000000.json         per frame: camera K and T_world_cam, and per object:
                               class, bbox_2d, bbox_2d_amodal, mask_rle, pixel_area, visible_fraction,
                               truncation, ignore, pose (T_cam_obj, position, quaternion_wxyz, T_world_obj),
                               bbox_3d (min/max in the part frame, corners in the camera frame and in pixels)
    rgb_cosmos/000000.png      after the Cosmos import
```

Conventions: camera frame = OpenCV (x right, y down, z forward); `T_a_b` maps frame b to frame a; boxes are
`[x_min, y_min, x_max, y_max]` with max exclusive; COCO boxes are `[x, y, w, h]`.
`visible_fraction` = visible pixels / full silhouette (including the part outside the image), so it covers both
occlusion and truncation.

## What is randomized

Per clip, from `configs/default.yaml`:

- **Parts**: number, which parts (up to `max_instances_per_class` of a class), resting pose (`stable`: physically
  stable poses on a plane from the convex hull, weighted by likelihood), position and yaw without overlap.
- **Surface properties**: 80 % of parts get a sampled metal (aluminium, steel, brass, ...) or painted/anodized
  material with random roughness; table, floor and walls get colors from industrial palettes.
- **Lighting**: dome light intensity, color temperature and rotation; 1-3 ceiling area lights.
- **Clutter**: 0-8 unlabelled distractor primitives. `randomization.clutter.enabled=false` renders the same
  scenes without them: clutter has its own random stream, so with the same `output.seed` the parts, lights,
  materials and camera paths are identical (useful for a with/without-clutter comparison).
- **Camera**: distance, elevation, azimuth, focal length and look-at target, plus a smooth move over the clip.

Optional: `scene.environment_usd` references a USD of the real cell (machine, fixtures, robot).

## Stage 2: Cosmos Transfer

Uses the Cosmos3 Transfer setup on the DGX Spark from the kuka project (`kuka/scripts/cosmos/run_transfer.sh`,
see the kuka README "Cosmos Transfer on the DGX Spark").

```bash
cd ~/research/isaaclab/worldsynth
B=/home/h/cosmos-work/worldsynth/run_000_cosmos     # where the batch will live on the Spark

# 1. export the clips as a Cosmos batch: vision/depth/seg videos, one spec per clip, prompts
python scripts/cosmos_prepare.py output/run_000 --out output/run_000_cosmos --remote_dir $B

# 2. run it on the Spark (skips jobs that already have an output)
rsync -a output/run_000_cosmos/ h@100.123.185.58:$B/
ssh h@100.123.185.58 "nohup $B/run_batch.sh > ~/cosmos-work/worldsynth_run_000.log 2>&1 &"

# 3. fetch the outputs and import them
rsync -a h@100.123.185.58:/home/h/cosmos-work/cosmos/cookbooks/cosmos3/generator/transfer/outputs/Cosmos3-Nano/ \
    output/run_000_cosmos/results/
python scripts/cosmos_import.py output/run_000 --batch output/run_000_cosmos --results output/run_000_cosmos/results
python scripts/visualize.py output/run_000 --image_dir rgb_cosmos --clip 0 --all --video
```

(Plain `python` here means any environment with the dependencies, e.g. `~/research/isaaclab/IsaacLab/.venv/bin/python`.)

- **Looks**: `configs/cosmos/prompts/*.json` are structured prompts (machine shop, electronics assembly, dim tending
  cell); `{parts}` is replaced by the class names. Each job gets its own prompt (`prompts/<job>.json`): when the
  clip has clutter, a subject is added that names it as everyday non-metal items (cube = cardboard box,
  cylinder = plastic cup, sphere = rubber ball, cone = plastic traffic cone), so Cosmos does not turn it into
  part-like metal objects. Clips without clutter get no such subject. Clips are spread round-robin over the looks;
  `--variants_per_clip 3` renders every clip in every look (`rgb_cosmos_<look>/`). Add your own look by adding a file.
- **Controls**: `--edge 1.0 --depth 0.5 --seg 0.5` (weights; 0 disables). Edges are computed by Cosmos from the
  vision video; depth (near = bright) and segmentation (random color per segment) come from the render.
- **Size and speed** (measured on the Spark, 61-frame clip, 50 steps, edge + depth + seg):

  | `--resolution` (16:9) | Output | Time per step | Per clip |
  |---|---|---|---|
  | `256` | 320 × 192 | 3.1 s | 200 s incl. model loading |
  | `480` | 832 × 480 | about 40 s (first steps about 95 s) | 2141 s (36 min) |

  Lowest free memory was about 70 GB. The output has as many frames as the clip, at the clip's fps.
- **Aspect ratio**: Cosmos center-crops its input to the output aspect ratio, so the dataset must be rendered at
  exactly that ratio or the labels end up offset (worst at the image borders). `dataset.resolution` defaults to
  1248 × 720 for `480` (832 × 480); use `dataset.resolution=[1200,720]` for `256` (320 × 192).
  `cosmos_prepare.py` refuses a mismatched dataset and `cosmos_import.py` rejects mismatched outputs.
- **Alignment check**: the import compares how well segment boundaries line up with image edges in the generated
  frame versus the rendered one. Jobs with a ratio below `--min_alignment` (0.5) are not imported;
  `annotations/rgb_cosmos_report.json` lists every job's score. The import writes `coco_rgb_cosmos.json` and a mixed
  `coco_rgb+rgb_cosmos.json`.

Verified on the Spark: the `depth`/`seg` `control_path` keys and per-control `weight` match the cookbook specs
(weights are normalized to sum to 1), our depth convention (near = bright) matches the cookbook depth control, and
the spec's `negative_prompt` is used (see `sample_args.json`).

## Benchmarking (WP2)

The dataset is set up for the planned comparisons: train the same perception model on `coco_rgb.json` (classic
simulation), `coco_rgb_cosmos.json` (WFM) and the mixed file, and evaluate on real images from the Fontys lab.
`clip.json` records every randomized parameter, so failures can be traced back to lighting, materials, poses or
looks, and `dataset.renderer: pathtracing` gives a higher-fidelity simulation baseline.

## Tests

```bash
python -m pytest tests        # pure Python, no Isaac Sim needed
```

## Limitations of this first iteration

- Parts lie on one surface in stable poses; no bins, fixtures, stacking or physics-based piles yet.
- Materials are simple PBR (UsdPreviewSurface) without textures or wear; the dome light has no HDRI.
- Parts with several meshes and a CAD assembly per file are treated as one rigid part.
- One camera per clip; no wrist camera or stereo.
- The Cosmos alignment check catches labelled parts that moved or vanished, but not extra objects Cosmos invents in
  empty, featureless areas (e.g. an overexposed render); check the side-by-side videos.
