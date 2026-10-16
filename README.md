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
so the geometry stays put while the appearance becomes realistic. The generated frames are imported next to the
rendered ones and reuse their labels; an alignment check rejects clips where labelled parts moved.

Clips (rather than single images) are used because Cosmos Transfer works on videos, and a camera move gives
temporally consistent views of one scene.

## Install

WorldSynth runs in the Isaac Lab environment (Isaac Sim 6.1). Install it editable into Isaac Lab's venv:

```bash
cd ~/research/isaaclab/IsaacLab
uv pip install --python .venv/bin/python --no-deps -e ../worldsynth
```

Its dependencies (numpy, opencv, imageio-ffmpeg, pyyaml, trimesh, networkx, scipy, h5py) are already in that venv.
The GUI also needs Qt (PySide6, the `gui` extra) and, on Linux with X11, the system library `libxcb-cursor0`:

```bash
uv pip install --python .venv/bin/python "PySide6-Essentials>=6.7"
sudo apt install libxcb-cursor0
```

Cosmos runs on the DGX Spark; set it up once as described in
[Cosmos on the DGX Spark (setup)](#cosmos-on-the-dgx-spark-setup).

## GUI

```bash
cd ~/research/isaaclab/worldsynth
scripts/gui.sh              # starts with the default settings; --restore loads the settings of the last run
```

The window (Qt) has four tabs on the left (Pipeline, Configuration, Prompts, Videos) and the run controls, image area
and log on the right; the divider between the two sides and the one between image and log can be dragged. It follows
the desktop's light or dark theme. Every start uses the defaults;
**Restore last session** brings back the settings of the last run (saved in `~/.config/worldsynth/gui.json`,
`WORLDSYNTH_GUI_SETTINGS` overrides), **Defaults** resets them.

**Pipeline tab**

- **Source:** render CAD parts in Isaac Sim, or **Recorded demos (HDF5)** from Isaac Lab teleop/Mimic: the demos file,
  the camera observations, how many demos and the class names of the segmentation ids (see
  [Recorded robot demos](#recorded-robot-demos-isaac-lab--mimic-hdf5)). Step 1 then becomes **1 Import demos**, and
  **Preview frame** shows the first frame with its labels without starting Isaac Sim.
- **Results (demos):** where step 4 writes the **Cosmos demos** HDF5 (the recorded demos with the Cosmos camera
  images), and the converter of step **5 Convert for training**, which merges the recorded and the Cosmos demos into
  one training dataset (from the robot project, e.g. the kuka GR00T/LeRobot converter, with the Python to run it, its
  arguments and the **training dataset** folder). The section shows the files both steps will write. **Run all
  steps** includes step 5 once a converter is chosen. In demos mode the CAD sections (Parts, Scene, Dataset) are
  hidden, and the Source section lists what each step does.
- **Parts:** the CAD/STL folder (optional when the scene provides the parts), units, and whether a class is a file
  or a folder.
- **Scene:** the built-in workcell, or a predesigned USD scene. After opening a scene, pick a **camera** from it
  (or random cameras around the parts), the **surface** to place the parts on, optionally the **parts in the scene**
  (a prim whose children are parts; *place randomly* or *keep as in scene*), how far the camera moves over a clip,
  and whether to use only the scene's own lights (default; added lights tend to wash a lit scene out, and Cosmos
  then invents objects in the blank areas).
- **Output:** the run name, the **dataset folder** (where Isaac Sim writes; empty: `output/<name>`) and the
  **Cosmos folder** (Cosmos input videos and batch; results are copied back to `<folder>/results`; empty:
  `output/<name>_cosmos`).
- **Dataset:** clips, frames per clip, parts per clip, clutter.
- **Cosmos:** resolution (256 or 480), steps, looks per clip, which looks to use and the negative prompt.
- **Spark:** host/IP, user and password, with **Test connection** (login, Cosmos runner, free memory). The password is
  only passed to `ssh` in the environment of the run; it is never saved (leave it empty with SSH keys).

**Configuration tab:** every setting of the YAML config with where its value comes from. **Load YAML...** picks the
base config (default `configs/default.yaml`); double-click a value to change it (blue, checked right away);
**Revert selected**, **Reset to defaults**, **Save as YAML...** to keep the result as a config file. Values set on the
Pipeline tab are grey: the Pipeline tab wins. Selecting a key shows its description from `default.yaml`.

**Prompts tab:** the look prompt and the negative prompt as editable text (JSON; `{parts}` becomes your class names).
Edits are used by the next run even when not saved; **Save** writes them to the file, **Save as...** to a new file
(new files in `configs/cosmos/` appear in the lists), **Revert** drops the edits.

**Videos tab:** plays the videos of a run in the large image area: the rendered Isaac Sim clips (view as rgb, rgb with
labels, depth, segmentation or instance masks), the imported Cosmos frames (with or without labels), the side-by-side
and label-preview videos, the Cosmos input videos (vision, depth, seg) and the raw Cosmos results. It opens on the
folders of the current run; **Browse...** shows any other dataset or Cosmos folder. Play/Pause, step, slider, speed.

**Run controls:** **Preview frame** renders one labelled frame (Isaac Sim starts, ~30 s) to check camera, surface and
part size. **Run all steps** runs the whole pipeline; the **steps** can also run one at a time:

| Button | Does | Result |
|---|---|---|
| 1 Generate (Isaac Sim) | renders the clips with all labels | dataset folder |
| 2 Make Cosmos videos | turns each clip into Cosmos input videos (rgb, depth, segmentation), prompts and specs | Cosmos folder `inputs/`, `specs/` |
| 3 Send to Cosmos | uploads the batch to the Spark and starts Cosmos there; returns right away | Cosmos runs on the Spark |
| 4 Fetch results | waits until Cosmos is done, copies the results back and imports them | Cosmos folder `results/`, `rgb_cosmos/` and COCO in the dataset, side-by-side videos; for demos also the HDF5 with Cosmos images |
| 5 Convert for training | (demos only) runs the converter on the original and Cosmos HDF5 files | training dataset folder, e.g. LeRobot for GR00T |

A step uses what the previous ones left in the dataset and Cosmos folders, so steps can run at different times (for
example send, close the GUI, and fetch the next day). **Stop** ends the local process; a Cosmos job already on the
Spark keeps running. **Open dataset** / **Open Cosmos folder** open the folders.

## Command line: `scripts/run_pipeline.sh`

The GUI runs this script; it does everything from a CAD folder to an imported Cosmos dataset: render the clips in Isaac
Sim, build the Cosmos batch, upload it to the Spark, run Cosmos there, copy the outputs back, import them, and write
label previews and side-by-side videos.

```bash
cd ~/research/isaaclab/worldsynth
scripts/run_pipeline.sh --cad /path/to/stl_folder --name run_001 --clips 20
```

Example: the LEGO pieces in this repo at Cosmos 480 (one clip: ~30 s in Isaac Sim, ~36 min on the Spark). The `--set`
lines adapt the scene to small plastic parts: closer camera, smaller placement area, glossy plastic, smaller clutter
close to the parts. (With a YAML file holding these values, `--config my_lego.yaml` does the same.)

```bash
scripts/run_pipeline.sh --cad assets/lego --name lego_480 --clips 1 --res 480 --units mm \
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
| `--name NAME` | Run name; names the Cosmos jobs and the default folders |
| `--config YAML` | Base config (default `configs/default.yaml`); the options here override it |
| `--dataset-dir DIR` | Where Isaac Sim writes the dataset (default `output/NAME`) |
| `--cosmos-dir DIR` | Cosmos batch folder; outputs are copied back to `DIR/results` (default `output/NAME_cosmos`) |
| `--clips N`, `--frames N` | Number of clips (default 20) and frames per clip (default 61) |
| `--res 256\|480` | Cosmos resolution; also sets the matching render size (256: 1200x720, ~3 min per clip; 480: 1248x720, ~36 min per clip) |
| `--steps N`, `--looks N` | Cosmos sampling steps (default 50), prompts (looks) per clip (default 1) |
| `--prompts FILE`, `--negative FILE` | Cosmos look prompt(s) and negative prompt (default: `configs/cosmos/`) |
| `--no-clutter` | Render without clutter |
| `--units mm\|m\|cm\|inch\|auto`, `--class-from file\|folder` | CAD units (default auto), class per file or folder |
| `--scene USD`, `--scene-surface PRIM` | Predesigned scene and the prim to place parts on |
| `--scene-camera PRIM`, `--scene-lights-only` | Render from a scene camera; light with the scene's lights only |
| `--scene-parts PRIM`, `--scene-parts-mode place\|keep` | Children of PRIM are labelled parts of the scene (see below); `--cad` becomes optional |
| `--spark USER@HOST` | Cosmos machine (default `h@100.123.185.58`) |
| `--set KEY=VALUE` | Any config override, repeatable |
| `--stages LIST` | Steps to run, comma separated: `generate`, `prepare` (Cosmos input videos), `send` (upload and start Cosmos, no waiting), `fetch` (wait, copy back, import); default all |
| `--skip-sim`, `--skip-cosmos` | Same as `--stages prepare,send,fetch` / `--stages generate` |
| `--demos HDF5` | Use recorded robot demos instead of CAD parts (next section); `--cad` and the scene options are not used |
| `--demo-cameras LIST`, `--max-demos N` | Camera observations to use (default `scene_cam,wrist_cam`), only the first N demos |
| `--seg-names SPEC`, `--demos-out FILE` | Class names of the segmentation ids; the HDF5 with Cosmos images (default `DATASET/demos_cosmos.hdf5`) |
| `--keep-missing` | Also export demos where only some cameras have Cosmos frames |
| `--convert-script PY`, `--convert-python BIN` | Converter to training data (robot project) and its interpreter; adds the `convert` step |
| `--convert-out DIR`, `--convert-args ARGS` | Training dataset folder (default `DATASET/training_data`); converter arguments (default `--input_files {inputs} --output_dir {output}`) |
| `--cosmos-only`, `--convert-overwrite` | Convert only the Cosmos demos; replace an existing training dataset |

- **Resuming:** re-running with the same `--name` keeps finished clips, finished Cosmos jobs and copied outputs. If
  the workstation or the connection drops while Cosmos runs, Cosmos keeps running on the Spark; run the same command
  with `--stages fetch` to wait for it and copy the results back.
- **Background:** `nohup scripts/run_pipeline.sh ... > run_001.log 2>&1 &`, then `tail -f run_001.log`.
- **SSH:** the Spark password is asked once per run (`ssh-copy-id h@100.123.185.58` removes the prompt).
  `ISAACLAB_DIR` overrides the Isaac Lab checkout (default `../IsaacLab`).

## Recorded robot demos (Isaac Lab / Mimic HDF5)

Demos recorded with Isaac Lab (teleop, Mimic `generate_dataset.py`) can go through Cosmos the same way, so a VLA can be
trained on photorealistic versions of the same trajectories. WorldSynth only reads the HDF5 layout of Isaac Lab's
recorder (`data/<demo>/obs/...`); it does not depend on the robot or task package.

For each camera observation `obs/<cam>` (uint8 RGB, shape `[T, H, W, 3]`) it also uses, when they were recorded:

- `obs/<cam>_depth`: depth [mm], integer, shape `[T, H, W, 1]`;
- `obs/<cam>_seg`: segmentation ids, integer, shape `[T, H, W, 1]`, 0 = background, the same id for the same object
  in every demo (masks and 2D boxes are derived from it).

Without them the clip is still usable; Cosmos is then conditioned on edges only. The kuka bin-picking task records
both (see its README).

```bash
cd ~/research/isaaclab/worldsynth
scripts/run_pipeline.sh --demos ../IsaacLab/datasets/kuka/binpick_generated.hdf5 --name binpick_cosmos \
  --max-demos 10 --seg-names "1=robot,2=table,3=ground,4-5=bin,10-15=soup_can"
```

- **generate** imports every demo and camera as one clip, `clips/<demo>__<cam>` (rgb, depth, segmentation, labels,
  COCO); failed demos are skipped. The Cosmos video keeps the camera's aspect ratio (square cameras give square
  videos), and all cameras of one demo get the same look.
- **fetch** also writes a copy of the HDF5 in which each camera's images are the Cosmos frames; states, actions and
  attributes are unchanged, so it converts to LeRobot/GR00T like the original. A demo is written only when all its
  cameras have Cosmos frames. With several looks (`--looks`), every look gets its own file, `demos_cosmos_<look>.hdf5`.
- **convert** (with `--convert-script`) runs the robot project's converter on the original HDF5 and every Cosmos HDF5,
  so each Cosmos demo becomes an extra training episode with the same states and actions. WorldSynth only runs the
  script; it needs no robot code. For the kuka GR00T setup:

```bash
scripts/run_pipeline.sh --demos ../IsaacLab/datasets/kuka/binpick_generated.hdf5 --name binpick_cosmos \
  --max-demos 10 --seg-names "1=robot,2=table,3=ground,4-5=bin,10-15=soup_can" \
  --convert-script ../kuka/scripts/gr00t/convert_to_gr00t_lerobot.py --convert-python ~/venvs/convert/bin/python \
  --convert-out ../IsaacLab/datasets/kuka/binpick_lerobot_cosmos
```

  `{inputs}` in `--convert-args` becomes the HDF5 files and `{output}` the output folder; add converter options there,
  e.g. `--convert-args "--input_files {inputs} --output_dir {output} --fps 20"`.

The steps also run on their own:

```bash
python scripts/demos.py import --hdf5 demos.hdf5 --cameras scene_cam wrist_cam --out output/demos --max_demos 10
python scripts/demos.py export --dataset output/demos --out demos_cosmos.hdf5 [--keep_missing]
python scripts/demos.py convert --hdf5 demos.hdf5 demos_cosmos.hdf5 --out lerobot_dataset \
  --script ../kuka/scripts/gr00t/convert_to_gr00t_lerobot.py --python ~/venvs/convert/bin/python
```

`--keep_missing` writes demos where only some cameras have Cosmos frames (e.g. only the scene camera was sent) and
keeps the original images for the others.

## Configuration

All settings live in `configs/default.yaml`, one commented key per setting (parts, scene, lighting, materials,
clutter, camera, rendering). Use another YAML with `--config` (pipeline and `generate.py`) or **Load YAML...** in the
GUI; override single keys with `--set section.key=value` (values in YAML syntax) or on the Configuration tab.

Isaac Sim alone (no Cosmos), from the Isaac Lab folder:

```bash
cd ~/research/isaaclab/IsaacLab
.venv/bin/python ../worldsynth/scripts/generate.py --config ../worldsynth/configs/default.yaml \
    cad.dir=../worldsynth/assets/lego cad.units=mm output.dir=../worldsynth/output/run_000 dataset.num_clips=5
.venv/bin/python ../worldsynth/scripts/generate.py --convert_only cad.dir=/path/to/cad   # check units and classes
```

Clip `i` always uses seed `(output.seed, i)`, so runs are reproducible and resume: finished clips in `output.dir` are
kept, and re-running with a larger `dataset.num_clips` adds clips.

## Using your own USD scene

Any USD scene works, in any units and with y or z up (converted to meters and z up). WorldSynth references it instead
of the built-in table and walls, places the parts on the chosen surface (the top of its bounding box, minus
`scene.surface_margin`), adds the clutter around them, and renders from the chosen scene camera with its focal length
and aperture. The camera moves by `camera.azimuth_sweep` / `elevation_sweep` / `distance_sweep` around the point it
looks at; the first frame is exactly the scene camera. The scene's top-level objects are labelled so the whole-scene
segmentation (Cosmos seg control) covers them.

```bash
scripts/run_pipeline.sh --cad assets/lego --units mm --name scene_001 --clips 20 \
  --scene assets/scenes/example_cell.usda --scene-surface /Cell/Workbench/Top \
  --scene-camera /Cell/Cameras/BenchCam --scene-lights-only \
  --set "camera.azimuth_sweep=[-10,10]" --set "camera.elevation_sweep=[-3,3]" --set "camera.distance_sweep=[0,0]"
```

Prim paths are the paths inside the scene file, as the GUI lists them. The surface is used as an axis-aligned box, so
a strongly rotated table gives a placement area that overhangs it. With a scene camera, parts are only placed where
the camera sees the surface (`scene.placement_in_view`).

### Parts that are in the scene

Your scene can bring its own parts: put them under one prim (e.g. `/Cell/Parts`) and select it as **Parts in scene**
(`--scene-parts /Cell/Parts`). Every child of that prim is one labelled part; its class is its semantic `class` label
if it has one, else its name without a trailing number (`Bolt_01`, `Bolt_02` -> `Bolt`). Everything else in the scene
stays as it is. Two modes:

- **place randomly** (`--scene-parts-mode place`): each clip shows a random subset (`parts per clip`) at random
  stable poses on the surface, like CAD parts; the other scene parts are hidden.
- **keep as in scene** (`keep`): all scene parts stay where the scene puts them, including animation (e.g. parts
  riding a conveyor). They are labelled in every frame where they are in that frame, and get random materials.
  CAD parts, if a CAD folder is set too, are placed around them.

CAD parts and scene parts can be mixed. `assets/scenes/example_cell.usda` has `/Cell/Parts` with two bolts, a block,
and `Block_02` riding a conveyor:

```bash
scripts/run_pipeline.sh --name scene_parts_001 --clips 10 \
  --scene assets/scenes/example_cell.usda --scene-surface /Cell/Workbench/Top \
  --scene-camera /Cell/Cameras/BenchCam --scene-lights-only \
  --scene-parts /Cell/Parts --scene-parts-mode keep \
  --set "camera.azimuth_sweep=[-8,8]" --set "camera.elevation_sweep=[0,0]" --set "camera.distance_sweep=[0,0]"
```

Animated parts move with the stage time: the timeline advances by one step per rendered frame and keeps running
across clips, so an animation that ends leaves its parts at their last position.

## CAD input

| Format | Converter |
|---|---|
| STEP, IGES, JT, Parasolid, SolidWorks, CATIA, Creo, Inventor, ACIS, 3DXML, Rhino | HOOPS (`omni.kit.converter.hoops_core`) |
| STL, OBJ, FBX, glTF/GLB | `omni.kit.asset_converter` |
| USD, USDA, USDC, USDZ | used directly |

- **Classes:** `cad.class_from: file` makes every file its own class (file name); `folder` uses the parent folder,
  so `cad/flange/a.step` and `cad/flange/b.step` are both `flange`.
- **Units:** `cad.units: auto` trusts the units in the file (STEP carries them). Unit-less meshes larger than 2 units
  are assumed to be millimeters (logged as a warning). Set `cad.units: mm` (or `m`, `cm`, `inch`) to be explicit.
- **Part frame:** poses refer to the CAD file's own origin and axes, scaled to meters, so a pose estimator can use the
  same CAD model.
- Converted USD files are cached in `cad.cache_dir` and reused while the source file is unchanged.

## What is randomized

Per clip (all ranges in the config):

- **Parts:** number, which parts (up to `max_instances_per_class` of a class), resting pose (`stable`: physically
  stable poses on a plane from the convex hull, weighted by likelihood), position and yaw without overlap.
- **Materials:** 80 % of parts get a sampled metal (aluminium, steel, brass, ...) or painted/anodized material with
  random roughness; the built-in table, floor and walls get colors from industrial palettes.
- **Lighting:** dome light intensity, color temperature and rotation; 1-3 ceiling area lights.
- **Clutter:** 0-8 unlabelled distractor primitives near the parts (`randomization.clutter.margin`).
  `randomization.clutter.enabled=false` renders the same scenes without them: clutter has its own random stream, so
  with the same seed the parts, lights, materials and camera paths are identical.
- **Camera:** distance, elevation, azimuth, focal length and look-at target, plus a smooth move over the clip (or the
  scene camera with its sweep).

## Dataset layout

```
<dataset folder>/
  dataset.json                 config, categories, parts (source, scale, bbox), label conventions
  annotations/                 COCO: coco_rgb.json, coco_rgb_cosmos.json, coco_rgb+rgb_cosmos.json,
                               rgb_cosmos_report.json (alignment of every Cosmos job)
  clips/clip_0000/
    clip.json                  everything randomized in this clip (parts, poses, materials, lights, camera path)
    rgb/000000.png
    rgb_cosmos/000000.png      after the Cosmos import (rgb_cosmos_<look>/ with several looks per clip)
    depth/000000.png           uint16, millimeters to the image plane, 0 = invalid
    instance/000000.png        uint16, per-clip instance_id of the parts, 0 = background
    semantic/000000.png        uint8, class_id, 0 = background
    segmentation/000000.png    uint16, whole scene (parts, scene objects 1000+, clutter 2000+); Cosmos seg control
    labels/000000.json         per frame: camera K and T_world_cam, and per object:
                               class, bbox_2d, bbox_2d_amodal, mask_rle, pixel_area, visible_fraction,
                               truncation, ignore, pose (T_cam_obj, position, quaternion_wxyz, T_world_obj),
                               bbox_3d (min/max in the part frame, corners in the camera frame and in pixels)
  viz/                         label previews and side-by-side videos
```

Conventions: camera frame = OpenCV (x right, y down, z forward); `T_a_b` maps frame b to frame a; boxes are
`[x_min, y_min, x_max, y_max]` with max exclusive; COCO boxes are `[x, y, w, h]`; occluded or tiny objects are
`iscrowd=1` in COCO. `visible_fraction` = visible pixels / full silhouette (including the part outside the image), so
it covers both occlusion and truncation. Cosmos images reuse the labels of the rendered frame with the same number.

## Viewing results

`scripts/visualize.py` draws the labels (masks, 2D boxes, 3D boxes, pose axes) on the images. With
`PY=~/research/isaaclab/IsaacLab/.venv/bin/python`, from the `worldsynth` folder:

```bash
$PY scripts/visualize.py output/NAME --image_dir rgb_cosmos --all_clips --side_by_side   # render | Cosmos | labels
$PY scripts/visualize.py output/NAME --image_dir rgb --clip 0 --all --video              # labels on the renders
$PY scripts/visualize.py output/NAME --image_dir rgb_cosmos --clip 0 --frames 0 30 60    # single frames as PNG
```

`--clip` takes clip numbers or folder names (`--clip demo_0__scene_cam` for imported demos). Videos go to
`output/NAME/viz/` (`side_by_side_rgb_cosmos/<clip>.mp4`, `<image_dir>/<clip>.mp4`). The pipeline writes
the side-by-side videos of every clip after the Cosmos import.

## Cosmos on the DGX Spark (setup)

The Cosmos Framework with `Cosmos3-Nano` is installed natively on the Spark (outside Docker), in `~/cosmos-work`.
One-time steps, on the Spark:


1. **Hugging Face licenses** (browser, with the account whose token is on the Spark): accept
   [nvidia/Cosmos3-Nano](https://huggingface.co/nvidia/Cosmos3-Nano) and
   [nvidia/Cosmos-Guardrail1](https://huggingface.co/nvidia/Cosmos-Guardrail1) (the safety models the runs download).
   The token in `~/.cache/huggingface/token` works right after accepting; no new token is needed.
2. **System packages** (on the Spark, needs `sudo`):

   ```bash
   sudo apt-get update && sudo apt-get install -y --no-install-recommends ffmpeg libx11-dev tree
   ```

3. **uv, repositories, and environment** (no `sudo`; the install takes a while and creates an ~9 GB environment):

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh && source $HOME/.local/bin/env
   mkdir -p ~/cosmos-work && cd ~/cosmos-work
   git clone https://github.com/NVIDIA/cosmos.git
   cd cosmos && mkdir -p packages
   git clone https://github.com/NVIDIA/cosmos-framework.git packages/cosmos3
   cd packages/cosmos3
   GIT_LFS_SKIP_SMUDGE=1 uv sync --all-extras --group=cu130-train
   ```

   Use the `cu130` group (the Spark has CUDA 13 only). The `-train` group is needed even for inference.

4. **Hugging Face cache ownership** (only if the GR00T Docker container also runs on the Spark). It runs as root, so
   the cache it created (`~/.cache/huggingface`) belongs to root and Cosmos (running as your user) cannot download into it
   (`PermissionError: ... ~/.cache/huggingface/hub/...`). Give it back to your user, without `sudo`:

   ```bash
   docker run --rm -v $HOME/.cache/huggingface:/hf --entrypoint chown gr00t-spark -R 1000:1000 /hf
   ```

   GR00T keeps working (root can use the folder regardless of owner). Repeat this if GR00T later downloads a new model
   and Cosmos fails with the same error.

5. **Check:**

   ```bash
   cd ~/cosmos-work/cosmos/packages/cosmos3
   source .venv/bin/activate && export LD_LIBRARY_PATH= PYTHONNOUSERSITE=1
   python -c "import torch, cv2, natten; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0)); print(cv2.__version__, natten.HAS_LIBNATTEN)"
   ```

   Expect `2.10.0+cu130 13.0 True NVIDIA GB10` and `libnatten True`. PyTorch warns that the GB10 (compute capability
   12.1) is newer than it was built for (up to 12.0); this is harmless, GPU kernels and NATTEN attention run correctly.

6. **Runner.** `scripts/cosmos/run_transfer.sh` runs one spec with `Cosmos3-Nano` and a memory watchdog (stops the
   run if less than 8 GB is free, `MIN_FREE_GIB`). The pipeline calls it as `~/cosmos-work/run_transfer.sh` on the
   Spark; copy it there once, from the `worldsynth` folder:

   ```bash
   scp scripts/cosmos/run_transfer.sh h@100.123.185.58:~/cosmos-work/
   ```

#### Running Cosmos by hand

The pipeline writes the specs itself. To try the cookbook samples or your own spec directly on the Spark:

Specs live in `~/cosmos-work/cosmos/cookbooks/cosmos3/generator/transfer/specs/` (`depth`, `edge`, `blur`, `seg`,
`wsm`, `multi_control`). The cookbook specs generate at 720p with 50 steps, which is very slow on the Spark. Make a
low-resolution variant at the size the VLA sees (256 × 256, then downscale to 224 × 224) with fewer steps:

```bash
cd ~/cosmos-work/cosmos/cookbooks/cosmos3/generator/transfer/specs
python3 -c "
import json; s = json.load(open('multi_control.json'))
s.update(name='transfer_multi_control_256', resolution='256', aspect_ratio='1,1', num_steps=20)
json.dump(s, open('multi_control_256.json', 'w'), indent=2)"

nohup ~/cosmos-work/run_transfer.sh multi_control_256 > /dev/null 2>&1 &
tail -f ~/cosmos-work/run_multi_control_256.log     # progress; ends with "exit 0 after ... s"
```

Keep the spec in `specs/`: its `prompt_path` and `control_path` entries are relative to that folder. Supported sizes:
`resolution` `256` (with `aspect_ratio` `1,1` → 256 × 256, `16,9` → 320 × 192), `480`, `720`, `768`.

| Run on the Spark | Time | Lowest free memory |
|---|---|---|
| `multi_control` (edge + blur) at 256 × 256, 20 steps, 93 frames | 130 s in total (about 3.5 s per step, plus model loading) | 75 GB |
| `depth` at 720p, 50 steps, 121 frames | about 175 s per step, about 2.4 h (stopped early) | 65 GB |

The first run also downloads `Cosmos3-Nano` (about 35 GB) and the safety models (7 GB) into the Hugging Face cache.
Do not run Cosmos and GR00T training at the same time: both need most of the 128 GB of unified memory.

**Outputs** go to `transfer/outputs/Cosmos3-Nano/<spec name>/`: `vision.mp4` (the generated video), one
`control_<type>.mp4` per control, and `sample_args.json`. A side-by-side video of the input, the controls, and the
result (the input is in `inputs/vision.mp4`):

```bash
D=~/cosmos-work/cosmos/cookbooks/cosmos3/generator/transfer/outputs/Cosmos3-Nano/transfer_multi_control_256
ffmpeg -y -i $D/inputs/vision.mp4 -i $D/control_edge.mp4 -i $D/control_blur.mp4 -i $D/vision.mp4 -filter_complex \
"[0:v]fps=16,scale=256:256:force_original_aspect_ratio=increase,crop=256:256,setsar=1[a];[1:v]setsar=1[b];[2:v]setsar=1[c];[3:v]setsar=1[d];[a][b][c][d]hstack=inputs=4" \
-c:v libx264 -crf 18 -pix_fmt yuv420p $D/side_by_side.mp4
```

Copy results to the workstation with `scp h@100.123.185.58:<path> .`.

**Choosing controls.** Each control keeps different things from the input; its `weight` sets how strongly. `blur`
keeps the original colors and lighting, so a high blur weight makes the output look like the input (little variety).
`edge`, `depth`, and `seg` keep shapes and positions but leave appearance to the prompt, which is what visual
augmentation needs; keep edges strong enough that the gripper and can outlines stay sharp. The output has as many
frames as the control video (at 16 fps), so trim the control clip with `ffmpeg` for shorter, faster runs.

**Your own videos.** Copy a spec and point it at your video: `vision_path` for a camera video (edge and blur controls
are computed from it), or a `control_path` per control (e.g. a depth video rendered in Isaac Sim), and write a
`prompt.json` like the ones in `transfer/assets/<spec>/`. Only `edge` and `blur` are computed from a camera video; `depth`, `seg`, and
`wsm` need a rendered control video (`control_path`).

Put a custom negative prompt into the spec's **`negative_prompt`** field as a JSON string. `negative_prompt_file` is
ignored once the built-in defaults have set a negative prompt, and it is resolved relative to the working directory
(not the spec), falling back silently to a built-in file of the same name. Check what a run used in
`outputs/.../<name>/sample_args.json` (`negative_prompt`).

## Cosmos Transfer details

- **Looks:** `configs/cosmos/prompts/*.json` are structured prompts (machine shop, electronics assembly, dim tending
  cell; `configs/cosmos/lego/` for LEGO); `{parts}` is replaced by the class names. Each job gets its own prompt:
  when the clip has clutter, a subject names it as everyday non-metal items (cube = cardboard box, cylinder = plastic
  cup, sphere = rubber ball, cone = plastic traffic cone) so Cosmos does not turn it into part-like metal objects.
  Clips are spread round-robin over the looks; `--looks 3` renders every clip in every look.
- **Controls:** edge (computed by Cosmos from the video, weight 1.0), depth (near = bright) and segmentation (random
  color per segment) from the render (weight 0.5 each); see `scripts/cosmos_prepare.py`.
- **Speed** on the Spark, 61-frame clip, 50 steps, edge + depth + seg:

  | Resolution (16:9) | Output | Time per step | Per clip |
  |---|---|---|---|
  | `256` | 320 × 192 | 3.1 s | 200 s incl. model loading |
  | `480` | 832 × 480 | about 40 s (first steps about 95 s) | 2141 s (36 min) |

- **Aspect ratio:** Cosmos center-crops its input to the output aspect ratio, so the dataset is rendered at exactly
  that ratio (`--res` picks it); otherwise labels would be offset toward the borders. `cosmos_prepare.py` refuses and
  `cosmos_import.py` rejects mismatched sizes.
- **Alignment check:** the import compares how well segment boundaries line up with image edges in the generated frame
  versus the render; jobs below 0.5 are not imported. It does not catch extra objects Cosmos invents in empty,
  featureless areas, so look at the side-by-side videos.
- The steps of the pipeline also exist on their own: `scripts/cosmos_prepare.py` (export a batch),
  `scripts/cosmos_import.py` (import results); see their `--help`.

## Repository layout

```
configs/default.yaml           all generation settings, commented
configs/cosmos/                Cosmos look prompts and negative prompts (prompts/, lego/)
scripts/gui.sh                 the GUI
scripts/run_pipeline.sh        CAD folder -> dataset -> Cosmos -> imported dataset
scripts/generate.py            Isaac Sim stage (dataset only)
scripts/cosmos_prepare.py      export a dataset as a Cosmos batch
scripts/cosmos_import.py       import Cosmos results into a dataset
scripts/visualize.py           label previews and side-by-side videos
scripts/demos.py               recorded robot demos (HDF5) to a dataset, and Cosmos frames back to HDF5
scripts/cosmos/run_transfer.sh Cosmos job runner, copied to ~/cosmos-work/ on the Spark
scripts/make_example_scene.py  writes assets/scenes/example_cell.usda
src/worldsynth/                cad (load/convert), scene + scene_info (USD), randomize, generate (render loop),
                               writer + labels (outputs, COCO), demos, cosmos, viz + videos (viewing), gui (settings,
                               checks, commands) + gui_qt (the window), config
assets/lego/                   8 LEGO car pieces as USD (millimeters), the parts of the test runs
assets/scenes/example_cell.usda  example scene (centimeters, y up): workbench, conveyor, machine, two cameras,
                               own parts under /Cell/Parts (one animated on the conveyor)
tests/                         pure-Python tests (no Isaac Sim needed)
```

## Benchmarking (WP2)

The dataset is set up for the planned comparisons: train the same perception model on `coco_rgb.json` (classic
simulation), `coco_rgb_cosmos.json` (WFM) and the mixed file, and evaluate on real images from the Fontys lab.
`clip.json` records every randomized parameter, so failures can be traced back to lighting, materials, poses or
looks, and `dataset.renderer: pathtracing` gives a higher-fidelity simulation baseline.

## Tests

```bash
python -m pytest tests        # pure Python (numpy, opencv, trimesh, pyyaml; usd-core for the scene tests)
```

## Limitations of this first iteration

- Parts lie on one surface in stable poses; no bins, fixtures, stacking or physics-based piles yet.
- Materials are simple PBR (UsdPreviewSurface) without textures or wear; the dome light has no HDRI.
- Parts with several meshes and a CAD assembly per file are treated as one rigid part.
- One camera per clip; no wrist camera or stereo.
