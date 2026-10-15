#!/bin/bash
# WorldSynth end to end: CAD folder -> Isaac Sim dataset -> Cosmos Transfer on the DGX Spark -> imported dataset.
#
# Usage:
#   scripts/run_pipeline.sh --cad /path/to/stl_folder --name run_001 [options]
#
# Options:
#   --cad DIR          folder with CAD/STL files (required unless --skip-sim or --scene-parts)
#   --name NAME        run name (required); names the Cosmos jobs and the default folders below
#   --config YAML      base config (default configs/default.yaml); the options here override it
#   --dataset-dir DIR  where Isaac Sim writes the dataset (default output/NAME)
#   --cosmos-dir DIR   Cosmos batch folder; outputs are copied back to DIR/results (default output/NAME_cosmos)
#   --clips N          number of clips (default 20)
#   --frames N         frames per clip (default 61)
#   --res 256|480      Cosmos resolution (default 256). Sets the matching dataset resolution:
#                      256 -> 1200x720 (~3 min per clip on the Spark), 480 -> 1248x720 (~36 min per clip)
#   --steps N          Cosmos sampling steps (default 50)
#   --looks N          prompts (looks) per clip (default 1)
#   --prompts FILES    Cosmos look prompts, repeatable or a folder (default configs/cosmos/prompts)
#   --negative FILE    Cosmos negative prompt (default configs/cosmos/negative_prompt.json)
#   --no-clutter       render without clutter
#   --units U          CAD units: auto|mm|m|cm|inch (default auto; unit-less STL > 2 units is taken as mm)
#   --class-from M     file (class = file name, default) | folder (class = parent folder)
#   --scene USD        predesigned USD scene instead of the built-in workcell (needs --scene-surface)
#   --scene-surface P  prim path in the scene to place the parts on (top of its bounding box)
#   --scene-camera P   camera prim path in the scene to render from (moves by the camera.*_sweep ranges)
#   --scene-parts P    prim path in the scene whose children are labelled parts (class = name without number)
#   --scene-parts-mode place|keep  place: random subset at random poses on the surface (default);
#                      keep: parts stay where the scene puts/animates them (still labelled, random materials)
#   --scene-lights-only  light the scene with its own lights only (no added randomized lights)
#   --spark USER@HOST  Cosmos machine (default $SPARK or h@100.123.185.58)
#   --demos HDF5       use recorded robot demos (Isaac Lab / Mimic HDF5) instead of rendering CAD parts: the generate
#                      step imports them (one clip per demo and camera, with depth/segmentation if recorded) and the
#                      fetch step also writes a copy of the HDF5 with the Cosmos images (see --demos-out)
#   --demo-cameras L   image observations to use, comma separated (default scene_cam,wrist_cam)
#   --max-demos N      only the first N demos
#   --seg-names S      class names of segmentation ids, e.g. "1=robot,2=table,10-15=soup_can"
#   --demos-out FILE   where the HDF5 with Cosmos images goes (default <dataset folder>/demos_cosmos.hdf5)
#   --stages LIST      which steps to run, comma separated (default generate,prepare,send,fetch):
#                        generate  render the dataset in Isaac Sim
#                        prepare   turn the clips into Cosmos input videos (rgb/depth/seg), prompts and specs
#                        send      upload the batch to the Spark and start Cosmos there (does not wait)
#                        fetch     wait until Cosmos is done, copy the results back and import them
#   --skip-sim         same as --stages prepare,send,fetch
#   --skip-cosmos      same as --stages generate
#   --set KEY=VALUE    extra generate.py config override, repeatable (e.g. --set camera.distance=[0.4,0.7])
#
# Environment:
#   SPARK=h@100.123.185.58          Cosmos machine (same as --spark)
#   ISAACLAB_DIR=../IsaacLab        Isaac Lab checkout whose .venv runs everything
#   WS_SSH_CONTROL=~/.ssh/ws-%C     SSH connection sharing socket (password is asked once)
#
# Re-running with the same --name resumes: finished clips, finished Cosmos jobs and copied outputs are kept.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
ISAACLAB_DIR=${ISAACLAB_DIR:-$ROOT/../IsaacLab}
PY=$ISAACLAB_DIR/.venv/bin/python
SPARK=${SPARK:-h@100.123.185.58}
WS_SSH_CONTROL=${WS_SSH_CONTROL:-$HOME/.ssh/ws-%C}

CAD="" NAME="" CLIPS=20 FRAMES=61 RES=256 STEPS=50 LOOKS=1 CLUTTER=true UNITS=auto CLASS_FROM=file
EXTRA=() PROMPTS=() NEGATIVE=$ROOT/configs/cosmos/negative_prompt.json
DEMOS="" DEMO_CAMERAS=scene_cam,wrist_cam MAX_DEMOS="" SEG_NAMES="" DEMOS_OUT=""
STAGES=generate,prepare,send,fetch SCENE_ARGS=() SCENE_PARTS="" CONFIG=$ROOT/configs/default.yaml DATASET="" BATCH=""
while [[ $# -gt 0 ]]; do
  case $1 in
    --cad) CAD=$(realpath "$2"); shift 2 ;;
    --name) NAME=$2; shift 2 ;;
    --config) CONFIG=$(realpath "$2"); shift 2 ;;
    --dataset-dir) DATASET=$(realpath -m "$2"); shift 2 ;;
    --cosmos-dir) BATCH=$(realpath -m "$2"); shift 2 ;;
    --clips) CLIPS=$2; shift 2 ;;
    --frames) FRAMES=$2; shift 2 ;;
    --res) RES=$2; shift 2 ;;
    --steps) STEPS=$2; shift 2 ;;
    --looks) LOOKS=$2; shift 2 ;;
    --prompts) if [[ -d $2 ]]; then PROMPTS+=("$(realpath "$2")"/*.json); else PROMPTS+=("$(realpath "$2")"); fi; shift 2 ;;
    --negative) NEGATIVE=$(realpath "$2"); shift 2 ;;
    --no-clutter) CLUTTER=false; shift ;;
    --units) UNITS=$2; shift 2 ;;
    --class-from) CLASS_FROM=$2; shift 2 ;;
    --scene) SCENE_ARGS+=("scene.scene_usd=$(realpath "$2")"); shift 2 ;;
    --scene-surface) SCENE_ARGS+=("scene.scene_surface=$2"); shift 2 ;;
    --scene-camera) SCENE_ARGS+=("scene.scene_camera=$2"); shift 2 ;;
    --scene-lights-only) SCENE_ARGS+=("scene.scene_lights_only=true"); shift ;;
    --scene-parts) SCENE_ARGS+=("scene.scene_parts=$2"); SCENE_PARTS=$2; shift 2 ;;
    --scene-parts-mode) SCENE_ARGS+=("scene.scene_parts_mode=$2"); shift 2 ;;
    --spark) SPARK=$2; shift 2 ;;
    --demos) DEMOS=$(realpath "$2"); shift 2 ;;
    --demo-cameras) DEMO_CAMERAS=$2; shift 2 ;;
    --max-demos) MAX_DEMOS=$2; shift 2 ;;
    --seg-names) SEG_NAMES=$2; shift 2 ;;
    --demos-out) DEMOS_OUT=$(realpath -m "$2"); shift 2 ;;
    --stages) STAGES=$2; shift 2 ;;
    --skip-sim) STAGES=prepare,send,fetch; shift ;;
    --skip-cosmos) STAGES=generate; shift ;;
    --set) EXTRA+=("$2"); shift 2 ;;
    -h|--help) sed -n '2,/^set -euo/{/^#/p}' "$0"; exit 0 ;;
    *) echo "unknown option $1 (see --help)"; exit 1 ;;
  esac
done
[[ -n $NAME ]] || { echo "--name is required"; exit 1; }
for st in ${STAGES//,/ }; do
  [[ $st =~ ^(generate|prepare|send|fetch)$ ]] || { echo "unknown stage $st (generate, prepare, send, fetch)"; exit 1; }
done
want() { [[ ",$STAGES," == *",$1,"* ]]; }
REMOTE_BASE=/home/${SPARK%%@*}/cosmos-work/worldsynth
REMOTE_OUTPUTS=/home/${SPARK%%@*}/cosmos-work/cosmos/cookbooks/cosmos3/generator/transfer/outputs/Cosmos3-Nano
case $RES in
  256) WIDTH=1200 ;;   # Cosmos 256 16:9 outputs 320x192 (square demo cameras use 1:1, 256x256)
  480) WIDTH=1248 ;;   # Cosmos 480 16:9 outputs 832x480
  *) echo "--res must be 256 or 480"; exit 1 ;;
esac

DATASET=${DATASET:-$ROOT/output/$NAME}
BATCH=${BATCH:-$ROOT/output/${NAME}_cosmos}
JOB_PREFIX=${NAME}_cosmos   # job names and the remote batch folder follow the run name, not the local folders
REMOTE_BATCH=$REMOTE_BASE/$JOB_PREFIX
# accept-new: a new host key is accepted without a terminal prompt (the GUI has none); changed keys still fail
SSH_OPTS="-o ControlMaster=auto -o ControlPath=$WS_SSH_CONTROL -o ControlPersist=4h -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new"
SSH=(ssh $SSH_OPTS "$SPARK")
RSYNC_SSH="ssh $SSH_OPTS"
step() { echo; echo "=== $(date +%T) $*"; }

# 1. generate: import recorded demos, or render the dataset -------------------------------------------------------------
if want generate && [[ -n $DEMOS ]]; then
  step "generate: import demos from $DEMOS (cameras $DEMO_CAMERAS)"
  "$PY" "$ROOT/scripts/demos.py" import --hdf5 "$DEMOS" --cameras ${DEMO_CAMERAS//,/ } --out "$DATASET" \
    ${MAX_DEMOS:+--max_demos "$MAX_DEMOS"} ${SEG_NAMES:+--seg_names "$SEG_NAMES"}
  echo "dataset: $DATASET"
elif want generate; then
  if [[ -n $CAD ]]; then
    [[ -d $CAD ]] || { echo "--cad folder not found: $CAD"; exit 1; }
    CAD_ARG="cad.dir=$CAD"
  else
    [[ -n $SCENE_PARTS ]] || { echo "--cad is required (or --scene-parts)"; exit 1; }
    CAD_ARG="cad.dir=null"
  fi
  step "generate: Isaac Sim, $CLIPS clips x $FRAMES frames at ${WIDTH}x720 from ${CAD:-the scene parts}"
  (cd "$ISAACLAB_DIR" && "$PY" "$ROOT/scripts/generate.py" --config "$CONFIG" \
    "$CAD_ARG" "cad.cache_dir=$ROOT/cache/usd" "cad.units=$UNITS" "cad.class_from=$CLASS_FROM" \
    "output.dir=$DATASET" "dataset.num_clips=$CLIPS" "dataset.frames_per_clip=$FRAMES" \
    "dataset.resolution=[$WIDTH,720]" "randomization.clutter.enabled=$CLUTTER" "${SCENE_ARGS[@]}" "${EXTRA[@]}" 2>&1 \
    | grep --line-buffered -E '^\[worldsynth\]|Error|Traceback|^ +File|Exception' ) || true
  [[ -f $DATASET/annotations/coco_rgb.json ]] || { echo "dataset generation failed (no COCO file)"; exit 1; }
  "$PY" "$ROOT/scripts/visualize.py" "$DATASET" --clip 0 --all --video > /dev/null
  echo "dataset: $DATASET   (label preview: $DATASET/viz/rgb/clip_0000.mp4)"
fi

# 2. prepare: Cosmos input videos, prompts and specs --------------------------------------------------------------------
if want prepare; then
  [[ -f $DATASET/dataset.json ]] || { echo "no dataset in $DATASET (run the generate step first)"; exit 1; }
  step "prepare: Cosmos batch ($RES, $STEPS steps, $LOOKS look(s) per clip) in $BATCH"
  "$PY" "$ROOT/scripts/cosmos_prepare.py" "$DATASET" --out "$BATCH" --remote_dir "$REMOTE_BATCH" \
    --resolution "$RES" --num_steps "$STEPS" --variants_per_clip "$LOOKS" --negative_prompt "$NEGATIVE" \
    --job_prefix "$JOB_PREFIX" ${PROMPTS[@]+--prompts "${PROMPTS[@]}"}
  echo "Cosmos input videos: $BATCH/inputs/<clip>/{vision,depth,seg}.mp4"
fi

# 3. send: upload and start Cosmos on the Spark ------------------------------------------------------------------------
if want send; then
  [[ -f $BATCH/manifest.json ]] || { echo "no Cosmos batch in $BATCH (run the prepare step first)"; exit 1; }
  step "send: upload to $SPARK:$REMOTE_BATCH"
  "${SSH[@]}" "mkdir -p $REMOTE_BASE && test -x ~/cosmos-work/run_transfer.sh" \
    || { echo "~/cosmos-work/run_transfer.sh missing on the Spark (copy kuka/scripts/cosmos/run_transfer.sh)"; exit 1; }
  rsync -a --exclude results -e "$RSYNC_SSH" "$BATCH/" "$SPARK:$REMOTE_BATCH/"
  if "${SSH[@]}" "pgrep -f '$REMOTE_BATCH/[r]un_batch.sh' > /dev/null"; then
    echo "this batch is already running on the Spark"
  else
    # Detached, so a dropped connection does not stop it; finished jobs are skipped when started again.
    # A lock serializes WorldSynth batches, and the job waits until no other Cosmos inference is using the GPU.
    "${SSH[@]}" "rm -f $REMOTE_BATCH/DONE; nohup flock $REMOTE_BASE/.lock sh -c '
      while pgrep -f \"[c]osmos_framework.scripts.inference\" > /dev/null; do echo waiting for running Cosmos job; sleep 30; done
      $REMOTE_BATCH/run_batch.sh && touch $REMOTE_BATCH/DONE' > $REMOTE_BATCH.log 2>&1 < /dev/null &"
    echo "Cosmos started on the Spark (log: $REMOTE_BATCH.log); use the fetch step to wait for it and copy back"
  fi
fi

# 4. fetch: wait for Cosmos, copy back, import -------------------------------------------------------------------------
if want fetch; then
  [[ -f $BATCH/manifest.json ]] || { echo "no Cosmos batch in $BATCH (run the prepare and send steps first)"; exit 1; }
  step "fetch: waiting for Cosmos on the Spark (log: $REMOTE_BATCH.log)"
  TOTAL=$("$PY" -c "import json; print(len(json.load(open('$BATCH/manifest.json'))['jobs']))")
  LAST=""
  while ! "${SSH[@]}" "test -f $REMOTE_BATCH/DONE"; do
    if ! "${SSH[@]}" "pgrep -f '$REMOTE_BATCH/[r]un_batch.sh' > /dev/null"; then
      echo "Cosmos is not running for this batch and has not finished; run the send step (log: $SPARK:$REMOTE_BATCH.log)"
      exit 1
    fi
    DONE=$("${SSH[@]}" "ls -d $REMOTE_OUTPUTS/${JOB_PREFIX}__*/vision.mp4 2>/dev/null | wc -l")
    NOW=$("${SSH[@]}" "tail -qn1 $REMOTE_BATCH.log; ls -t ~/cosmos-work/run_${JOB_PREFIX}__*.log 2>/dev/null | grep -v _mem.log | head -1 \
      | xargs -r grep -ho '[0-9]*/[0-9]* \[[^]]*\]' | tail -1" | tr '\n' ' ' || true)
    MSG="jobs done $DONE/$TOTAL, current: ${NOW:-starting}"
    [[ $MSG != "$LAST" ]] && echo "$(date +%T) $MSG"
    LAST=$MSG
    sleep 30
  done
  "${SSH[@]}" "grep -h '^exit' ~/cosmos-work/run_${JOB_PREFIX}__*.log | sort | uniq -c" || true
  step "fetch: copy back to $BATCH/results and import"
  mkdir -p "$BATCH/results"
  rsync -a -e "$RSYNC_SSH" --include "${JOB_PREFIX}__*/***" --exclude '*' "$SPARK:$REMOTE_OUTPUTS/" "$BATCH/results/"
  "$PY" "$ROOT/scripts/cosmos_import.py" "$DATASET" --batch "$BATCH" --results "$BATCH/results"
  for image_dir in $(ls -d "$DATASET"/clips/*/rgb_cosmos* 2>/dev/null | xargs -rn1 basename | sort -u); do
    "$PY" "$ROOT/scripts/visualize.py" "$DATASET" --image_dir "$image_dir" --all_clips --side_by_side
  done
  echo
  echo "dataset:          $DATASET"
  echo "COCO files:       $DATASET/annotations/"
  echo "Cosmos report:    $DATASET/annotations/rgb_cosmos_report.json"
  echo "Cosmos videos:    $BATCH/results/<job>/vision.mp4"
  echo "side by side:     $DATASET/viz/side_by_side_rgb_cosmos/<clip>.mp4  (render | Cosmos | labels)"
  if [[ -n $DEMOS ]] || [[ $("$PY" -c "import json; print(json.load(open('$DATASET/dataset.json')).get('generator'))") == "worldsynth demos" ]]; then
    OUT=${DEMOS_OUT:-$DATASET/demos_cosmos.hdf5}
    "$PY" "$ROOT/scripts/demos.py" export --dataset "$DATASET" --out "$OUT"
    echo "demos with Cosmos images: $OUT"
  fi
fi
