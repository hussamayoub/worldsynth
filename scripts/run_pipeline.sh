#!/bin/bash
# WorldSynth end to end: CAD folder -> Isaac Sim dataset -> Cosmos Transfer on the DGX Spark -> imported dataset.
#
# Usage:
#   scripts/run_pipeline.sh --cad /path/to/stl_folder --name run_001 [options]
#
# Options:
#   --cad DIR          folder with CAD/STL files (required unless --skip-sim)
#   --name NAME        run name; dataset goes to output/NAME, Cosmos batch to output/NAME_cosmos (required)
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
#   --skip-sim         reuse an existing dataset in output/NAME
#   --skip-cosmos      only render the dataset
#   --set KEY=VALUE    extra generate.py config override, repeatable (e.g. --set camera.distance=[0.4,0.7])
#
# Environment:
#   SPARK=h@100.123.185.58          Cosmos machine
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
REMOTE_BASE=/home/${SPARK%%@*}/cosmos-work/worldsynth
REMOTE_OUTPUTS=/home/${SPARK%%@*}/cosmos-work/cosmos/cookbooks/cosmos3/generator/transfer/outputs/Cosmos3-Nano

CAD="" NAME="" CLIPS=20 FRAMES=61 RES=256 STEPS=50 LOOKS=1 CLUTTER=true UNITS=auto CLASS_FROM=file
SKIP_SIM=false SKIP_COSMOS=false EXTRA=() PROMPTS=() NEGATIVE=$ROOT/configs/cosmos/negative_prompt.json
while [[ $# -gt 0 ]]; do
  case $1 in
    --cad) CAD=$(realpath "$2"); shift 2 ;;
    --name) NAME=$2; shift 2 ;;
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
    --skip-sim) SKIP_SIM=true; shift ;;
    --skip-cosmos) SKIP_COSMOS=true; shift ;;
    --set) EXTRA+=("$2"); shift 2 ;;
    -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
    *) echo "unknown option $1 (see --help)"; exit 1 ;;
  esac
done
[[ -n $NAME ]] || { echo "--name is required"; exit 1; }
case $RES in
  256) WIDTH=1200 ;;   # Cosmos 256 16:9 outputs 320x192
  480) WIDTH=1248 ;;   # Cosmos 480 16:9 outputs 832x480
  *) echo "--res must be 256 or 480"; exit 1 ;;
esac

DATASET=$ROOT/output/$NAME
BATCH=$ROOT/output/${NAME}_cosmos
REMOTE_BATCH=$REMOTE_BASE/${NAME}_cosmos
SSH=(ssh -o ControlMaster=auto -o "ControlPath=$WS_SSH_CONTROL" -o ControlPersist=4h "$SPARK")
RSYNC_SSH="ssh -o ControlMaster=auto -o ControlPath=$WS_SSH_CONTROL -o ControlPersist=4h"
step() { echo; echo "=== $(date +%T) $*"; }

# 1. render the dataset ------------------------------------------------------------------------------------------------
if ! $SKIP_SIM; then
  [[ -d $CAD ]] || { echo "--cad folder not found: $CAD"; exit 1; }
  step "1/5 Isaac Sim: $CLIPS clips x $FRAMES frames at ${WIDTH}x720 from $CAD"
  (cd "$ISAACLAB_DIR" && "$PY" "$ROOT/scripts/generate.py" --config "$ROOT/configs/default.yaml" \
    "cad.dir=$CAD" "cad.cache_dir=$ROOT/cache/usd" "cad.units=$UNITS" "cad.class_from=$CLASS_FROM" \
    "output.dir=$DATASET" "dataset.num_clips=$CLIPS" "dataset.frames_per_clip=$FRAMES" \
    "dataset.resolution=[$WIDTH,720]" "randomization.clutter.enabled=$CLUTTER" "${EXTRA[@]}" 2>&1 \
    | grep --line-buffered -E '^\[worldsynth\]|Error|Traceback|^ +File|Exception' ) || true
  [[ -f $DATASET/annotations/coco_rgb.json ]] || { echo "dataset generation failed (no COCO file)"; exit 1; }
  "$PY" "$ROOT/scripts/visualize.py" "$DATASET" --clip 0 --all --video > /dev/null
  echo "label preview: $DATASET/viz/rgb/clip_0000.mp4"
fi
$SKIP_COSMOS && { echo "done (dataset only): $DATASET"; exit 0; }

# 2. Cosmos batch ------------------------------------------------------------------------------------------------------
step "2/5 Cosmos batch ($RES, $STEPS steps, $LOOKS look(s) per clip)"
"$PY" "$ROOT/scripts/cosmos_prepare.py" "$DATASET" --out "$BATCH" --remote_dir "$REMOTE_BATCH" \
  --resolution "$RES" --num_steps "$STEPS" --variants_per_clip "$LOOKS" --negative_prompt "$NEGATIVE" \
  ${PROMPTS[@]+--prompts "${PROMPTS[@]}"}

# 3. upload and run on the Spark ---------------------------------------------------------------------------------------
step "3/5 upload to $SPARK:$REMOTE_BATCH"
"${SSH[@]}" "mkdir -p $REMOTE_BASE && test -x ~/cosmos-work/run_transfer.sh" \
  || { echo "~/cosmos-work/run_transfer.sh missing on the Spark (copy kuka/scripts/cosmos/run_transfer.sh)"; exit 1; }
rsync -a --exclude results -e "$RSYNC_SSH" "$BATCH/" "$SPARK:$REMOTE_BATCH/"

step "4/5 Cosmos running on the Spark (log: $REMOTE_BATCH.log)"
# Detached, so a dropped connection does not stop it; re-running this script resumes and skips finished jobs.
# A lock serializes WorldSynth batches, and the job waits until no other Cosmos inference is using the GPU.
"${SSH[@]}" "rm -f $REMOTE_BATCH/DONE; nohup flock $REMOTE_BASE/.lock sh -c '
  while pgrep -f \"[c]osmos_framework.scripts.inference\" > /dev/null; do echo waiting for running Cosmos job; sleep 30; done
  $REMOTE_BATCH/run_batch.sh && touch $REMOTE_BATCH/DONE' > $REMOTE_BATCH.log 2>&1 < /dev/null &"
TOTAL=$("$PY" -c "import json; print(len(json.load(open('$BATCH/manifest.json'))['jobs']))")
LAST=""
while ! "${SSH[@]}" "test -f $REMOTE_BATCH/DONE"; do
  DONE=$("${SSH[@]}" "ls -d $REMOTE_OUTPUTS/${NAME}_cosmos__*/vision.mp4 2>/dev/null | wc -l")
  NOW=$("${SSH[@]}" "tail -qn1 $REMOTE_BATCH.log; ls -t ~/cosmos-work/run_${NAME}_cosmos__*.log 2>/dev/null | grep -v _mem.log | head -1 \
    | xargs -r grep -ho '[0-9]*/[0-9]* \[[^]]*\]' | tail -1" | tr '\n' ' ' || true)
  MSG="jobs done $DONE/$TOTAL, current: ${NOW:-starting}"
  [[ $MSG != "$LAST" ]] && echo "$(date +%T) $MSG"
  LAST=$MSG
  sleep 60
  if ! "${SSH[@]}" "test -f $REMOTE_BATCH/DONE || pgrep -f '$REMOTE_BATCH/[r]un_batch.sh' > /dev/null"; then
    echo "the Cosmos batch stopped without finishing; see $SPARK:$REMOTE_BATCH.log"; exit 1
  fi
done
"${SSH[@]}" "grep -h '^exit' ~/cosmos-work/run_${NAME}_cosmos__*.log | sort | uniq -c"

# 5. copy back and import ----------------------------------------------------------------------------------------------
step "5/5 copy outputs back and import"
mkdir -p "$BATCH/results"
rsync -a -e "$RSYNC_SSH" --include "${NAME}_cosmos__*/***" --exclude '*' "$SPARK:$REMOTE_OUTPUTS/" "$BATCH/results/"
"$PY" "$ROOT/scripts/cosmos_import.py" "$DATASET" --batch "$BATCH" --results "$BATCH/results"
for d in "$DATASET"/clips/clip_0000/rgb_cosmos*; do
  [[ -d $d ]] && "$PY" "$ROOT/scripts/visualize.py" "$DATASET" --image_dir "$(basename "$d")" --clip 0 --all --video > /dev/null
done

echo
echo "dataset:          $DATASET"
echo "COCO files:       $DATASET/annotations/"
echo "Cosmos report:    $DATASET/annotations/rgb_cosmos_report.json"
echo "Cosmos videos:    $BATCH/results/<job>/vision.mp4"
echo "label previews:   $DATASET/viz/"
