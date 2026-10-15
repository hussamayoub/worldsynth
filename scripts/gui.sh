#!/bin/bash
# Start the WorldSynth GUI with the Isaac Lab Python (it needs pxr to read USD scenes and runs Isaac Sim for previews).
#   scripts/gui.sh                       (ISAACLAB_DIR defaults to ../IsaacLab next to this repo)
ROOT=$(cd "$(dirname "$0")/.." && pwd)
export ISAACLAB_DIR=${ISAACLAB_DIR:-$ROOT/../IsaacLab}
exec "$ISAACLAB_DIR/.venv/bin/python" -m worldsynth.gui "$@"
