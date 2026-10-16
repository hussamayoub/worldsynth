#!/bin/bash
# Start the WorldSynth GUI with the Isaac Lab Python (it needs pxr to read USD scenes and runs Isaac Sim for previews).
#   scripts/gui.sh                       (ISAACLAB_DIR defaults to ../IsaacLab next to this repo)
ROOT=$(cd "$(dirname "$0")/.." && pwd)
export ISAACLAB_DIR=${ISAACLAB_DIR:-$ROOT/../IsaacLab}
PY=$ISAACLAB_DIR/.venv/bin/python
"$PY" -c "import PySide6" 2>/dev/null || {
  echo "The GUI needs Qt (PySide6). Install it into the Isaac Lab venv:"
  echo "  uv pip install --python $PY PySide6-Essentials"
  exit 1
}
# Qt's X11 plugin needs libxcb-cursor (Qt >= 6.5); without it the window cannot open
if [[ $(uname) == Linux && ${XDG_SESSION_TYPE:-x11} != wayland ]] && ! ldconfig -p | grep -q libxcb-cursor.so.0; then
  echo "Qt needs the system library libxcb-cursor0:  sudo apt install libxcb-cursor0"
  exit 1
fi
exec "$PY" -m worldsynth.gui "$@"
