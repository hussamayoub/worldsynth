"""Generate a labelled synthetic dataset from a folder of CAD files with Isaac Sim.

Run from the IsaacLab checkout (its .venv provides Isaac Sim), e.g.:

    ./.venv/bin/python ../worldsynth/scripts/generate.py --config ../worldsynth/configs/default.yaml \
        cad.dir=/data/affix/cad output.dir=/data/worldsynth/run_000 dataset.num_clips=50

Arguments after the options are ``section.key=value`` overrides of the YAML config. Interrupted runs resume: finished
clips are kept and the rest is generated again.
"""

import argparse
import logging
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
parser.add_argument("--config", default=None, help="YAML config (defaults apply when omitted).")
parser.add_argument("--convert_only", action="store_true", help="Only convert the CAD files to USD and report them.")
parser.add_argument("overrides", nargs="*", help="Config overrides, e.g. dataset.num_clips=10 output.overwrite=true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
# Isaac Lab's kit files disable the Replicator orchestrator for RL speed; synthetic data capture needs it.
orchestrator = "--/exts/omni.replicator.core/Orchestrator/enabled=true"
args.kit_args = f"{args.kit_args} {orchestrator}" if args.kit_args else orchestrator
app = AppLauncher(args, enable_cameras=True).app

from worldsynth.config import load_config
from worldsynth.generate import load_parts, run

# Kit routes Python logging into its own log at warning level; print our progress directly.
logger = logging.getLogger("worldsynth")
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter("[worldsynth] %(message)s"))
logger.addHandler(handler)
logger.setLevel(logging.INFO)
logger.propagate = False

cfg = load_config(args.config, args.overrides)
try:
    if args.convert_only:
        entries, categories = load_parts(cfg)
        print(f"{len(entries)} parts in {len(categories)} classes: {[c['name'] for c in categories]}")
    else:
        run(cfg)
finally:
    app.close()
