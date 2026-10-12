"""Export WorldSynth clips as a Cosmos Transfer batch (input videos, control videos, prompts, specs).

    python scripts/cosmos_prepare.py output/run_000 --out output/run_000_cosmos \
        --remote_dir /home/h/cosmos-work/worldsynth/run_000_cosmos

Then copy the batch to the Cosmos machine and run it there:

    rsync -a output/run_000_cosmos/ h@100.123.185.58:/home/h/cosmos-work/worldsynth/run_000_cosmos/
    ssh h@100.123.185.58 \
        'nohup /home/h/cosmos-work/worldsynth/run_000_cosmos/run_batch.sh > ~/cosmos-work/ws.log 2>&1 &'
"""

import argparse
import json
import logging
from pathlib import Path

from worldsynth import cosmos

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("dataset", help="WorldSynth dataset folder (output.dir of generate.py).")
    parser.add_argument("--out", required=True, help="Batch folder to create.")
    parser.add_argument("--remote_dir", default=None, help="Absolute path of the batch on the Cosmos machine.")
    parser.add_argument(
        "--prompts", nargs="*", default=None, help="Prompt templates (default: configs/cosmos/prompts/*.json)."
    )
    parser.add_argument("--negative_prompt", default=str(ROOT / "configs/cosmos/negative_prompt.json"))
    parser.add_argument("--variants_per_clip", type=int, default=1, help="Prompts (looks) per clip.")
    parser.add_argument("--clips", type=int, nargs="*", default=None, help="Clip indices (default: all).")
    parser.add_argument("--resolution", default="480", help="Cosmos resolution preset: 256, 480, 720 or 768.")
    parser.add_argument("--aspect_ratio", default="16,9", help="Cosmos aspect ratio, e.g. 16,9 or 1,1.")
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--edge", type=float, default=1.0, help="Edge control weight (0 disables).")
    parser.add_argument("--depth", type=float, default=0.5, help="Depth control weight (0 disables).")
    parser.add_argument("--seg", type=float, default=0.5, help="Segmentation control weight (0 disables).")
    parser.add_argument("--video_size", type=int, nargs=2, default=None, metavar=("W", "H"))
    parser.add_argument("--spec", nargs="*", default=[], help="Extra spec fields as key=json_value.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="[worldsynth] %(message)s")

    controls = {}
    if args.edge > 0:
        controls["edge"] = {"weight": args.edge, "preset_edge_threshold": "medium"}
    if args.depth > 0:
        controls["depth"] = {"weight": args.depth}
    if args.seg > 0:
        controls["seg"] = {"weight": args.seg}
    if not controls:
        parser.error("enable at least one control")
    overrides = {"resolution": args.resolution, "aspect_ratio": args.aspect_ratio, "num_steps": args.num_steps}
    for item in args.spec:
        key, _, value = item.partition("=")
        overrides[key] = json.loads(value)
    prompts = (
        [Path(p) for p in args.prompts] if args.prompts else sorted((ROOT / "configs/cosmos/prompts").glob("*.json"))
    )
    manifest = cosmos.prepare(
        args.dataset,
        args.out,
        prompts,
        Path(args.negative_prompt) if args.negative_prompt else None,
        remote_dir=args.remote_dir,
        clips=args.clips,
        variants_per_clip=args.variants_per_clip,
        spec_overrides=overrides,
        controls=controls,
        video_size=tuple(args.video_size) if args.video_size else None,
        seed=args.seed,
    )
    print(f"{len(manifest['jobs'])} Cosmos jobs in {args.out} (run_batch.sh runs them all)")


if __name__ == "__main__":
    main()
