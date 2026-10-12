"""Import Cosmos Transfer outputs into a WorldSynth dataset and write COCO files for them.

    rsync -a h@100.123.185.58:/home/h/cosmos-work/cosmos/cookbooks/cosmos3/generator/transfer/outputs/Cosmos3-Nano/ \
        output/run_000_cosmos/results/
    python scripts/cosmos_import.py output/run_000 --batch output/run_000_cosmos \
        --results output/run_000_cosmos/results

Generated frames go to ``clips/<clip>/rgb_cosmos/`` (``rgb_cosmos_<prompt>/`` with several prompts per clip) and reuse
the labels of the rendered frames. Writes ``annotations/coco_rgb_cosmos*.json`` and a report with the alignment of
each job; jobs below ``--min_alignment`` are not imported.
"""

import argparse
import logging

from worldsynth import cosmos
from worldsynth.labels import build_coco


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("dataset")
    parser.add_argument("--batch", required=True, help="Batch folder written by cosmos_prepare.py.")
    parser.add_argument("--results", required=True, help="Folder with <job name>/vision.mp4 from Cosmos.")
    parser.add_argument("--image_dir", default="rgb_cosmos")
    parser.add_argument("--min_alignment", type=float, default=0.5, help="Reject jobs below this alignment ratio.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="[worldsynth] %(message)s")

    report = cosmos.import_results(args.dataset, args.batch, args.results, args.image_dir, args.min_alignment)
    counts = {
        s: sum(j["status"] == s for j in report["jobs"]) for s in ("imported", "rejected", "aspect_mismatch", "missing")
    }
    print(f"jobs: {counts}")
    if report["image_dirs"]:
        coco = build_coco(args.dataset, report["image_dirs"])
        print(f"COCO for {report['image_dirs']}: {len(coco['images'])} images")
        mixed = build_coco(args.dataset, ["rgb", *report["image_dirs"]])
        print(f"COCO for rendered + Cosmos: {len(mixed['images'])} images")


if __name__ == "__main__":
    main()
