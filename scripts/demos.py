"""Recorded robot demos (Isaac Lab / Mimic HDF5) through Cosmos.

    # 1. demos -> WorldSynth dataset (one clip per demo and camera; depth and segmentation if recorded)
    python scripts/demos.py import --hdf5 datasets/kuka/binpick_generated.hdf5 --cameras scene_cam wrist_cam \
        --out output/kuka_demos --max_demos 10 --seg_names "1=robot,2=table,3=ground,4-5=bin,10-15=soup_can"

    # 2. the usual Cosmos steps (prepare, send, fetch), e.g. scripts/run_pipeline.sh --stages prepare,send,fetch

    # 3. Cosmos frames -> a copy of the HDF5 with the camera images replaced (states and actions unchanged)
    python scripts/demos.py export --dataset output/kuka_demos --out datasets/kuka/binpick_cosmos.hdf5

scripts/run_pipeline.sh --demos ... does all of this in one go.
"""

import argparse
import logging

from worldsynth import demos


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    imp = sub.add_parser("import", help="HDF5 demos -> WorldSynth dataset")
    imp.add_argument("--hdf5", required=True)
    imp.add_argument("--cameras", nargs="+", required=True, help="Image observations, e.g. scene_cam wrist_cam.")
    imp.add_argument("--out", required=True, help="Dataset folder to write.")
    imp.add_argument("--max_demos", type=int, default=None)
    imp.add_argument("--seg_names", default=None, help='Class per segmentation id: "1=robot,10-15=can" or a JSON file.')
    imp.add_argument("--fps", type=int, default=20)
    imp.add_argument("--include_failed", action="store_true", help="Also import demos marked as failed.")
    imp.add_argument("--overwrite", action="store_true")
    exp = sub.add_parser("export", help="Cosmos frames -> HDF5 copy with replaced camera images")
    exp.add_argument("--dataset", required=True, help="Dataset folder written by 'import' (after the Cosmos import).")
    exp.add_argument("--out", required=True, help="HDF5 file to write.")
    exp.add_argument("--image_dir", default="rgb_cosmos", help="Frames to use (rgb_cosmos or rgb_cosmos_<look>).")
    exp.add_argument(
        "--keep_missing",
        action="store_true",
        help="Also write demos where only some cameras have Cosmos frames (the others keep the originals).",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="[worldsynth] %(message)s")

    if args.cmd == "import":
        meta = demos.import_demos(
            args.hdf5, args.out, args.cameras, args.max_demos, demos.parse_seg_names(args.seg_names), args.fps,
            successful_only=not args.include_failed, overwrite=args.overwrite,
        )  # fmt: skip
        print(f"{len(meta['source']['demos'])} demos x {len(args.cameras)} cameras -> {args.out}")
    else:
        written = demos.export_demos(args.dataset, args.out, args.image_dir, args.keep_missing)
        print(f"{len(written)} demos -> {args.out}")


if __name__ == "__main__":
    main()
