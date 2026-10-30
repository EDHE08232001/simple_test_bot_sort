"""Command line entry: `live`, `record`, `evaluate`, `plot`, or an interactive menu with no arguments."""
import argparse
import json
import sys
from pathlib import Path

from .config import ARMS, ROOT, default_device, resolve_weights


def build_parser():
    ap = argparse.ArgumentParser(prog="run.py", description=__doc__)
    sub = ap.add_subparsers(dest="cmd")

    def common(p):
        p.add_argument("--weights", default="yolov8s.pt", help="YOLO detector weights")
        p.add_argument("--device", default=default_device(), help="mps, cpu, or a CUDA id like 0")
        p.add_argument("--imgsz", type=int, default=640)
        p.add_argument("--det-conf", type=float, default=0.1, help="detector confidence floor")
        p.add_argument("--classes", type=int, nargs="*", default=None,
                       help="COCO class ids to track; give none to track every class "
                            "(default: person for live / MOT17, all classes for LaSOT)")

    live = sub.add_parser("live", help="track people on the device camera")
    common(live)
    live.add_argument("--camera", type=int, default=0)
    live.add_argument("--arm", choices=list(ARMS), default="gmc", help="starting config (g toggles)")
    live.add_argument("--no-mirror", action="store_true", help="do not flip the image like a mirror")

    rec = sub.add_parser("record", help="record a raw camera clip (no tracking)")
    rec.add_argument("out", help="output .mp4 path")
    rec.add_argument("--camera", type=int, default=0)
    rec.add_argument("--seconds", type=int, default=20)

    ev = sub.add_parser("evaluate", help="score the tracker on labeled MOT17 / LaSOT sequences")
    common(ev)
    ev.add_argument("--dataset", choices=["mot17", "lasot"], required=True)
    ev.add_argument("--data", required=True, help="dataset root folder")
    ev.add_argument("--split", default="train", help="MOT17 split folder (the test split has no labels)")
    ev.add_argument("--mot-variant", default="FRCNN", help="MOT17 detector variant to use (images/GT are the same)")
    ev.add_argument("--seqs", nargs="*", help="only these sequence names (MOT17: prefixes like MOT17-02 work)")
    ev.add_argument("--max-seqs", type=int, help="use only the first N sequences")
    ev.add_argument("--max-frames", type=int, help="use only the first N frames of each sequence (quick test)")
    ev.add_argument("--arms", nargs="+", choices=list(ARMS), default=list(ARMS), help="configs to compare")
    viz = ev.add_mutually_exclusive_group()
    viz.add_argument("--plot", dest="plot", action="store_true", default=None, help="show diagrams without asking")
    viz.add_argument("--no-plot", dest="plot", action="store_false", help="skip the diagrams without asking")

    pl = sub.add_parser("plot", help="draw diagrams from a saved results folder")
    pl.add_argument("results", help="a folder under results/ (or its results.json)")
    pl.add_argument("--no-show", action="store_true", help="only save the PNGs")
    return ap


def ask_visualize(plot):
    if plot is not None:
        return plot
    if not sys.stdin.isatty():
        return False
    return input("\nVisualize the metrics as diagrams? [y/N] ").strip().lower() in ("y", "yes")


def do_evaluate(args):
    from .evaluate import run_evaluation
    from .report import print_report, save_csv, visualize

    results, out_dir = run_evaluation(args)
    print_report(results)
    save_csv(results, out_dir)
    print(f"\nSaved results.json, summary.csv and raw tracker output in {out_dir}")
    if ask_visualize(args.plot):
        visualize(results, out_dir)


def do_plot(args):
    from .report import visualize

    path = Path(args.results)
    path = path / "results.json" if path.is_dir() else path
    visualize(json.loads(path.read_text()), path.parent, show=not args.no_show)


def menu_argv():
    print("BoT-SORT lab\n  1) Live camera\n  2) Evaluate on a labeled dataset (MOT17 / LaSOT)\n"
          "  3) Draw diagrams from saved results")
    choice = input("Choose 1-3: ").strip()
    if choice == "1":
        return ["live"]
    if choice == "2":
        ds = input("Dataset (mot17 / lasot): ").strip().lower()
        data = input("Path to the dataset folder: ").strip().strip("'\"")
        return ["evaluate", "--dataset", ds, "--data", data]
    if choice == "3":
        return ["plot", input("Results folder: ").strip().strip("'\"")]
    raise SystemExit("Nothing chosen.")


def main(argv=None):
    ap = build_parser()
    argv = sys.argv[1:] if argv is None else argv
    args = ap.parse_args(argv or menu_argv())
    if hasattr(args, "weights"):
        args.weights = resolve_weights(args.weights)

    if args.cmd == "live":
        from .live import run_live
        run_live(args)
    elif args.cmd == "record":
        from .live import run_record
        run_record(args)
    elif args.cmd == "evaluate":
        do_evaluate(args)
    elif args.cmd == "plot":
        do_plot(args)
    else:
        ap.print_help()
