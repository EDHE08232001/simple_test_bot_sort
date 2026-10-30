"""Mode 2: run BoT-SORT (GMC off / on) over labeled sequences and score it."""
import json
import time
from datetime import datetime

import cv2
import numpy as np

from . import metrics_mot, metrics_sot
from .config import ARMS, ROOT, make_cfgs
from .datasets import load_sequences
from .tracking import BotSortTracker


def track_sequence(tracker, seq, max_frames=None):
    """Returns per-frame [(ids, xyxy, conf)] and the seconds spent in detection + tracking."""
    paths = seq.frame_paths[:max_frames] if max_frames else seq.frame_paths
    tracker.reset()
    out, elapsed = [], 0.0
    for i, path in enumerate(paths, 1):
        frame = cv2.imread(str(path))
        t0 = time.perf_counter()
        _, boxes, ids, conf = tracker.update(frame)
        elapsed += time.perf_counter() - t0
        out.append((ids, boxes, conf))
        if i % 25 == 0 or i == len(paths):
            print(f"\r    {seq.name}: {i}/{len(paths)} frames", end="", flush=True)
    print()
    return out, elapsed


def save_mot_txt(path, tracks):
    """Tracker output in MOTChallenge format, so it can also be scored with other tools."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for f, (ids, boxes, conf) in enumerate(tracks, 1):
        for i, b, c in zip(ids, boxes, conf):
            rows.append(f"{f},{i},{b[0]:.2f},{b[1]:.2f},{b[2] - b[0]:.2f},{b[3] - b[1]:.2f},{c:.3f},-1,-1,-1")
    path.write_text("\n".join(rows))


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def run_evaluation(args):
    seqs = load_sequences(args.dataset, args.data, args.split, args.mot_variant, args.seqs, args.max_seqs)
    kind = seqs[0].kind
    classes = args.classes if args.classes is not None else ([0] if kind == "mot" else [])
    arms = args.arms
    cfgs = make_cfgs(arms)
    out_dir = ROOT / "results" / f"{args.dataset}_{datetime.now():%Y%m%d_%H%M%S}"
    print(f"Evaluating {len(seqs)} {args.dataset} sequence(s) with {args.weights} on {args.device}")
    print(f"Arms: {', '.join(ARMS[a]['label'] for a in arms)}   results -> {out_dir}\n")

    tracker = BotSortTracker(args.weights, args.device, args.imgsz, classes, args.det_conf)
    tracker.warmup(cv2.imread(str(seqs[0].frame_paths[0])))

    results = {"dataset": args.dataset, "kind": kind, "weights": args.weights, "device": args.device,
               "imgsz": args.imgsz, "det_conf": args.det_conf, "max_frames": args.max_frames, "arms": {}}
    for arm in arms:
        print(f"== {ARMS[arm]['label']}")
        tracker.set_config(cfgs[arm])
        per_seq = {}
        for seq in seqs:
            tracks, elapsed = track_sequence(tracker, seq, args.max_frames)
            n = len(tracks)
            save_mot_txt(out_dir / "tracks" / arm / f"{seq.name}.txt", tracks)
            entry = {"frames": n, "time": elapsed, "fps": n / max(elapsed, 1e-9)}
            if kind == "mot":
                raw = metrics_mot.evaluate_sequence(seq.gt[:n], [(ids, b) for ids, b, _ in tracks])
                entry.update(raw=raw, metrics=metrics_mot.finalize(raw))
            else:
                preds = [(ids, b) for ids, b, _ in tracks]
                gt, valid = seq.gt[:n], seq.valid[:n]
                entry["metrics"] = metrics_sot.evaluate_sequence(gt, metrics_sot.follow_target(preds, gt, valid), valid)
            per_seq[seq.name] = entry

        frames = sum(e["frames"] for e in per_seq.values())
        secs = sum(e["time"] for e in per_seq.values())
        if kind == "mot":
            raw = metrics_mot.combine([e["raw"] for e in per_seq.values()])
            combined = {"raw": raw, "metrics": metrics_mot.finalize(raw)}
        else:
            combined = {"metrics": metrics_sot.combine([e["metrics"] for e in per_seq.values()])}
        combined.update(frames=frames, time=secs, fps=frames / max(secs, 1e-9))
        results["arms"][arm] = {"label": ARMS[arm]["label"], "sequences": per_seq, "combined": combined}

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(results, default=_json_default))
    return results, out_dir
