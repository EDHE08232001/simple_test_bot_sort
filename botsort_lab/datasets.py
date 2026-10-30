"""Loaders for labeled sequences: MOT17 (multi-object) and LaSOT (single-object)."""
import configparser
from dataclasses import dataclass
from pathlib import Path

import numpy as np

IMG_EXT = {".jpg", ".jpeg", ".png"}
MOT_PEDESTRIAN = 1
MOT_DISTRACTORS = (2, 7, 8, 12)  # person on vehicle, static person, distractor, reflection


@dataclass
class Sequence:
    name: str
    kind: str           # "mot" or "sot"
    frame_paths: list
    fps: float
    gt: object          # mot: list of (ids, xyxy boxes, ignore xyxy boxes) per frame; sot: (N,4) xywh
    valid: np.ndarray = None  # sot only: frames where the target is visible


def _frames_in(directory: Path):
    return sorted(p for p in directory.iterdir() if p.suffix.lower() in IMG_EXT)


def _xywh_to_xyxy(b):
    b = np.asarray(b, dtype=float).reshape(-1, 4)
    return np.concatenate([b[:, :2], b[:, :2] + b[:, 2:]], axis=1)


# ----------------------------------------------------------------------------- MOT17

def _load_mot_sequence(seq_dir: Path) -> Sequence:
    info = configparser.ConfigParser()
    info.read(seq_dir / "seqinfo.ini")
    fps = info.getfloat("Sequence", "frameRate", fallback=30.0)
    paths = _frames_in(seq_dir / info.get("Sequence", "imDir", fallback="img1"))

    gt = np.loadtxt(seq_dir / "gt" / "gt.txt", delimiter=",", ndmin=2)
    frame_nums, cls, flag = gt[:, 0].astype(int), gt[:, 7].astype(int), gt[:, 6]
    # Pedestrians the benchmark scores (flag 0 = "ignore this box"), and distractor regions,
    # where a tracker box that overlaps is neither a hit nor a false positive.
    is_ped = (cls == MOT_PEDESTRIAN) & (flag != 0)
    is_ign = np.isin(cls, MOT_DISTRACTORS)

    frames = []
    for p in paths:
        rows = frame_nums == int(p.stem)
        ped, ign = gt[rows & is_ped], gt[rows & is_ign]
        frames.append((ped[:, 1].astype(int), _xywh_to_xyxy(ped[:, 2:6]), _xywh_to_xyxy(ign[:, 2:6])))
    return Sequence(seq_dir.name, "mot", paths, fps, frames)


def find_mot_sequences(root, split="train", variant="FRCNN", names=None):
    """MOT17 ships each sequence three times (DPM/FRCNN/SDP) with identical GT; keep one variant."""
    root = Path(root)
    base = root / split if (root / split).is_dir() else root
    seqs = []
    for d in sorted(base.iterdir()):
        if not (d / "seqinfo.ini").exists():
            continue
        if not (d / "gt" / "gt.txt").exists():
            print(f"  skipping {d.name}: no gt/gt.txt (test split has no labels)")
            continue
        if variant and d.name.count("-") >= 2 and not d.name.endswith("-" + variant):
            continue
        if names and d.name not in names and not any(d.name.startswith(n) for n in names):
            continue
        seqs.append(_load_mot_sequence(d))
    return seqs


# ----------------------------------------------------------------------------- LaSOT

def _load_lasot_sequence(seq_dir: Path) -> Sequence:
    paths = _frames_in(seq_dir / "img")
    gt = np.loadtxt(seq_dir / "groundtruth.txt", delimiter=",", ndmin=2)
    n = min(len(paths), len(gt))
    gt = gt[:n]
    valid = (gt[:, 2] > 0) & (gt[:, 3] > 0)
    oov = seq_dir / "out_of_view.txt"
    if oov.exists():
        flags = np.loadtxt(oov, delimiter=",", ndmin=1)[:n]
        valid[: len(flags)] &= flags == 0
    return Sequence(seq_dir.name, "sot", paths[:n], 30.0, gt, valid)


def find_lasot_sequences(root, names=None):
    root = Path(root)
    gts = sorted(root.glob("*/*/groundtruth.txt")) + sorted(root.glob("*/groundtruth.txt"))
    gts += [root / "groundtruth.txt"] if (root / "groundtruth.txt").exists() else []
    seqs = []
    for gt in gts:
        if names and gt.parent.name not in names:
            continue
        if (gt.parent / "img").is_dir():
            seqs.append(_load_lasot_sequence(gt.parent))
    return seqs


def load_sequences(dataset, root, split="train", variant="FRCNN", names=None, max_seqs=None):
    if not Path(root).is_dir():
        raise SystemExit(f"Dataset folder not found: {root}")
    seqs = (find_mot_sequences(root, split, variant, names) if dataset == "mot17"
            else find_lasot_sequences(root, names))
    if not seqs:
        raise SystemExit(f"No labeled {dataset} sequences found under {root}")
    return seqs[:max_seqs] if max_seqs else seqs
