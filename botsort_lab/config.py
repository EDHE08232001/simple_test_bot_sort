"""Tracker configs (the A0 / A1 arms) and small path / device helpers."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent

# A0 = GMC off, A1 = GMC on. Everything else in the tracker config is identical.
ARMS = {
    "nogmc": {"label": "A0 (GMC off)", "gmc_method": "none"},
    "gmc": {"label": "A1 (GMC on)", "gmc_method": "sparseOptFlow"},
}

TRACKER_DEFAULTS = dict(
    tracker_type="botsort",
    track_high_thresh=0.25,
    track_low_thresh=0.1,
    new_track_thresh=0.25,
    track_buffer=30,
    match_thresh=0.8,
    fuse_score=True,
    proximity_thresh=0.5,
    appearance_thresh=0.8,
    with_reid=False,
)


def default_device() -> str:
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_weights(weights: str) -> str:
    """Prefer a weights file next to the project; otherwise let Ultralytics download it."""
    if Path(weights).exists():
        return weights
    local = ROOT / weights
    return str(local) if local.exists() else weights


def make_cfgs(arms=None, directory: Path = ROOT / "tracker_cfgs") -> dict:
    """Write one tracker yaml per arm and return {arm_key: path}."""
    directory.mkdir(exist_ok=True)
    cfgs = {}
    for key in arms or ARMS:
        path = directory / f"botsort_{key}.yaml"
        path.write_text(yaml.safe_dump({**TRACKER_DEFAULTS, "gmc_method": ARMS[key]["gmc_method"]}))
        cfgs[key] = path
    return cfgs
