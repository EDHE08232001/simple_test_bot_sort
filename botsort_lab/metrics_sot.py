"""Single-object tracking metrics as used by LaSOT: success (AUC), precision and normalized precision.

BoT-SORT is a multi-object tracker, so a single target is extracted from its output: the track whose
box first overlaps the ground-truth box (IoU >= ACQUIRE_IOU) is followed *by track ID* from then on.
If BoT-SORT drops the target or switches its ID, the frames count as lost. Frames before the target is
acquired are lost too. Absent / out-of-view frames are not scored.
"""
import numpy as np

from .metrics_mot import iou_matrix

ACQUIRE_IOU = 0.3
SUCCESS_TH = np.linspace(0, 1, 21)
PRECISION_TH = np.arange(0, 51)
NORM_PRECISION_TH = np.linspace(0, 0.5, 51)


def follow_target(preds, gt_xywh, valid):
    """Pick the target track out of the multi-object output. Returns (N,4) xywh, NaN where lost."""
    out = np.full((len(preds), 4), np.nan)
    target = None
    for t, (ids, boxes) in enumerate(preds):
        if target is None and valid[t] and len(ids):
            g = gt_xywh[t]
            sim = iou_matrix([[g[0], g[1], g[0] + g[2], g[1] + g[3]]], boxes)[0]
            j = int(sim.argmax())
            if sim[j] >= ACQUIRE_IOU:
                target = ids[j]
        if target is not None:
            hit = np.nonzero(ids == target)[0]
            if len(hit):
                x1, y1, x2, y2 = boxes[hit[0]]
                out[t] = (x1, y1, x2 - x1, y2 - y1)
    return out


def evaluate_sequence(gt_xywh, pred_xywh, valid):
    gt, pr = gt_xywh[valid], pred_xywh[valid]
    found = ~np.isnan(pr[:, 0])
    pr_f = np.where(found[:, None], pr, 0.0)

    iou = np.zeros(len(gt))
    if len(gt):
        gx = np.concatenate([gt[:, :2], gt[:, :2] + gt[:, 2:]], 1)
        px = np.concatenate([pr_f[:, :2], pr_f[:, :2] + pr_f[:, 2:]], 1)
        inter = np.clip(np.minimum(gx[:, 2:], px[:, 2:]) - np.maximum(gx[:, :2], px[:, :2]), 0, None).prod(1)
        union = gt[:, 2] * gt[:, 3] + pr_f[:, 2] * pr_f[:, 3] - inter
        iou = np.where(found, inter / np.maximum(union, 1e-9), 0.0)

    d = (pr_f[:, :2] + pr_f[:, 2:] / 2) - (gt[:, :2] + gt[:, 2:] / 2)
    center_err = np.where(found, np.linalg.norm(d, axis=1), np.inf)
    norm_err = np.where(found, np.linalg.norm(d / np.maximum(gt[:, 2:], 1e-9), axis=1), np.inf)

    n = max(len(gt), 1)
    success = [float((iou > th).sum() / n) for th in SUCCESS_TH]
    precision = [float((center_err <= th).sum() / n) for th in PRECISION_TH]
    norm_precision = [float((norm_err <= th).sum() / n) for th in NORM_PRECISION_TH]
    return {
        "Success_AUC": float(np.mean(success)) * 100,
        "Precision@20": precision[20] * 100,
        "NormPrecision_AUC": float(np.mean(norm_precision)) * 100,
        "Lost%": float((~found).sum() / n) * 100,
        "scored_frames": int(len(gt)),
        "curves": {"success": success, "precision": precision, "norm_precision": norm_precision,
                   "success_th": SUCCESS_TH.tolist(), "precision_th": PRECISION_TH.tolist(),
                   "norm_precision_th": NORM_PRECISION_TH.tolist()},
    }


def combine(seq_metrics):
    """Dataset level = mean over sequences (the LaSOT protocol), curves included."""
    scalars = ["Success_AUC", "Precision@20", "NormPrecision_AUC", "Lost%"]
    out = {k: float(np.mean([m[k] for m in seq_metrics])) for k in scalars}
    out["scored_frames"] = int(sum(m["scored_frames"] for m in seq_metrics))
    out["curves"] = {k: np.mean([m["curves"][k] for m in seq_metrics], axis=0).tolist()
                     for k in ("success", "precision", "norm_precision")}
    first = seq_metrics[0]["curves"]
    out["curves"].update({k: first[k] for k in ("success_th", "precision_th", "norm_precision_th")})
    return out
