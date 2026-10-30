"""Multi-object tracking metrics: CLEAR-MOT (MOTA/MOTP/IDSW/MT/ML), IDF1 and HOTA.

Follows the MOTChallenge / TrackEval definitions. Everything is computed from per-sequence raw
counts that can be summed, so a whole dataset is scored by combining sequences' counts rather than
averaging percentages. Values are reported in percent.
"""
import lap
import numpy as np

THR = 0.5  # IoU threshold for CLEAR-MOT and IDF1
ALPHAS = np.arange(0.05, 0.99, 0.05)  # HOTA localization thresholds (19 values)
COUNT_KEYS = ["num_gt", "num_pred", "TP", "FP", "FN", "IDSW", "sum_iou", "Frag", "MT", "ML",
              "gt_tracks", "pred_tracks", "IDTP"]
ARRAY_KEYS = ["HOTA_TP", "HOTA_FN", "HOTA_FP"]
WEIGHTED_KEYS = ["AssA", "AssRe", "AssPr"]  # averaged with HOTA_TP as weights when combining


def iou_matrix(a, b):
    """Pairwise IoU of two xyxy box arrays."""
    a, b = np.asarray(a, float).reshape(-1, 4), np.asarray(b, float).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(br - tl, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


def match(sim, thr):
    """Max-IoU one-to-one assignment, keeping only pairs with sim >= thr."""
    if sim.size == 0:
        return []
    _, x, _ = lap.lapjv(1 - sim, extend_cost=True, cost_limit=1 - thr + 1e-9)
    return [(i, int(j)) for i, j in enumerate(x) if 0 <= j < sim.shape[1]]


def max_assignment_sum(pot):
    if pot.size == 0:
        return 0.0
    n = max(pot.shape)
    sq = np.zeros((n, n))
    sq[: pot.shape[0], : pot.shape[1]] = pot
    _, x, _ = lap.lapjv(sq.max() - sq)
    return float(sq[np.arange(n), x].sum())


def _prepare(gt_frames, preds):
    """Drop predictions that sit on distractor regions, remap ids to 0..n-1, precompute IoUs."""
    g_map, p_map, frames = {}, {}, []
    for (g_ids, g_box, ign), (p_ids, p_box) in zip(gt_frames, preds):
        p_ids, p_box = np.asarray(p_ids), np.asarray(p_box, float).reshape(-1, 4)
        if len(p_ids) and len(ign):
            sim = iou_matrix(p_box, np.concatenate([g_box, ign]))
            drop = {i for i, j in match(sim, THR) if j >= len(g_box)}
            if drop:
                keep = np.array([i not in drop for i in range(len(p_ids))])
                p_ids, p_box = p_ids[keep], p_box[keep]
        g_idx = np.array([g_map.setdefault(int(i), len(g_map)) for i in g_ids], dtype=int)
        p_idx = np.array([p_map.setdefault(int(i), len(p_map)) for i in p_ids], dtype=int)
        frames.append((g_idx, p_idx, iou_matrix(g_box, p_box)))
    return frames, len(g_map), len(p_map)


def _clear(frames, n_g):
    c = dict(TP=0, FP=0, FN=0, IDSW=0, sum_iou=0.0, Frag=0)
    last_pid, prev_match, state = {}, {}, {}
    present, tracked = np.zeros(n_g), np.zeros(n_g)
    for g_idx, p_idx, sim in frames:
        matches, ug, up = [], set(range(len(g_idx))), set(range(len(p_idx)))
        p_pos = {int(p): j for j, p in enumerate(p_idx)}
        for gi, g in enumerate(g_idx):  # keep last frame's pairing while it still overlaps
            pj = p_pos.get(prev_match.get(int(g)))
            if pj is not None and sim[gi, pj] >= THR:
                matches.append((gi, pj))
                ug.discard(gi)
                up.discard(pj)
        if ug and up:
            ul, vl = sorted(ug), sorted(up)
            matches += [(ul[a], vl[b]) for a, b in match(sim[np.ix_(ul, vl)], THR)]

        matched_g = {gi for gi, _ in matches}
        prev_match = {}
        for gi, pj in matches:
            g, p = int(g_idx[gi]), int(p_idx[pj])
            if g in last_pid and last_pid[g] != p:
                c["IDSW"] += 1
            if state.get(g) == "lost":
                c["Frag"] += 1
            last_pid[g], prev_match[g], state[g] = p, p, "tracked"
            c["sum_iou"] += sim[gi, pj]
        for gi, g in enumerate(g_idx):
            present[g] += 1
            if gi in matched_g:
                tracked[g] += 1
            elif state.get(int(g)) == "tracked":
                state[int(g)] = "lost"
        c["TP"] += len(matches)
        c["FN"] += len(g_idx) - len(matches)
        c["FP"] += len(p_idx) - len(matches)

    ratio = tracked[present > 0] / present[present > 0]
    c["MT"], c["ML"] = int((ratio >= 0.8).sum()), int((ratio <= 0.2).sum())
    return c


def _idf1(frames, n_g, n_p):
    pot = np.zeros((n_g, n_p))
    for g_idx, p_idx, sim in frames:
        ii, jj = np.nonzero(sim >= THR)
        pot[g_idx[ii], p_idx[jj]] += 1
    return max_assignment_sum(pot)


def _hota(frames, n_g, n_p):
    g_cnt, p_cnt, pmc = np.zeros(n_g), np.zeros(n_p), np.zeros((n_g, n_p))
    for g_idx, p_idx, sim in frames:
        g_cnt[g_idx] += 1
        p_cnt[p_idx] += 1
        pmc[np.ix_(g_idx, p_idx)] += sim
    align = pmc / np.maximum(g_cnt[:, None] + p_cnt[None, :] - pmc, 1e-9)

    A = len(ALPHAS)
    tp, fn, fp = np.zeros(A), np.zeros(A), np.zeros(A)
    counts = np.zeros((A, n_g, n_p))
    for g_idx, p_idx, sim in frames:
        if len(g_idx) == 0 or len(p_idx) == 0:
            fn += len(g_idx)
            fp += len(p_idx)
            continue
        _, x, _ = lap.lapjv(1 - align[np.ix_(g_idx, p_idx)] * sim, extend_cost=True)
        rows = np.array([i for i, j in enumerate(x) if 0 <= j < len(p_idx)], dtype=int)
        cols = x[rows]
        s = sim[rows, cols]
        for a, alpha in enumerate(ALPHAS):
            ok = s >= alpha - 1e-9
            k = int(ok.sum())
            tp[a] += k
            fn[a] += len(g_idx) - k
            fp[a] += len(p_idx) - k
            counts[a][g_idx[rows[ok]], p_idx[cols[ok]]] += 1

    g3, p3 = g_cnt[None, :, None], p_cnt[None, None, :]
    denom_tp = np.maximum(1, tp)
    ass_a = (counts * counts / np.maximum(1, g3 + p3 - counts)).sum((1, 2)) / denom_tp
    ass_re = (counts * counts / np.maximum(1, g3)).sum((1, 2)) / denom_tp
    ass_pr = (counts * counts / np.maximum(1, p3)).sum((1, 2)) / denom_tp
    return dict(HOTA_TP=tp, HOTA_FN=fn, HOTA_FP=fp, AssA=ass_a, AssRe=ass_re, AssPr=ass_pr)


def evaluate_sequence(gt_frames, preds):
    """gt_frames: [(ids, xyxy, ignore_xyxy)] per frame. preds: [(ids, xyxy)] per frame."""
    frames, n_g, n_p = _prepare(gt_frames, preds)
    raw = _clear(frames, n_g)
    raw["num_gt"] = sum(len(g) for g, _, _ in frames)
    raw["num_pred"] = sum(len(p) for _, p, _ in frames)
    raw["gt_tracks"], raw["pred_tracks"] = n_g, n_p
    raw["IDTP"] = _idf1(frames, n_g, n_p)
    raw.update(_hota(frames, n_g, n_p))
    return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in raw.items()}


def combine(raws):
    """Sum per-sequence raw counts into dataset-level raw counts."""
    out = {k: sum(r[k] for r in raws) for k in COUNT_KEYS}
    for k in ARRAY_KEYS:
        out[k] = np.sum([r[k] for r in raws], axis=0).tolist()
    total_tp = np.maximum(1, out["HOTA_TP"])
    for k in WEIGHTED_KEYS:
        out[k] = (np.sum([np.array(r[k]) * np.array(r["HOTA_TP"]) for r in raws], axis=0) / total_tp).tolist()
    return out


def finalize(raw):
    """Raw counts -> the metrics people read (percentages)."""
    tp, fn, fp = (np.array(raw[k], float) for k in ARRAY_KEYS)
    det_a = tp / np.maximum(1, tp + fn + fp)
    ass_a = np.array(raw["AssA"])
    hota = np.sqrt(det_a * ass_a)
    idtp, n_gt, n_pred = raw["IDTP"], raw["num_gt"], raw["num_pred"]
    pct = lambda v: float(v) * 100
    return {
        "HOTA": pct(hota.mean()),
        "DetA": pct(det_a.mean()),
        "AssA": pct(ass_a.mean()),
        "DetRe": pct((tp / np.maximum(1, tp + fn)).mean()),
        "DetPr": pct((tp / np.maximum(1, tp + fp)).mean()),
        "AssRe": pct(np.mean(raw["AssRe"])),
        "AssPr": pct(np.mean(raw["AssPr"])),
        "MOTA": pct(1 - (raw["FN"] + raw["FP"] + raw["IDSW"]) / max(n_gt, 1)),
        "MOTP": pct(raw["sum_iou"] / max(raw["TP"], 1)),
        "IDF1": pct(idtp / max(idtp + 0.5 * ((n_gt - idtp) + (n_pred - idtp)), 1e-9)),
        "IDP": pct(idtp / max(n_pred, 1)),
        "IDR": pct(idtp / max(n_gt, 1)),
        "IDSW": raw["IDSW"], "FP": raw["FP"], "FN": raw["FN"], "Frag": raw["Frag"],
        "MT": raw["MT"], "ML": raw["ML"],
        "GT_tracks": raw["gt_tracks"], "pred_tracks": raw["pred_tracks"],
        "GT_dets": n_gt, "pred_dets": n_pred,
        "curves": {"alpha": ALPHAS.tolist(), "HOTA": (hota * 100).tolist(),
                   "DetA": (det_a * 100).tolist(), "AssA": (ass_a * 100).tolist()},
    }
