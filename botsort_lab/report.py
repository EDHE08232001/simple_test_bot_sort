"""Console tables, CSV export and the optional diagrams."""
import csv

MOT_COLUMNS = ["HOTA", "DetA", "AssA", "MOTA", "MOTP", "IDF1", "IDSW", "FP", "FN", "MT", "ML"]
SOT_COLUMNS = ["Success_AUC", "Precision@20", "NormPrecision_AUC", "Lost%"]


def _columns(results):
    return MOT_COLUMNS if results["kind"] == "mot" else SOT_COLUMNS


def _fmt(v):
    return f"{v:.1f}" if isinstance(v, float) else str(v)


def _table(header, rows):
    widths = [max(len(str(r[i])) for r in [header] + rows) for i in range(len(header))]
    line = lambda r: "  ".join(str(c).ljust(w) if i == 0 else str(c).rjust(w) for i, (c, w) in enumerate(zip(r, widths)))
    print(line(header))
    print("  ".join("-" * w for w in widths))
    for r in rows:
        print(line(r))


def print_report(results):
    cols = _columns(results)
    print("\n" + "=" * 20 + f" {results['dataset'].upper()} results " + "=" * 20)
    print("All metrics in %, except counts (IDSW/FP/FN/MT/ML) and FPS.\n")
    print("Overall:")
    _table(["arm"] + cols + ["FPS"],
           [[a["label"]] + [_fmt(a["combined"]["metrics"][c]) for c in cols] + [f"{a['combined']['fps']:.1f}"]
            for a in results["arms"].values()])
    for arm in results["arms"].values():
        print(f"\nPer sequence - {arm['label']}:")
        _table(["sequence"] + cols + ["FPS"],
               [[name] + [_fmt(e["metrics"][c]) for c in cols] + [f"{e['fps']:.1f}"]
                for name, e in arm["sequences"].items()])
    if results["kind"] == "mot":
        print("\nTip: compare HOTA / IDF1 / IDSW between arms. Fewer ID switches and a higher AssA "
              "mean the configuration keeps identities better.")
    else:
        print("\nTip: Success_AUC and Precision@20 drop when BoT-SORT loses the target or changes its ID; "
              "Lost% is the share of scored frames with no box for the target.")


def save_csv(results, out_dir):
    cols = _columns(results)
    with open(out_dir / "summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arm", "sequence"] + cols + ["frames", "fps"])
        for key, arm in results["arms"].items():
            for name, e in list(arm["sequences"].items()) + [("ALL", arm["combined"])]:
                w.writerow([key, name] + [e["metrics"][c] for c in cols] + [e["frames"], round(e["fps"], 2)])


# ----------------------------------------------------------------------------- diagrams

def _grouped_bars(ax, labels, series, title, ylabel=None):
    """series: {arm label: [values per label]}"""
    n = len(series)
    width = 0.8 / n
    for i, (name, vals) in enumerate(series.items()):
        xs = [x + (i - (n - 1) / 2) * width for x in range(len(labels))]
        bars = ax.bar(xs, vals, width, label=name, color=f"C{i}")
        if len(labels) <= 6:
            ax.bar_label(bars, fmt="%.1f", fontsize=7)
    ax.set_xticks(range(len(labels)), labels, rotation=30 if max(map(len, labels)) > 8 else 0, ha="right" if max(map(len, labels)) > 8 else "center", fontsize=8)
    ax.set_title(title)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.3)


def _mot_figures(results, plt):
    arms = results["arms"].values()
    figs = {}

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.5), gridspec_kw={"width_ratios": [3, 1]})
    keys = ["HOTA", "DetA", "AssA", "MOTA", "IDF1"]
    _grouped_bars(a1, keys, {a["label"]: [a["combined"]["metrics"][k] for k in keys] for a in arms}, "Overall accuracy (higher is better)", "%")
    a1.legend()
    _grouped_bars(a2, ["FPS"], {a["label"]: [a["combined"]["fps"]] for a in arms}, "Speed")
    fig.tight_layout()
    figs["summary"] = fig

    names = list(next(iter(arms))["sequences"])
    fig, axes = plt.subplots(3, 1, figsize=(max(8, 1.3 * len(names)), 10), sharex=True)
    for ax, key, title in zip(axes, ["HOTA", "IDF1", "IDSW"], ["HOTA per sequence (%)", "IDF1 per sequence (%)", "ID switches per sequence (lower is better)"]):
        _grouped_bars(ax, names, {a["label"]: [a["sequences"][n]["metrics"][key] for n in names] for a in arms}, title)
    axes[0].legend()
    fig.tight_layout()
    figs["per_sequence"] = fig

    fig, axes = plt.subplots(1, 3, figsize=(11, 4))
    for ax, key in zip(axes, ["FP", "FN", "IDSW"]):
        _grouped_bars(ax, [key], {a["label"]: [a["combined"]["metrics"][key]] for a in arms}, f"{key} (lower is better)")
    axes[0].legend()
    fig.suptitle("Where the errors come from (false positives / misses / identity switches)")
    fig.tight_layout()
    figs["errors"] = fig

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    for ax, key in zip(axes, ["HOTA", "DetA", "AssA"]):
        for i, a in enumerate(arms):
            c = a["combined"]["metrics"]["curves"]
            ax.plot(c["alpha"], c[key], color=f"C{i}", label=a["label"])
        ax.set_title(f"{key} vs localization threshold")
        ax.set_xlabel("IoU threshold alpha")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("%")
    axes[0].legend()
    fig.tight_layout()
    figs["hota_curves"] = fig
    return figs


def _sot_figures(results, plt):
    arms = results["arms"].values()
    figs = {}

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    specs = [("success", "success_th", "Success plot", "IoU threshold", "Success_AUC", "AUC"),
             ("precision", "precision_th", "Precision plot", "Center error threshold (px)", "Precision@20", "P@20"),
             ("norm_precision", "norm_precision_th", "Normalized precision plot", "Normalized center error", "NormPrecision_AUC", "AUC")]
    for ax, (ck, tk, title, xlabel, mk, short) in zip(axes, specs):
        for i, a in enumerate(arms):
            m = a["combined"]["metrics"]
            ax.plot(m["curves"][tk], [v * 100 for v in m["curves"][ck]], color=f"C{i}", label=f"{a['label']} [{short}: {m[mk]:.1f}]")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("%")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    figs["curves"] = fig

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.5), gridspec_kw={"width_ratios": [3, 1]})
    keys = ["Success_AUC", "Precision@20", "NormPrecision_AUC", "Lost%"]
    _grouped_bars(a1, keys, {a["label"]: [a["combined"]["metrics"][k] for k in keys] for a in arms}, "Overall (Lost% is lower-is-better)", "%")
    a1.legend()
    _grouped_bars(a2, ["FPS"], {a["label"]: [a["combined"]["fps"]] for a in arms}, "Speed")
    fig.tight_layout()
    figs["summary"] = fig

    names = list(next(iter(arms))["sequences"])
    fig, ax = plt.subplots(figsize=(max(8, 0.8 * len(names)), 4.5))
    _grouped_bars(ax, names, {a["label"]: [a["sequences"][n]["metrics"]["Success_AUC"] for n in names] for a in arms}, "Success AUC per sequence (%)")
    ax.legend()
    fig.tight_layout()
    figs["per_sequence"] = fig
    return figs


def visualize(results, out_dir, show=True):
    """Save PNGs next to the results and (optionally) open them in windows."""
    import matplotlib
    import matplotlib.pyplot as plt

    if not show:
        matplotlib.use("Agg")
    figs = (_mot_figures if results["kind"] == "mot" else _sot_figures)(results, plt)
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(exist_ok=True)
    for name, fig in figs.items():
        fig.savefig(plot_dir / f"{name}.png", dpi=140)
    print(f"Diagrams saved to {plot_dir}")
    if show:
        plt.show()
