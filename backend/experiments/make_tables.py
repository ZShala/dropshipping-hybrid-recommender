"""Aggregate multi-seed results into publication-ready markdown tables.

Usage (from backend/):  python -m experiments.make_tables
Writes experiments/results/tables.md and sensitivity_grid.csv.
"""
import json
import re
from pathlib import Path

import numpy as np
from scipy import stats

RESULTS_DIR = Path(__file__).resolve().parent / "results"

ACC = ["Precision@10", "Recall@10", "F1@10", "HR@10", "NDCG@10", "MAP@10"]
BEYOND = ["ILD", "CategoryDiversity", "Novelty", "Coverage"]

BASELINES = [
    "Trend (popularity)",
    "Content-based",
    "CF proxy (fixed 0.8)",
    "ItemKNN",
    "PureSVD",
    "BPR",
]
PAPER_HYBRIDS = [
    "Hybrid-proxy 50/30/20",
    "Hybrid 33/33/34",
    "Hybrid 40/35/25",
    "Hybrid 50/30/20",
    "Hybrid 30/50/20",
    "Hybrid 20/40/40",
]
ABLATION = ["Content-based", "ItemKNN", "Trend (popularity)", "C+CF", "C+T", "CF+T",
            "Hybrid 50/30/20"]


def load_runs():
    summaries = []
    for f in sorted(RESULTS_DIR.glob("summary_seed*.json")):
        summaries.append(json.load(open(f)))
    per_user = {}
    for f in sorted(RESULTS_DIR.glob("per_user_seed*.npz")):
        npz = np.load(f)
        for key in npz.files:
            per_user.setdefault(key, []).append(npz[key])
    per_user = {k: np.concatenate(v) for k, v in per_user.items()}
    return summaries, per_user


def mean_std(summaries, system, metric):
    vals = [s["systems"][system][metric] for s in summaries if system in s["systems"]]
    return np.mean(vals), np.std(vals)


def fmt(m, s, pct=False, digits=4):
    if pct:
        return f"{100*m:.2f} ± {100*s:.2f}"
    return f"{m:.{digits}f} ± {s:.{digits}f}"


def table(summaries, systems, metrics, header):
    lines = [f"| System | " + " | ".join(metrics) + " |",
             "|" + "---|" * (len(metrics) + 1)]
    for sys_name in systems:
        if sys_name not in summaries[0]["systems"]:
            continue
        cells = []
        for met in metrics:
            m, s = mean_std(summaries, sys_name, met)
            cells.append(fmt(m, s))
        lines.append(f"| {sys_name} | " + " | ".join(cells) + " |")
    return f"\n### {header}\n\n" + "\n".join(lines) + "\n"


def wilcoxon_table(per_user, reference, competitors, metric="NDCG@10"):
    ref = per_user[f"{reference}|{metric}"]
    rows = [f"| Comparison ({metric}) | mean Δ | Wilcoxon W | p-value | p (Holm) | sig |",
            "|---|---|---|---|---|---|"]
    raw = []
    for comp in competitors:
        key = f"{comp}|{metric}"
        if key not in per_user:
            continue
        arr = per_user[key]
        n = min(len(ref), len(arr))
        a, b = ref[:n], arr[:n]
        delta = float(np.mean(a - b))
        try:
            w, p = stats.wilcoxon(a, b, zero_method="wilcox")
        except ValueError:
            w, p = np.nan, 1.0
        raw.append((comp, delta, w, p))
    # Holm-Bonferroni step-down correction for m comparisons: sort p-values
    # ascending, multiply the smallest by m, the next by m-1, and so on.
    # The running max enforces monotonicity (an adjusted p-value can never
    # be smaller than that of a stronger comparison earlier in the order).
    order = np.argsort([r[3] for r in raw])
    holm = {}
    m = len(raw)
    running_max = 0.0
    for rank, idx in enumerate(order):
        adj = min(1.0, (m - rank) * raw[idx][3])
        running_max = max(running_max, adj)
        holm[raw[idx][0]] = running_max
    for comp, delta, w, p in raw:
        ph = holm[comp]
        sig = "***" if ph < 0.001 else "**" if ph < 0.01 else "*" if ph < 0.05 else "ns"
        rows.append(f"| {reference} vs {comp} | {delta:+.4f} | {w:.0f} | {p:.2e} | {ph:.2e} | {sig} |")
    return "\n".join(rows) + "\n"


def main():
    summaries, per_user = load_runs()
    n_seeds = len(summaries)
    stats_json = json.load(open(RESULTS_DIR / "dataset_stats.json"))
    all_systems = list(summaries[0]["systems"].keys())

    # best hybrid on the sensitivity grid by mean NDCG@10
    hybrid_names = [s for s in all_systems if re.match(r"Hybrid \d+/\d+/\d+", s)]
    best_hybrid = max(hybrid_names, key=lambda s: mean_std(summaries, s, "NDCG@10")[0])

    out = ["# Offline Experiment Results",
           f"\n{n_seeds} independent runs; each run evaluates a random sample of "
           f"{summaries[0]['n_users']} users (per-user temporal 80/20 split, "
           f"full-catalogue ranking, k=10). Values are mean ± std across runs.\n"]

    out.append("\n### Dataset and split statistics\n")
    out.append("| Statistic | Value |\n|---|---|")
    for k, v in stats_json.items():
        out.append(f"| {k} | {v} |")
    out.append("")

    out.append(table(summaries, BASELINES + ["Hybrid-proxy 50/30/20",
                                             "Hybrid 50/30/20", best_hybrid],
                     ACC, "Baseline comparison — ranking accuracy"))
    out.append(table(summaries, BASELINES + ["Hybrid-proxy 50/30/20",
                                             "Hybrid 50/30/20", best_hybrid],
                     BEYOND, "Baseline comparison — beyond-accuracy"))
    out.append(table(summaries, ABLATION + ([best_hybrid] if best_hybrid not in ABLATION else []),
                     ACC, "Ablation study (C = content, CF = ItemKNN, T = trend)"))
    out.append(table(summaries, PAPER_HYBRIDS, ACC,
                     "Weight configurations from the original paper (real CF)"))

    # sensitivity: top 10 + worst 3
    ranked = sorted(hybrid_names, key=lambda s: -mean_std(summaries, s, "NDCG@10")[0])
    out.append(table(summaries, ranked[:10], ["NDCG@10", "F1@10", "HR@10"],
                     "Weight sensitivity — top 10 of 66 grid points by NDCG@10"))

    out.append("\n### Statistical significance (pooled per-user, Wilcoxon signed-rank, Holm-corrected)\n")
    competitors = BASELINES + ["Hybrid-proxy 50/30/20", "Hybrid 50/30/20"]
    out.append(wilcoxon_table(per_user, best_hybrid,
                              [c for c in competitors if c != best_hybrid]))
    out.append("\nKey pairwise comparisons:\n")
    out.append(wilcoxon_table(per_user, "ItemKNN", ["CF proxy (fixed 0.8)"]))
    out.append(wilcoxon_table(per_user, "Hybrid 50/30/20", ["Hybrid-proxy 50/30/20"]))

    out.append("\n### Model fit times (seconds)\n")
    out.append("| Model | Fit time (s) |\n|---|---|")
    for k, v in summaries[-1]["fit_times_sec"].items():
        out.append(f"| {k} | {v:.1f} |")
    out.append("")

    # full sensitivity grid to CSV
    with open(RESULTS_DIR / "sensitivity_grid.csv", "w") as f:
        f.write("content,cf,trend,ndcg10_mean,ndcg10_std,f1_mean,hr_mean\n")
        for name in hybrid_names:
            c, cf, t = re.match(r"Hybrid (\d+)/(\d+)/(\d+)", name).groups()
            nm, ns = mean_std(summaries, name, "NDCG@10")
            fm, _ = mean_std(summaries, name, "F1@10")
            hm, _ = mean_std(summaries, name, "HR@10")
            f.write(f"{c},{cf},{t},{nm:.5f},{ns:.5f},{fm:.5f},{hm:.5f}\n")

    (RESULTS_DIR / "tables.md").write_text("\n".join(out), encoding="utf-8")
    print("Wrote", RESULTS_DIR / "tables.md")
    print("Best hybrid:", best_hybrid)


if __name__ == "__main__":
    main()
