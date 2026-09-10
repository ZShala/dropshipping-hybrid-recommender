"""Effect sizes for the headline comparisons.

A p-value at n = 25,000 says a difference is real, not that it is large. For
each comparison this reports Cohen's dz, the matched-pairs rank-biserial
correlation, a bootstrap CI on the mean difference, win/tie/loss counts, and
the hit-ratio difference expressed per 1,000 recommendation lists.

Uses the same pooling and zero_method as make_tables.py, so p-values match.

    python -m experiments.analyse_effect_sizes

Writes results/effect_sizes.json.
"""
import json
from pathlib import Path

import numpy as np
from scipy import stats

RESULTS_DIR = Path(__file__).resolve().parent / "results"
N_BOOT = 10_000
BOOT_SEED = 0

# (reference, competitor) -- reference first, so a positive delta favours it
COMPARISONS = [
    ("Hybrid 10/70/20", "ItemKNN"),
    ("Hybrid 10/70/20", "Hybrid 50/30/20"),
    ("Hybrid 10/70/20", "Trend (popularity)"),
    ("Hybrid 10/70/20", "Content-based"),
    ("Hybrid 10/70/20", "PureSVD"),
    ("Hybrid 10/70/20", "BPR"),
    ("Hybrid 10/70/20", "Hybrid-proxy 50/30/20"),
    ("ItemKNN", "CF proxy (fixed 0.8)"),
    ("ItemKNN", "PureSVD"),
    ("ItemKNN", "BPR"),
    ("ItemKNN", "Trend (popularity)"),
    ("ItemKNN", "Content-based"),
    ("Trend (popularity)", "Content-based"),
]


def load_pooled():
    """Concatenate per-user metric arrays across seeds (as make_tables.py does)."""
    pooled = {}
    for f in sorted(RESULTS_DIR.glob("per_user_seed*.npz")):
        with np.load(f, allow_pickle=True) as npz:
            for key in npz.files:
                pooled.setdefault(key, []).append(npz[key])
    return {k: np.concatenate(v) for k, v in pooled.items()}


def rank_biserial(diff):
    """Matched-pairs rank-biserial correlation, (W+ - W-) / (W+ + W-).

    Zeros dropped, matching zero_method="wilcox"; unaffected by ties.
    """
    nz = diff[diff != 0]
    if nz.size == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(nz))
    w_plus = ranks[nz > 0].sum()
    w_minus = ranks[nz < 0].sum()
    return float((w_plus - w_minus) / (w_plus + w_minus))


def boot_ci(diff, n_boot=N_BOOT, seed=BOOT_SEED):
    """Percentile bootstrap CI for the mean paired difference."""
    rng = np.random.default_rng(seed)
    n = diff.size
    means = np.empty(n_boot)
    for b in range(n_boot):
        means[b] = diff[rng.integers(0, n, n)].mean()
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def cohens_dz(diff):
    sd = diff.std(ddof=1)
    return float(diff.mean() / sd) if sd > 0 else 0.0


def compare(pooled, ref, comp, metric="NDCG@10"):
    a = pooled[f"{ref}|{metric}"]
    b = pooled[f"{comp}|{metric}"]
    n = min(a.size, b.size)
    a, b = a[:n], b[:n]
    diff = a - b

    wins = int((diff > 0).sum())
    losses = int((diff < 0).sum())
    ties = int((diff == 0).sum())
    changed = wins + losses

    try:
        w, p = stats.wilcoxon(a, b, zero_method="wilcox")
        w, p = float(w), float(p)
    except ValueError:                              # all differences zero
        w, p = float("nan"), 1.0

    lo, hi = boot_ci(diff)
    res = {
        "reference": ref,
        "competitor": comp,
        "metric": metric,
        "n_users": int(n),
        "mean_ref": float(a.mean()),
        "mean_comp": float(b.mean()),
        "mean_delta": float(diff.mean()),
        "delta_ci95": [lo, hi],
        "relative_gain_pct": float(100 * diff.mean() / b.mean()) if b.mean() > 0 else float("nan"),
        "cohens_dz": cohens_dz(diff),
        "rank_biserial_r": rank_biserial(diff),
        "wilcoxon_W": w,
        "wilcoxon_p": p,
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "changed": changed,
        "changed_pct": round(100 * changed / n, 2),
        "win_loss_ratio": round(wins / losses, 2) if losses else float("inf"),
    }
    # restricted to users the two systems rank differently
    if changed:
        d_ch = diff[diff != 0]
        res["changed_mean_delta"] = float(d_ch.mean())
        res["changed_cohens_dz"] = cohens_dz(d_ch)
    else:
        res["changed_mean_delta"] = 0.0
        res["changed_cohens_dz"] = 0.0
    return res


def operational(pooled, ref, comp):
    """Hit-ratio delta expressed per 1,000 recommendation lists."""
    a = pooled[f"{ref}|HR@10"]
    b = pooled[f"{comp}|HR@10"]
    n = min(a.size, b.size)
    a, b = a[:n], b[:n]
    delta = float(a.mean() - b.mean())
    return {
        "hr_ref": float(a.mean()),
        "hr_comp": float(b.mean()),
        "hr_delta": delta,
        "extra_users_with_a_hit_per_1000": round(1000 * delta, 2),
        "users_needed_for_one_extra_hit": round(1.0 / delta, 1) if delta > 0 else None,
    }


def label(d):
    """Cohen's conventional bands."""
    m = abs(d)
    if m < 0.01:
        return "negligible"
    if m < 0.2:
        return "very small"
    if m < 0.5:
        return "small"
    if m < 0.8:
        return "medium"
    return "large"


def main():
    pooled = load_pooled()
    out = {"n_boot": N_BOOT, "boot_seed": BOOT_SEED, "comparisons": []}

    for ref, comp in COMPARISONS:
        if f"{ref}|NDCG@10" not in pooled or f"{comp}|NDCG@10" not in pooled:
            print(f"  !! skipped (missing): {ref} vs {comp}")
            continue
        res = compare(pooled, ref, comp)
        res["dz_label"] = label(res["cohens_dz"])
        res["operational"] = operational(pooled, ref, comp)
        out["comparisons"].append(res)

        lo, hi = res["delta_ci95"]
        print(f"\n{ref}  vs  {comp}   (n = {res['n_users']:,})")
        print(f"  NDCG@10      {res['mean_ref']:.4f} vs {res['mean_comp']:.4f}"
              f"   delta = {res['mean_delta']:+.5f}  [{lo:+.5f}, {hi:+.5f}]"
              f"   ({res['relative_gain_pct']:+.1f}%)")
        print(f"  effect       d_z = {res['cohens_dz']:+.4f} ({res['dz_label']})"
              f"   rank-biserial r = {res['rank_biserial_r']:+.3f}")
        print(f"  Wilcoxon     W = {res['wilcoxon_W']:.0f}   p = {res['wilcoxon_p']:.3g}")
        print(f"  users        {res['wins']:,} win / {res['losses']:,} lose /"
              f" {res['ties']:,} tie   -> only {res['changed_pct']}% ranked differently")
        print(f"  among those  delta = {res['changed_mean_delta']:+.5f}"
              f"   d_z = {res['changed_cohens_dz']:+.3f}")
        op = res["operational"]
        print(f"  operational  HR@10 {op['hr_ref']:.4f} vs {op['hr_comp']:.4f}"
              f"  -> {op['extra_users_with_a_hit_per_1000']:+.2f} users per 1,000"
              f" gain a relevant item in the top 10")
        if op["users_needed_for_one_extra_hit"]:
            print(f"               = 1 extra served user per"
                  f" {op['users_needed_for_one_extra_hit']:,.0f} lists")

    path = RESULTS_DIR / "effect_sizes.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
