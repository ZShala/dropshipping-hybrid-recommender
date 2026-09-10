"""Model ordering under varying sparsity.

Four arms: a no-refit stratification of the published per-user arrays by user
profile length, then random thinning, a per-user cap and a per-item cap. The
last two thin one margin of the interaction matrix at a time, which is what
separates item-side density from user-side sparsity.

The test set and the evaluated user samples are identical in every condition,
so a user whose profile the thinning empties stays in the denominator.

    python -m experiments.analyse_sparsity [--seeds 3] [--users 5000]

Writes results/sparsity_sweep.json.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.decomposition import TruncatedSVD

from .data import load_dataset
from .evaluate import evaluate_systems
from .models import BPR, ContentBased, ItemKNN, Popularity, PureSVD, fit_timed

RESULTS_DIR = Path(__file__).resolve().parent / "results"

GLOBAL_LEVELS = [1.0, 0.75, 0.50, 0.25, 0.10]
USER_CAPS = [1, 2, 3, 5]
ITEM_CAPS = [5, 10, 25, 50]

# how many training positives a user has, as ordered buckets
PROFILE_BINS = [(2, 2), (3, 3), (4, 5), (6, 10), (11, 10 ** 9)]
PROFILE_LABELS = ["2", "3", "4-5", "6-10", "11+"]

SYSTEMS = [
    {"name": "Trend (popularity)", "model": "T"},
    {"name": "Content-based", "model": "C"},
    {"name": "ItemKNN", "model": "CF"},
    {"name": "PureSVD", "model": "SVD"},
    {"name": "BPR", "model": "BPR"},
    {"name": "Hybrid 10/70/20", "weights": {"C": 0.1, "CF": 0.7, "T": 0.2}},
]

STRATIFY_SYSTEMS = ["Trend (popularity)", "Content-based", "ItemKNN",
                    "PureSVD", "BPR", "Hybrid 10/70/20"]


def sampled_users(ds, seed, n_users):
    """Reproduce run_experiments.py's evaluation sample for a given seed."""
    rng = np.random.default_rng(seed)
    return rng.choice(ds.eval_users, size=min(n_users, len(ds.eval_users)),
                      replace=False)


def stratify(ds, seeds, n_users):
    """Accuracy by user profile length, from the already-published per-user arrays."""
    profile_len = np.asarray(ds.train_csr.sum(axis=1)).ravel()

    lengths, per_sys = [], {}
    for seed in range(seeds):
        f = RESULTS_DIR / f"per_user_seed{seed}.npz"
        if not f.exists():
            print(f"  (no {f.name}; skipping stratification for seed {seed})")
            continue
        users = sampled_users(ds, seed, n_users)
        with np.load(f, allow_pickle=True) as npz:
            for name in STRATIFY_SYSTEMS:
                key = f"{name}|NDCG@10"
                if key not in npz.files:
                    continue
                arr = npz[key]
                if arr.size != users.size:
                    print(f"  !! seed {seed}: {arr.size} metrics vs {users.size}"
                          f" users -- cannot align, skipping")
                    return None
                per_sys.setdefault(name, []).append(arr)
        lengths.append(profile_len[users])

    if not lengths:
        return None
    lengths = np.concatenate(lengths)
    per_sys = {k: np.concatenate(v) for k, v in per_sys.items()}

    rows = []
    for (lo, hi), label in zip(PROFILE_BINS, PROFILE_LABELS):
        sel = (lengths >= lo) & (lengths <= hi)
        if not sel.any():
            continue
        row = {"profile_length": label, "users": int(sel.sum()),
               "share_of_users_pct": round(100 * float(sel.mean()), 1)}
        for name, arr in per_sys.items():
            row[name] = float(arr[sel].mean())
        rows.append(row)
    return {"bins": rows,
            "median_profile_length": float(np.median(lengths)),
            "mean_profile_length": float(lengths.mean())}


def run_condition(spec, seeds, n_users, chunk, full_eval_users):
    """Fit and evaluate every system under one thinning specification."""
    ds = load_dataset(verbose=False, thin=spec, thin_seed=0, cache_raw=True)

    content, t_c = fit_timed(ContentBased(), ds, 0)
    knn, t_k = fit_timed(ItemKNN(k=200), ds, 0)
    pop, _ = fit_timed(Popularity(), ds, 0)
    item_emb = TruncatedSVD(n_components=64, random_state=0).fit_transform(
        content.item_vecs).astype(np.float32)

    acc = {}
    for seed in range(seeds):
        svd, _ = fit_timed(PureSVD(factors=256), ds, seed)
        bpr, _ = fit_timed(BPR(factors=128, learning_rate=0.05, iterations=300),
                           ds, seed)
        components = {"C": content, "CF": knn, "T": pop, "SVD": svd, "BPR": bpr}
        users = full_eval_users[seed]
        res = evaluate_systems(ds, users, components, SYSTEMS, item_emb,
                               chunk_size=chunk)
        for n, r in res.items():
            acc.setdefault(n, []).append(r["summary"])
        del svd, bpr

    out = {
        "train_positives": int(ds.train_csr.nnz),
        "train_retained_pct": ds.stats["train_retained_pct"],
        "items_with_positives": int((ds.item_pop > 0).sum()),
        "items_passing_trend_gate": int((ds.trend_score > 0).sum()),
        "mean_positives_per_item": round(float(ds.item_pop[ds.item_pop > 0].mean()), 2),
        "eligible_users_after_thinning": int(len(ds.eval_users)),
        "itemknn_fit_sec": round(t_k, 2),
        "content_fit_sec": round(t_c, 2),
        "systems": {},
    }
    for n, sums in acc.items():
        out["systems"][n] = {
            "NDCG@10": float(np.mean([s["NDCG@10"] for s in sums])),
            "NDCG@10_sd": float(np.std([s["NDCG@10"] for s in sums])),
            "HR@10": float(np.mean([s["HR@10"] for s in sums])),
            "Coverage": float(np.mean([s["Coverage"] for s in sums])),
        }
    return out


def print_arm(title, conditions, key_label):
    print(f"\n{title}")
    names = [s["name"] for s in SYSTEMS]
    head = f"  {key_label:>10} {'kept%':>7} {'pos/item':>9} " + " ".join(
        f"{n.split()[0][:9]:>9}" for n in names)
    print(head)
    for label, c in conditions:
        cells = " ".join(f"{c['systems'][n]['NDCG@10']:9.4f}" for n in names)
        print(f"  {label:>10} {c['train_retained_pct']:7.1f}"
              f" {c['mean_positives_per_item']:9.1f} {cells}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--users", type=int, default=5000)
    ap.add_argument("--chunk", type=int, default=500)
    args = ap.parse_args()

    print("Loading full dataset ...")
    full = load_dataset(verbose=False, cache_raw=True)
    full_eval_users = [sampled_users(full, s, args.users) for s in range(args.seeds)]

    out = {"seeds": args.seeds, "users_per_seed": args.users}

    print("\n=== arm 0: stratification by user profile length (no refit) ===")
    strat = stratify(full, args.seeds, args.users)
    if strat:
        out["stratify"] = strat
        names = [s["name"] for s in SYSTEMS]
        print(f"  {'profile':>9} {'users':>7} {'share':>7} " +
              " ".join(f"{n.split()[0][:9]:>9}" for n in names))
        for r in strat["bins"]:
            cells = " ".join(f"{r.get(n, float('nan')):9.4f}" for n in names)
            print(f"  {r['profile_length']:>9} {r['users']:>7,}"
                  f" {r['share_of_users_pct']:6.1f}% {cells}")
        print(f"  median profile length = {strat['median_profile_length']:.0f}"
              f" training positives")

    arms = [("global", GLOBAL_LEVELS, "fraction"),
            ("user", USER_CAPS, "cap/user"),
            ("item", ITEM_CAPS, "cap/item")]
    for mode, levels, label in arms:
        print(f"\n=== arm: {mode} ===")
        conditions = []
        for lv in levels:
            spec = None if (mode == "global" and lv == 1.0) else {"mode": mode, "level": lv}
            t0 = time.perf_counter()
            c = run_condition(spec, args.seeds, args.users, args.chunk, full_eval_users)
            conditions.append((str(lv), c))
            print(f"  {mode}={lv}: kept {c['train_retained_pct']}% "
                  f"({c['train_positives']:,} positives, "
                  f"{c['mean_positives_per_item']} per item) "
                  f"in {time.perf_counter() - t0:.0f}s")
        out[f"arm_{mode}"] = {lv: c for lv, c in conditions}
        print_arm(f"NDCG@10 -- arm {mode}", conditions, label)

    path = RESULTS_DIR / "sparsity_sweep.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
