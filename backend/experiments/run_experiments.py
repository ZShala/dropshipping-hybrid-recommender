"""Run the full offline experiment suite.

Usage (from the backend/ directory):
    python -m experiments.run_experiments --seeds 5 --users 5000

Per seed:
  - fit stochastic models (PureSVD, BPR) with that seed,
  - sample an independent set of evaluation users,
  - evaluate all systems (baselines, hybrids, ablation, sensitivity grid),
  - write summaries + per-user metric arrays to experiments/results/.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.decomposition import TruncatedSVD

from .data import load_dataset
from .evaluate import evaluate_systems
from .models import BPR, CFProxy, ContentBased, ItemKNN, Popularity, PureSVD, fit_timed

RESULTS_DIR = Path(__file__).resolve().parent / "results"

# hybrid components: C = content, CF = ItemKNN, P = fixed proxy, T = trend
PAPER_CONFIGS = [
    (0.33, 0.33, 0.34),
    (0.40, 0.35, 0.25),
    (0.50, 0.30, 0.20),
    (0.30, 0.50, 0.20),
    (0.20, 0.40, 0.40),
]


def hybrid_name(c, cf, t, cf_key="CF"):
    tag = "Hybrid" if cf_key == "CF" else "Hybrid-proxy"
    return f"{tag} {round(c*100)}/{round(cf*100)}/{round(t*100)}"


def build_systems():
    systems = [
        {"name": "Trend (popularity)", "model": "T"},
        {"name": "Content-based", "model": "C"},
        {"name": "CF proxy (fixed 0.8)", "model": "P"},
        {"name": "ItemKNN", "model": "CF"},
        {"name": "PureSVD", "model": "SVD"},
        {"name": "BPR", "model": "BPR"},
        # paper replication: proxy as the collaborative component
        {"name": hybrid_name(0.5, 0.3, 0.2, "P"), "weights": {"C": 0.5, "P": 0.3, "T": 0.2}},
        # ablation pairs (weights renormalised from 50/30/20)
        {"name": "C+CF", "weights": {"C": 0.625, "CF": 0.375}},
        {"name": "C+T", "weights": {"C": 5 / 7, "T": 2 / 7}},
        {"name": "CF+T", "weights": {"CF": 0.6, "T": 0.4}},
    ]
    seen = {s["name"] for s in systems}
    for c, cf, t in PAPER_CONFIGS:
        name = hybrid_name(c, cf, t)
        if name not in seen:
            systems.append({"name": name, "weights": {"C": c, "CF": cf, "T": t}})
            seen.add(name)
    # sensitivity: full simplex grid, step 0.1
    for ci in range(11):
        for cfi in range(11 - ci):
            ti = 10 - ci - cfi
            c, cf, t = ci / 10, cfi / 10, ti / 10
            name = hybrid_name(c, cf, t)
            if name not in seen:
                systems.append({"name": name, "weights": {"C": c, "CF": cf, "T": t}})
                seen.add(name)
    return systems


def tune(users_n=2000, chunk=500):
    """Hyperparameter search on a validation split carved from training data."""
    print("Loading dataset (validation split) ...")
    ds = load_dataset(validation=True)
    content, _ = fit_timed(ContentBased(), ds, 0)
    emb_svd = TruncatedSVD(n_components=64, random_state=0)
    item_emb = emb_svd.fit_transform(content.item_vecs).astype(np.float32)

    components, systems = {}, []
    for k in [50, 100, 200, 500]:
        key = f"knn{k}"
        components[key], t = fit_timed(ItemKNN(k=k), ds, 0)
        systems.append({"name": f"ItemKNN k={k}", "model": key})
        print(f"  fitted ItemKNN k={k} in {t:.0f}s")
    for f in [16, 32, 64, 128, 256]:
        key = f"svd{f}"
        components[key], t = fit_timed(PureSVD(factors=f), ds, 0)
        systems.append({"name": f"PureSVD f={f}", "model": key})
        print(f"  fitted PureSVD f={f} in {t:.0f}s")
    for f in [32, 64, 128]:
        for lr in [0.01, 0.05]:
            for it in [100, 300]:
                key = f"bpr{f}_{lr}_{it}"
                components[key], t = fit_timed(
                    BPR(factors=f, learning_rate=lr, iterations=it), ds, 0)
                systems.append({"name": f"BPR f={f} lr={lr} it={it}", "model": key})
                print(f"  fitted BPR f={f} lr={lr} it={it} in {t:.0f}s")

    rng = np.random.default_rng(0)
    users = rng.choice(ds.eval_users, size=min(users_n, len(ds.eval_users)), replace=False)
    results = evaluate_systems(ds, users, components, systems, item_emb, chunk_size=chunk)
    print(f"\n{'variant':18s} {'NDCG@10':>8} {'HR@10':>8} {'MAP@10':>8}")
    for s in systems:
        m = results[s["name"]]["summary"]
        print(f"{s['name']:18s} {m['NDCG@10']:8.4f} {m['HR@10']:8.4f} {m['MAP@10']:8.4f}")
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "tuning.json", "w") as f:
        json.dump({s["name"]: results[s["name"]]["summary"] for s in systems}, f, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--users", type=int, default=5000)
    ap.add_argument("--chunk", type=int, default=500)
    ap.add_argument("--smoke", action="store_true", help="tiny run for a quick sanity check")
    ap.add_argument("--tune", action="store_true", help="hyperparameter search on validation split")
    args = ap.parse_args()
    if args.tune:
        tune()
        return
    if args.smoke:
        args.seeds, args.users = 1, 300

    RESULTS_DIR.mkdir(exist_ok=True)
    print("Loading dataset ...")
    ds = load_dataset()
    with open(RESULTS_DIR / "dataset_stats.json", "w") as f:
        json.dump(ds.stats, f, indent=2)

    print("Fitting deterministic models (shared across seeds) ...")
    fit_times = {}
    content, fit_times["Content-based"] = fit_timed(ContentBased(), ds, 0)
    knn, fit_times["ItemKNN"] = fit_timed(ItemKNN(k=200), ds, 0)
    pop, fit_times["Trend (popularity)"] = fit_timed(Popularity(), ds, 0)
    proxy, fit_times["CF proxy (fixed 0.8)"] = fit_timed(CFProxy(), ds, 0)
    print(f"  fit times: { {k: round(v, 1) for k, v in fit_times.items()} }")

    print("Building 64-d content embedding for diversity metrics ...")
    emb_svd = TruncatedSVD(n_components=64, random_state=0)
    item_emb = emb_svd.fit_transform(content.item_vecs).astype(np.float32)

    systems = build_systems()
    print(f"{len(systems)} systems to evaluate per seed")

    for seed in range(args.seeds):
        t0 = time.perf_counter()
        print(f"\n=== Seed {seed} ===")
        svd, t_svd = fit_timed(PureSVD(factors=256), ds, seed)
        bpr, t_bpr = fit_timed(
            BPR(factors=128, learning_rate=0.05, iterations=300), ds, seed
        )
        print(f"  PureSVD fit {t_svd:.0f}s, BPR fit {t_bpr:.0f}s")
        fit_times["PureSVD"], fit_times["BPR"] = t_svd, t_bpr

        rng = np.random.default_rng(seed)
        users = rng.choice(ds.eval_users, size=min(args.users, len(ds.eval_users)),
                           replace=False)

        components = {"C": content, "CF": knn, "T": pop, "P": proxy,
                      "SVD": svd, "BPR": bpr}
        results = evaluate_systems(ds, users, components, systems, item_emb,
                                   chunk_size=args.chunk)

        summaries = {n: r["summary"] for n, r in results.items()}
        with open(RESULTS_DIR / f"summary_seed{seed}.json", "w") as f:
            json.dump({"seed": seed, "n_users": len(users),
                       "fit_times_sec": fit_times, "systems": summaries}, f, indent=2)

        per_user = {}
        for n, r in results.items():
            for metric, arr in r["per_user"].items():
                per_user[f"{n}|{metric}"] = arr
        np.savez_compressed(RESULTS_DIR / f"per_user_seed{seed}.npz", **per_user)
        print(f"  seed {seed} done in {time.perf_counter() - t0:.0f}s")
        best = max(summaries.items(), key=lambda kv: kv[1]["NDCG@10"])
        print(f"  best NDCG@10: {best[0]} = {best[1]['NDCG@10']:.4f}")

    print("\nAll seeds complete. Results in", RESULTS_DIR)


if __name__ == "__main__":
    main()
