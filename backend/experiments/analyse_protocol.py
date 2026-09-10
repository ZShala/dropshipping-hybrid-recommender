"""Sensitivity to the relevance threshold and the evaluation floor.

Varies each constant around its published setting and reports Kendall's tau of
the system ordering against that baseline. Absolute NDCG@10 must move, since
both constants change the number of relevant items; what matters is whether
the ordering holds. Each condition is scored on its own eligible-user pool.

    python -m experiments.analyse_protocol [--seeds 3] [--users 5000]

Writes results/protocol_sensitivity.json.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.decomposition import TruncatedSVD

from .data import load_dataset
from .evaluate import evaluate_systems
from .models import BPR, ContentBased, ItemKNN, Popularity, PureSVD, fit_timed

RESULTS_DIR = Path(__file__).resolve().parent / "results"

BASELINE = {"positive_threshold": 4.0, "min_user_interactions": 5}
CONDITIONS = [
    ("published (>=4.0, >=5)", {"positive_threshold": 4.0, "min_user_interactions": 5}),
    ("relevance >=3.0", {"positive_threshold": 3.0, "min_user_interactions": 5}),
    ("relevance >=5.0", {"positive_threshold": 5.0, "min_user_interactions": 5}),
    ("eval floor >=10", {"positive_threshold": 4.0, "min_user_interactions": 10}),
    ("eval floor >=20", {"positive_threshold": 4.0, "min_user_interactions": 20}),
]

SYSTEMS = [
    {"name": "Trend (popularity)", "model": "T"},
    {"name": "Content-based", "model": "C"},
    {"name": "ItemKNN", "model": "CF"},
    {"name": "PureSVD", "model": "SVD"},
    {"name": "BPR", "model": "BPR"},
    {"name": "Hybrid 10/70/20", "weights": {"C": 0.1, "CF": 0.7, "T": 0.2}},
]


def run_condition(cfg, seeds, n_users, chunk):
    ds = load_dataset(verbose=False, cache_raw=True, **cfg)

    content, _ = fit_timed(ContentBased(), ds, 0)
    knn, _ = fit_timed(ItemKNN(k=200), ds, 0)
    pop, _ = fit_timed(Popularity(), ds, 0)
    item_emb = TruncatedSVD(n_components=64, random_state=0).fit_transform(
        content.item_vecs).astype(np.float32)

    acc = {}
    for seed in range(seeds):
        svd, _ = fit_timed(PureSVD(factors=256), ds, seed)
        bpr, _ = fit_timed(BPR(factors=128, learning_rate=0.05, iterations=300),
                           ds, seed)
        components = {"C": content, "CF": knn, "T": pop, "SVD": svd, "BPR": bpr}
        rng = np.random.default_rng(seed)
        users = rng.choice(ds.eval_users, size=min(n_users, len(ds.eval_users)),
                           replace=False)
        res = evaluate_systems(ds, users, components, SYSTEMS, item_emb,
                               chunk_size=chunk)
        for n, r in res.items():
            acc.setdefault(n, []).append(r["summary"])
        del svd, bpr

    return {
        "config": cfg,
        "eval_users_available": int(len(ds.eval_users)),
        "train_positives": int(ds.train_csr.nnz),
        "test_positives": int(ds.stats["test_positives"]),
        "systems": {
            n: {"NDCG@10": float(np.mean([s["NDCG@10"] for s in sums])),
                "NDCG@10_sd": float(np.std([s["NDCG@10"] for s in sums])),
                "HR@10": float(np.mean([s["HR@10"] for s in sums]))}
            for n, sums in acc.items()
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--users", type=int, default=5000)
    ap.add_argument("--chunk", type=int, default=500)
    args = ap.parse_args()

    names = [s["name"] for s in SYSTEMS]
    results = []
    for label, cfg in CONDITIONS:
        t0 = time.perf_counter()
        r = run_condition(cfg, args.seeds, args.users, args.chunk)
        r["label"] = label
        results.append(r)
        print(f"  {label:24s} {r['eval_users_available']:>7,} eligible users, "
              f"{r['train_positives']:>9,} train positives "
              f"({time.perf_counter() - t0:.0f}s)")

    base = results[0]
    base_order = [base["systems"][n]["NDCG@10"] for n in names]

    print(f"\nNDCG@10 by condition")
    print(f"  {'condition':24s} " + " ".join(f"{n[:14]:>14}" for n in names) + "   tau")
    for r in results:
        vals = [r["systems"][n]["NDCG@10"] for n in names]
        tau = stats.kendalltau(base_order, vals).statistic
        r["kendall_tau_vs_published"] = float(tau)
        r["best_system"] = names[int(np.argmax(vals))]
        print(f"  {r['label']:24s} " + " ".join(f"{v:14.4f}" for v in vals)
              + f" {tau:6.3f}")

    print(f"\n  best system per condition:")
    for r in results:
        print(f"    {r['label']:24s} -> {r['best_system']}")

    stable = all(r["best_system"] == results[0]["best_system"] for r in results)
    print(f"\n  top-ranked system identical across all conditions: {stable}")
    print(f"  minimum Kendall tau vs published ordering: "
          f"{min(r['kendall_tau_vs_published'] for r in results):.3f}")

    path = RESULTS_DIR / "protocol_sensitivity.json"
    path.write_text(json.dumps({"seeds": args.seeds, "users_per_seed": args.users,
                                "conditions": results}, indent=2), encoding="utf-8")
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
