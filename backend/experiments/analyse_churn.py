"""Catalogue-churn stress test: how the systems behave as stock turns over.

Removes all training interactions from a random share of products, leaving
their held-out purchases in the test set, then reports overall NDCG@10, recall
on the cold products themselves, and the share of each list made up of them.

    python -m experiments.analyse_churn [--seeds 3] [--users 5000]

Writes results/churn_stress.json.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.decomposition import TruncatedSVD

from .data import load_dataset
from .evaluate import evaluate_systems
from .metrics import K
from .models import ContentBased, ItemKNN, Popularity, fit_timed

RESULTS_DIR = Path(__file__).resolve().parent / "results"
CHURN_LEVELS = [0.0, 0.10, 0.25, 0.50]

SYSTEMS = [
    {"name": "Trend (popularity)", "model": "T"},
    {"name": "Content-based", "model": "C"},
    {"name": "ItemKNN", "model": "CF"},
    {"name": "Hybrid 10/70/20", "weights": {"C": 0.1, "CF": 0.7, "T": 0.2}},
    {"name": "Hybrid 30/50/20", "weights": {"C": 0.3, "CF": 0.5, "T": 0.2}},
    {"name": "Hybrid 50/30/20", "weights": {"C": 0.5, "CF": 0.3, "T": 0.2}},
]


def _minmax_rows(x):
    lo = x.min(axis=1, keepdims=True)
    hi = x.max(axis=1, keepdims=True)
    return (x - lo) / np.where(hi - lo > 0, hi - lo, 1.0)


def cold_item_analysis(ds, users, components, cold_mask, chunk=500):
    """Top-k recall of held-out purchases that fall on cold items.

    evaluate_systems averages over all positives, which the warm ones dominate;
    this scores only the cold ones.
    """
    hits = {s["name"]: 0 for s in SYSTEMS}
    cold_in_list = {s["name"]: 0 for s in SYSTEMS}
    listed = {s["name"]: 0 for s in SYSTEMS}
    n_cold_targets = 0

    for start in range(0, len(users), chunk):
        cu = users[start:start + chunk]
        raw = {k: m.score_users(cu) for k, m in components.items()}
        norm = {k: _minmax_rows(v) for k, v in raw.items()}
        mask = ds.train_items_csr[cu]
        rows, cols = mask.nonzero()

        # which of each user's held-out positives are on cold items
        targets = [np.asarray(ds.test_pos[u]) for u in cu]
        targets = [t[cold_mask[t]] for t in targets]
        n_cold_targets += sum(t.size for t in targets)

        for spec in SYSTEMS:
            if "model" in spec:
                scores = raw[spec["model"]].copy()
            else:
                scores = np.zeros_like(next(iter(norm.values())))
                for key, w in spec["weights"].items():
                    if w:
                        scores += np.float32(w) * norm[key]
            scores[rows, cols] = -1e9
            part = np.argpartition(scores, -K, axis=1)[:, -K:]
            row_scores = np.take_along_axis(scores, part, axis=1)
            topk = np.take_along_axis(part, np.argsort(-row_scores, axis=1), axis=1)

            cold_in_list[spec["name"]] += int(cold_mask[topk].sum())
            listed[spec["name"]] += topk.size
            for r, t in enumerate(targets):
                if t.size:
                    hits[spec["name"]] += int(np.isin(t, topk[r]).sum())

    return {
        "cold_test_positives": int(n_cold_targets),
        "per_system": {
            n: {
                "cold_recall@10": (hits[n] / n_cold_targets) if n_cold_targets else 0.0,
                "cold_hits": hits[n],
                "cold_share_of_list_pct": round(100 * cold_in_list[n] / listed[n], 2)
                if listed[n] else 0.0,
            }
            for n in hits
        },
    }


def run_level(level, seeds, n_users, chunk, eval_samples, full_pop):
    spec = None if level == 0.0 else {"mode": "cold", "level": level}
    ds = load_dataset(verbose=False, thin=spec, thin_seed=0, cache_raw=True)

    # an item is cold if it had training positives before and has none now
    cold_mask = (ds.item_pop == 0) & (full_pop > 0)

    content, _ = fit_timed(ContentBased(), ds, 0)
    knn, _ = fit_timed(ItemKNN(k=200), ds, 0)
    pop, _ = fit_timed(Popularity(), ds, 0)
    components = {"C": content, "CF": knn, "T": pop}
    item_emb = TruncatedSVD(n_components=64, random_state=0).fit_transform(
        content.item_vecs).astype(np.float32)

    acc, cold = {}, []
    for seed in range(seeds):
        users = eval_samples[seed]
        res = evaluate_systems(ds, users, components, SYSTEMS, item_emb,
                               chunk_size=chunk)
        for n, r in res.items():
            acc.setdefault(n, []).append(r["summary"])
        cold.append(cold_item_analysis(ds, users, components, cold_mask, chunk))

    out = {
        "churn_level": level,
        "cold_items": int(cold_mask.sum()),
        "cold_items_pct": round(100 * float(cold_mask.mean()), 1),
        "train_positives": int(ds.train_csr.nnz),
        "cold_test_positives": int(np.mean([c["cold_test_positives"] for c in cold])),
        "systems": {},
    }
    for n, sums in acc.items():
        cr = [c["per_system"][n]["cold_recall@10"] for c in cold]
        cs = [c["per_system"][n]["cold_share_of_list_pct"] for c in cold]
        out["systems"][n] = {
            "NDCG@10": float(np.mean([s["NDCG@10"] for s in sums])),
            "NDCG@10_sd": float(np.std([s["NDCG@10"] for s in sums])),
            "HR@10": float(np.mean([s["HR@10"] for s in sums])),
            "cold_recall@10": float(np.mean(cr)),
            "cold_share_of_list_pct": float(np.mean(cs)),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--users", type=int, default=5000)
    ap.add_argument("--chunk", type=int, default=500)
    args = ap.parse_args()

    print("Loading full dataset ...")
    full = load_dataset(verbose=False, cache_raw=True)
    full_pop = full.item_pop.copy()
    eval_samples = []
    for s in range(args.seeds):
        rng = np.random.default_rng(s)
        eval_samples.append(rng.choice(full.eval_users,
                                       size=min(args.users, len(full.eval_users)),
                                       replace=False))

    levels = []
    for lv in CHURN_LEVELS:
        t0 = time.perf_counter()
        r = run_level(lv, args.seeds, args.users, args.chunk, eval_samples, full_pop)
        levels.append(r)
        print(f"  churn {lv:.0%}: {r['cold_items']:,} cold items "
              f"({r['cold_items_pct']}%), {r['cold_test_positives']:,} cold held-out "
              f"purchases, in {time.perf_counter() - t0:.0f}s")

    names = [s["name"] for s in SYSTEMS]
    print(f"\nNDCG@10 overall")
    print(f"  {'churn':>7} " + " ".join(f"{n[:15]:>15}" for n in names))
    for r in levels:
        print(f"  {r['churn_level']:>6.0%} " +
              " ".join(f"{r['systems'][n]['NDCG@10']:15.4f}" for n in names))

    print(f"\nRecall@10 on held-out purchases of COLD items "
          f"(the fallback's actual job)")
    print(f"  {'churn':>7} " + " ".join(f"{n[:15]:>15}" for n in names))
    for r in levels:
        if r["churn_level"] == 0.0:
            continue
        print(f"  {r['churn_level']:>6.0%} " +
              " ".join(f"{r['systems'][n]['cold_recall@10']:15.4f}" for n in names))

    print(f"\nShare of the recommended list made up of cold items (%)")
    print(f"  {'churn':>7} " + " ".join(f"{n[:15]:>15}" for n in names))
    for r in levels:
        if r["churn_level"] == 0.0:
            continue
        print(f"  {r['churn_level']:>6.0%} " +
              " ".join(f"{r['systems'][n]['cold_share_of_list_pct']:15.2f}" for n in names))

    path = RESULTS_DIR / "churn_stress.json"
    path.write_text(json.dumps({"seeds": args.seeds, "users_per_seed": args.users,
                                "levels": levels}, indent=2), encoding="utf-8")
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
