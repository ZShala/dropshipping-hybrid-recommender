"""Audit of the item metadata, and a re-test of H2 without it.

ContentBased tokenises "{ProductType} {slug}", where slug comes from the row's
URL. That URL does not describe the row's item: no ProductId matches the ASIN
in its own URL, and 23,838 products resolve to ~844 distinct slugs. This
quantifies the resulting degeneracy and re-runs the content-vs-popularity
comparison on the product-type field alone, which is complete for every item.

    python -m experiments.analyse_metadata_quality [--seeds 5] [--users 5000]

Writes results/metadata_quality.json.
"""
import argparse
import dataclasses
import json
import re
import urllib.parse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import TruncatedSVD

from .data import DATA_PATH, load_dataset
from .evaluate import evaluate_systems
from .models import ContentBased, ItemKNN, Popularity, fit_timed

RESULTS_DIR = Path(__file__).resolve().parent / "results"

PUBLISHED = "type+slug"
FEATURE_SETS = (PUBLISHED, "type", "slug")


def _slug(url):
    m = re.search(r"amazon\.[a-z.]+/([^/]+)/dp/", str(url))
    if not m:
        return ""
    return urllib.parse.unquote(m.group(1)).replace("-", " ").replace("_", " ")


def audit_url_column():
    """Quantifies the ProductId / URL mismatch and the slug reuse it causes."""
    df = pd.read_csv(DATA_PATH, usecols=["ProductId", "ProductType", "URL"])
    per_item = df.drop_duplicates("ProductId")
    asin = per_item["URL"].map(
        lambda u: (re.search(r"/dp/([A-Z0-9]{10})", str(u)) or [None, ""])[1]
        if re.search(r"/dp/([A-Z0-9]{10})", str(u)) else ""
    )
    slug = per_item["URL"].map(_slug).str.strip()
    nonempty = slug[slug != ""]
    per_slug = nonempty.groupby(nonempty).size()
    ptype = per_item["ProductType"].fillna("").astype(str).str.strip()

    by_type = per_item.assign(_s=slug)
    by_type = by_type[by_type["_s"] != ""].groupby("ProductType")["_s"].nunique()

    return {
        "items": int(len(per_item)),
        "url_with_parseable_asin_pct": round(100 * float((asin != "").mean()), 1),
        "productid_matches_own_url_asin_pct": round(
            100 * float((per_item["ProductId"] == asin).mean()), 2),
        "distinct_url_asins": int(asin[asin != ""].nunique()),
        "distinct_slugs": int(per_slug.size),
        "items_without_slug": int((slug == "").sum()),
        "items_without_slug_pct": round(100 * float((slug == "").mean()), 1),
        "items_per_slug_median": float(per_slug.median()),
        "items_per_slug_max": int(per_slug.max()),
        "product_type_coverage_pct": round(100 * float((ptype != "").mean()), 1),
        "distinct_product_types": int(ptype[ptype != ""].nunique()),
        "distinct_slugs_per_type_min": int(by_type.min()),
        "distinct_slugs_per_type_max": int(by_type.max()),
        "note": "no ProductId matches the ASIN in its own URL; the URL column "
                "was evidently joined to the interaction rows independently of "
                "item identity",
    }


def build_text(ds, feature_set):
    """Rebuild `item_text` from the requested fields.

    `ds.item_text[i]` is `f"{ProductType} {slug}"`, and `ds.type_names[
    ds.item_type[i]]` recovers the type, so the slug is the remainder.
    """
    out = []
    for i, text in enumerate(ds.item_text):
        ptype = ds.type_names[ds.item_type[i]]
        rest = text[len(ptype):].strip() if text.startswith(ptype) else text
        if feature_set == "type":
            out.append(ptype)
        elif feature_set == "slug":
            out.append(rest)
        else:
            out.append(text)
    return out


def degeneracy(model, n_items):
    """How many items an average item is indistinguishable from in TF-IDF space.

    Rows of the L2-normalised TF-IDF matrix are hashed; items with an identical
    sparse pattern and identical weights get the same content vector and so can
    never be separated by cosine similarity.
    """
    m = model.item_vecs.tocsr()
    sigs = []
    for r in range(n_items):
        s, e = m.indptr[r], m.indptr[r + 1]
        sigs.append((tuple(m.indices[s:e].tolist()),
                     tuple(np.round(m.data[s:e], 6).tolist())))
    counts = pd.Series(sigs).value_counts()
    group_of_item = counts.reindex(pd.Series(sigs)).to_numpy()
    return {
        "distinct_content_vectors": int(counts.size),
        "distinct_content_vectors_pct": round(100 * counts.size / n_items, 1),
        "empty_vectors": int(sum(1 for s in sigs if not s[0])),
        "mean_indistinguishable_group_size": round(float(group_of_item.mean()), 1),
        "median_indistinguishable_group_size": float(np.median(group_of_item)),
        "largest_indistinguishable_group": int(counts.max()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--users", type=int, default=5000)
    ap.add_argument("--chunk", type=int, default=500)
    args = ap.parse_args()

    print("Auditing the URL / ProductId correspondence ...")
    audit = audit_url_column()
    for k, v in audit.items():
        print(f"  {k}: {v}")

    print("\nLoading dataset ...")
    ds = load_dataset(verbose=False)

    # content models, one per feature set
    content = {}
    deg = {}
    for fs in FEATURE_SETS:
        ds_fs = dataclasses.replace(ds, item_text=build_text(ds, fs))
        model, t = fit_timed(ContentBased(), ds_fs, 0)
        content[fs] = model
        deg[fs] = degeneracy(model, ds.n_items)
        print(f"\ncontent[{fs}] fitted in {t:.2f}s")
        for k, v in deg[fs].items():
            print(f"    {k}: {v}")

    knn, _ = fit_timed(ItemKNN(k=200), ds, 0)
    pop, _ = fit_timed(Popularity(), ds, 0)

    # diversity embedding: keep the published one so ILD stays comparable
    item_emb = TruncatedSVD(n_components=64, random_state=0).fit_transform(
        content[PUBLISHED].item_vecs).astype(np.float32)

    components = {"CF": knn, "T": pop}
    for fs in FEATURE_SETS:
        components[f"C_{fs}"] = content[fs]

    systems = [{"name": "Trend (popularity)", "model": "T"},
               {"name": "ItemKNN", "model": "CF"}]
    for fs in FEATURE_SETS:
        systems.append({"name": f"Content [{fs}]", "model": f"C_{fs}"})
    # the paper's hybrid, rebuilt on each content variant
    for fs in FEATURE_SETS:
        systems.append({"name": f"Hybrid 10/70/20 [{fs}]",
                        "weights": {f"C_{fs}": 0.1, "CF": 0.7, "T": 0.2}})

    pooled = {}
    summaries = []
    for seed in range(args.seeds):
        rng = np.random.default_rng(seed)          # same sampling as the paper
        users = rng.choice(ds.eval_users,
                           size=min(args.users, len(ds.eval_users)), replace=False)
        res = evaluate_systems(ds, users, components, systems, item_emb,
                               chunk_size=args.chunk)
        summaries.append({n: r["summary"] for n, r in res.items()})
        for n, r in res.items():
            for metric, arr in r["per_user"].items():
                pooled.setdefault(f"{n}|{metric}", []).append(arr)
        print(f"  seed {seed} done")
    pooled = {k: np.concatenate(v) for k, v in pooled.items()}

    def mean_std(name, metric):
        vals = [s[name][metric] for s in summaries]
        return float(np.mean(vals)), float(np.std(vals))

    print(f"\n{'system':30s} {'NDCG@10':>16} {'HR@10':>16} {'Coverage':>10}")
    table = {}
    for spec in systems:
        n = spec["name"]
        nd, nds = mean_std(n, "NDCG@10")
        hr, hrs = mean_std(n, "HR@10")
        cov, _ = mean_std(n, "Coverage")
        table[n] = {"NDCG@10": nd, "NDCG@10_sd": nds, "HR@10": hr,
                    "HR@10_sd": hrs, "Coverage": cov}
        print(f"{n:30s} {nd:8.4f} +-{nds:.4f} {hr:8.4f} +-{hrs:.4f} {cov:9.3f}")

    # H2 re-test: does each content variant still lose to popularity?
    print("\nH2 re-test -- content vs popularity (pooled per-user Wilcoxon)")
    h2 = []
    ref = pooled["Trend (popularity)|NDCG@10"]
    for fs in FEATURE_SETS:
        arr = pooled[f"Content [{fs}]|NDCG@10"]
        n = min(ref.size, arr.size)
        a, b = ref[:n], arr[:n]
        try:
            w, p = stats.wilcoxon(a, b, zero_method="wilcox")
            w, p = float(w), float(p)
        except ValueError:
            w, p = float("nan"), 1.0
        ratio = float(a.mean() / b.mean()) if b.mean() > 0 else float("inf")
        row = {"feature_set": fs, "popularity_ndcg": float(a.mean()),
               "content_ndcg": float(b.mean()),
               "popularity_over_content": round(ratio, 2),
               "wilcoxon_W": w, "wilcoxon_p": p,
               "popularity_wins": int((a > b).sum()),
               "content_wins": int((b > a).sum()),
               "ties": int((a == b).sum())}
        h2.append(row)
        print(f"  content [{fs:9s}] {b.mean():.5f}  vs popularity {a.mean():.5f}"
              f"  -> popularity is {ratio:5.2f}x better, p = {p:.3g}"
              f"   ({row['popularity_wins']:,} / {row['content_wins']:,} users)")

    # does the hybrid's gain over ItemKNN survive on clean features?
    print("\nHybrid gain over ItemKNN, per content feature set")
    hyb = []
    knn_arr = pooled["ItemKNN|NDCG@10"]
    for fs in FEATURE_SETS:
        arr = pooled[f"Hybrid 10/70/20 [{fs}]|NDCG@10"]
        n = min(knn_arr.size, arr.size)
        a, b = arr[:n], knn_arr[:n]
        try:
            w, p = stats.wilcoxon(a, b, zero_method="wilcox")
            w, p = float(w), float(p)
        except ValueError:
            w, p = float("nan"), 1.0
        d = a - b
        sd = d.std(ddof=1)
        row = {"feature_set": fs, "hybrid_ndcg": float(a.mean()),
               "itemknn_ndcg": float(b.mean()), "delta": float(d.mean()),
               "relative_pct": round(100 * float(d.mean() / b.mean()), 2),
               "cohens_dz": round(float(d.mean() / sd), 4) if sd > 0 else 0.0,
               "wilcoxon_p": p,
               "wins": int((d > 0).sum()), "losses": int((d < 0).sum()),
               "ties": int((d == 0).sum())}
        hyb.append(row)
        print(f"  hybrid [{fs:9s}] {a.mean():.5f} vs ItemKNN {b.mean():.5f}"
              f"  delta = {d.mean():+.5f} ({row['relative_pct']:+.2f}%),"
              f" p = {p:.3g}, d_z = {row['cohens_dz']:+.4f}")

    out = {"url_audit": audit, "degeneracy": deg, "systems": table,
           "h2_retest": h2, "hybrid_gain": hyb,
           "seeds": args.seeds, "users_per_seed": args.users}
    path = RESULTS_DIR / "metadata_quality.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
