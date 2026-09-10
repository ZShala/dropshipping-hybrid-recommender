"""How far the dataset reproduces dropshipping conditions.

Measures catalogue turnover, item lifespan, cold-item pressure, metadata
coverage and demand concentration. Factors the data cannot speak to (supplier
lead time, stockouts, fulfilment delay, abandonment, margin, returns) are
listed in the output rather than estimated.

    python -m experiments.analyse_ecology

Writes results/ecology_stats.json.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import DATA_PATH, POSITIVE_THRESHOLD, _slug_from_url

RESULTS_DIR = Path(__file__).resolve().parent / "results"
N_WINDOWS = 8            # equal-mass, not equal-width: see below
SHORT_LIFE_DAYS = 90     # an item whose whole history fits in one quarter
DAY = 86_400

# Volume grows roughly exponentially over the 14.6-year span, so equal-width
# windows would put 76% of interactions in the last one. Windows are time
# quantiles instead, making turnover rates comparable across them.


def gini(counts):
    """Gini coefficient of a popularity distribution (0 = flat, 1 = one item)."""
    x = np.sort(np.asarray(counts, dtype=np.float64))
    n = x.size
    if n == 0 or x.sum() == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float((2 * (idx * x).sum() - (n + 1) * x.sum()) / (n * x.sum()))


def main():
    df = pd.read_csv(
        DATA_PATH,
        usecols=["UserId", "ProductId", "ProductType", "Rating", "Timestamp", "URL"],
    )
    n_rows = len(df)
    ts = df["Timestamp"].to_numpy()
    t_min, t_max = int(ts.min()), int(ts.max())
    span_days = (t_max - t_min) / DAY

    out = {
        "interactions": int(n_rows),
        "items": int(df["ProductId"].nunique()),
        "users": int(df["UserId"].nunique()),
        "observed_span_days": round(span_days, 1),
        "observed_span_years": round(span_days / 365.25, 2),
        "first_interaction": pd.to_datetime(t_min, unit="s").strftime("%Y-%m-%d"),
        "last_interaction": pd.to_datetime(t_max, unit="s").strftime("%Y-%m-%d"),
    }

    # ---- catalogue turnover over equal-mass windows ------------------------
    qs = np.linspace(0, 1, N_WINDOWS + 1)[1:-1]
    edges = np.concatenate([[t_min], np.quantile(ts, qs), [t_max + 1]])
    win = np.digitize(ts, edges[1:-1])            # 0 .. N_WINDOWS-1
    df["win"] = win

    item_windows = df.groupby("ProductId")["win"]
    first_win = item_windows.min()
    last_win = item_windows.max()

    windows = []
    seen_before = set()
    for w in range(N_WINDOWS):
        sub = df[df["win"] == w]
        items_w = set(sub["ProductId"].unique())
        new_items = items_w - seen_before
        # items active in this window that never appear again
        retiring = {i for i in items_w if last_win[i] == w}
        rows_on_new = int(sub["ProductId"].isin(new_items).sum())
        is_final = w == N_WINDOWS - 1
        windows.append({
            "window": w + 1,
            "start": pd.to_datetime(edges[w], unit="s").strftime("%Y-%m-%d"),
            "days": round((edges[w + 1] - edges[w]) / DAY, 1),
            "interactions": int(len(sub)),
            "active_items": len(items_w),
            "new_items": len(new_items),
            "new_items_pct": round(100 * len(new_items) / len(items_w), 1) if items_w else 0.0,
            # every item active in the final window trivially "retires" there,
            # so the exit rate is undefined for it
            "last_seen_here": None if is_final else len(retiring),
            "last_seen_here_pct": None if is_final else (
                round(100 * len(retiring) / len(items_w), 1) if items_w else 0.0),
            "interactions_on_new_items_pct": round(100 * rows_on_new / len(sub), 1) if len(sub) else 0.0,
        })
        seen_before |= items_w
    out["windows"] = windows

    # drop window 1: everything is new there by construction
    steady = windows[1:]
    out["mean_new_items_pct_steady"] = round(float(np.mean([w["new_items_pct"] for w in steady])), 1)
    out["mean_interactions_on_new_items_pct_steady"] = round(
        float(np.mean([w["interactions_on_new_items_pct"] for w in steady])), 1)
    exits = [w["last_seen_here_pct"] for w in steady if w["last_seen_here_pct"] is not None]
    out["mean_exit_pct_steady"] = round(float(np.mean(exits)), 1) if exits else None

    # ---- item lifespan ----------------------------------------------------
    life = (item_windows.count() * 0).astype(float)   # placeholder to keep index
    spans = df.groupby("ProductId")["Timestamp"].agg(["min", "max"])
    life = (spans["max"] - spans["min"]) / DAY
    out["item_lifespan_days"] = {
        "median": round(float(life.median()), 1),
        "mean": round(float(life.mean()), 1),
        "p25": round(float(life.quantile(0.25)), 1),
        "p75": round(float(life.quantile(0.75)), 1),
        "single_day_items_pct": round(100 * float((life == 0).mean()), 1),
        "under_90_days_pct": round(100 * float((life < SHORT_LIFE_DAYS).mean()), 1),
    }

    # ---- cold-item pressure at the temporal boundary ----------------------
    last = df[df["win"] == N_WINDOWS - 1]
    cold = set(first_win[first_win == N_WINDOWS - 1].index)
    rows_cold = int(last["ProductId"].isin(cold).sum())
    pos_last = last[last["Rating"] >= POSITIVE_THRESHOLD]
    pos_cold = int(pos_last["ProductId"].isin(cold).sum())
    out["final_window"] = {
        "interactions": int(len(last)),
        "active_items": int(last["ProductId"].nunique()),
        "items_first_seen_here": len(cold),
        "items_first_seen_here_pct": round(100 * len(cold) / last["ProductId"].nunique(), 1),
        "interactions_on_such_items": rows_cold,
        "interactions_on_such_items_pct": round(100 * rows_cold / len(last), 1) if len(last) else 0.0,
        "positives_on_such_items": pos_cold,
        "positives_on_such_items_pct": round(100 * pos_cold / len(pos_last), 1) if len(pos_last) else 0.0,
    }

    # ---- metadata quality -------------------------------------------------
    per_item = df.drop_duplicates("ProductId").set_index("ProductId")
    ptype = per_item["ProductType"].fillna("").astype(str).str.strip()
    slug = per_item["URL"].map(_slug_from_url)
    slug_tokens = slug.str.split().map(len)
    out["metadata"] = {
        "product_type_present_pct": round(100 * float((ptype != "").mean()), 1),
        "distinct_product_types": int(ptype[ptype != ""].nunique()),
        "median_items_per_type": int(ptype[ptype != ""].value_counts().median()),
        "url_slug_present_pct": round(100 * float((slug.str.strip() != "").mean()), 1),
        "slug_tokens_median": int(slug_tokens.median()),
        "slug_tokens_p10": int(slug_tokens.quantile(0.10)),
        "slug_tokens_under_5_pct": round(100 * float((slug_tokens < 5).mean()), 1),
        "note": "the content model tokenises the URL slug plus the product type; "
                "no description, brand, price or image field exists in this dataset",
    }

    # ---- slug reuse: a metadata artefact, NOT catalogue duplication -------
    # The URL column does not correspond to item identity (see
    # analyse_metadata_quality.py), so shared slugs are not duplicate listings.
    slug_nonempty = slug[slug.str.strip() != ""]
    per_slug = slug_nonempty.groupby(slug_nonempty).size()
    dup_slugs = per_slug[per_slug > 1]
    out["slug_reuse_artefact"] = {
        "items_with_slug": int(slug_nonempty.size),
        "distinct_slugs": int(per_slug.size),
        "slugs_used_by_more_than_one_item": int(dup_slugs.size),
        "items_sharing_a_slug": int(dup_slugs.sum()),
        "items_sharing_a_slug_pct": round(100 * float(dup_slugs.sum()) / slug_nonempty.size, 1),
        "largest_group": int(dup_slugs.max()) if dup_slugs.size else 0,
        "interpretation": "metadata defect, not catalogue duplication; "
                          "real listing duplication is not measurable here",
    }

    # ---- demand concentration --------------------------------------------
    pop = df["ProductId"].value_counts().to_numpy()
    pos_pop = df[df["Rating"] >= POSITIVE_THRESHOLD]["ProductId"].value_counts().to_numpy()
    n_items = pop.size
    top1 = max(1, int(round(0.01 * n_items)))
    top10 = max(1, int(round(0.10 * n_items)))
    out["concentration"] = {
        "gini_all_interactions": round(gini(pop), 3),
        "gini_positives": round(gini(pos_pop), 3),
        "interaction_share_top_1pct_items": round(100 * float(pop[:top1].sum() / pop.sum()), 1),
        "interaction_share_top_10pct_items": round(100 * float(pop[:top10].sum() / pop.sum()), 1),
        "items_covering_half_of_interactions": int(np.searchsorted(np.cumsum(pop), pop.sum() / 2) + 1),
        "median_interactions_per_item": int(np.median(pop)),
    }

    out["not_measurable"] = [
        "supplier lead time and stockout frequency",
        "delayed fulfilment and its effect on repeat purchase",
        "cart abandonment",
        "product margin and price volatility",
        "returns and refunds",
        "session or click sequence (ratings only, no browsing log)",
    ]

    path = RESULTS_DIR / "ecology_stats.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    # ---- console summary --------------------------------------------------
    print(f"span            {out['first_interaction']} to {out['last_interaction']}"
          f"  ({out['observed_span_years']} years, {N_WINDOWS} equal-mass windows)")
    print(f"scale           {out['interactions']:,} interactions,"
          f" {out['items']:,} items, {out['users']:,} users")
    print("\ncatalogue turnover per window (equal interaction mass)")
    print("  win  start          days  interactions  active     new items       exiting")
    for w in windows:
        exit_s = ("     n/a" if w["last_seen_here_pct"] is None
                  else f"{w['last_seen_here']:>5} ({w['last_seen_here_pct']:>4.1f}%)")
        print(f"  {w['window']:>3}  {w['start']}  {w['days']:>7.0f}"
              f"  {w['interactions']:>12,}  {w['active_items']:>6}"
              f"  {w['new_items']:>6} ({w['new_items_pct']:>4.1f}%)  {exit_s}")
    print(f"  steady-state: {out['mean_new_items_pct_steady']}% of a window's active items"
          f" are new, carrying {out['mean_interactions_on_new_items_pct_steady']}% of its"
          f" interactions; {out['mean_exit_pct_steady']}% exit")
    lf = out["item_lifespan_days"]
    print(f"\nitem lifespan   median {lf['median']} d, IQR [{lf['p25']}, {lf['p75']}];"
          f" {lf['single_day_items_pct']}% live one day,"
          f" {lf['under_90_days_pct']}% under {SHORT_LIFE_DAYS} d")
    fw = out["final_window"]
    print(f"cold pressure   final window: {fw['items_first_seen_here_pct']}% of active items"
          f" are new, carrying {fw['interactions_on_such_items_pct']}% of interactions"
          f" ({fw['positives_on_such_items_pct']}% of positives)")
    md = out["metadata"]
    print(f"metadata        product type on {md['product_type_present_pct']}% of items"
          f" ({md['distinct_product_types']} types); slug on {md['url_slug_present_pct']}%,"
          f" median {md['slug_tokens_median']} tokens,"
          f" {md['slug_tokens_under_5_pct']}% under 5 tokens")
    dp = out["slug_reuse_artefact"]
    print(f"slug reuse      {dp['items_sharing_a_slug_pct']}% of items share a URL slug"
          f" (largest group {dp['largest_group']}) -- a metadata defect, not"
          f" catalogue duplication")
    cc = out["concentration"]
    print(f"concentration   Gini {cc['gini_all_interactions']}"
          f" ({cc['gini_positives']} on positives);"
          f" top 1% of items hold {cc['interaction_share_top_1pct_items']}% of interactions;"
          f" {cc['items_covering_half_of_interactions']} items cover half")
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
