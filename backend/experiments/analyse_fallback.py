"""Catalogue reachability of each hybrid component.

Quantifies how much of the catalogue the behavioural components can rank at
all, and therefore where the content-based fallback is the only option.

An item with no positive training interaction has an all-zero column in
`train_csr`, so its cosine similarity to every other item is zero and ItemKNN
can never place it in a recommendation list, at any neighbourhood size. The
trend component is gated on >= 10 reviews and mean rating > 4.0 (Eq. 4), so
items below that gate score zero as well. The content component depends only
on product text, which every item has.

    python -m experiments.analyse_fallback
"""
import json

import numpy as np

from .data import DATA_PATH, load_dataset

RESULTS = DATA_PATH.parents[1] / "backend" / "experiments" / "results"


def main():
    ds = load_dataset(verbose=False)
    n = ds.n_items

    cf_dead = ds.item_pop == 0                      # unreachable by ItemKNN
    trend_dead = ds.trend_score == 0                # fails the Eq. 4 gate
    has_text = np.array([bool(t.strip()) for t in ds.item_text])
    any_train = np.asarray(ds.train_items_csr.sum(axis=0)).ravel() > 0

    only_content = cf_dead & trend_dead & has_text

    # share of held-out purchases that land on CF-unreachable items
    test_items = np.concatenate([v for v in ds.test_pos.values()]) if ds.test_pos else np.array([], dtype=int)
    cold_test = int(cf_dead[test_items].sum()) if test_items.size else 0

    out = {
        "n_items": int(n),
        "cf_unreachable": int(cf_dead.sum()),
        "cf_unreachable_pct": round(100 * float(cf_dead.mean()), 2),
        "no_train_interaction_at_all": int((~any_train).sum()),
        "interactions_but_no_positives": int((any_train & cf_dead).sum()),
        "trend_ineligible": int(trend_dead.sum()),
        "trend_ineligible_pct": round(100 * float(trend_dead.mean()), 2),
        "content_scoreable": int(has_text.sum()),
        "content_scoreable_pct": round(100 * float(has_text.mean()), 2),
        "only_content_can_rank": int(only_content.sum()),
        "only_content_can_rank_pct": round(100 * float(only_content.mean()), 2),
        "test_positives": int(test_items.size),
        "test_positives_on_cf_unreachable_items": cold_test,
        "test_positives_on_cf_unreachable_items_pct": round(100 * cold_test / test_items.size, 2) if test_items.size else 0.0,
    }

    for k, v in out.items():
        print(f"  {k}: {v}")
    (RESULTS / "fallback_stats.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nWrote {RESULTS / 'fallback_stats.json'}")


if __name__ == "__main__":
    main()
