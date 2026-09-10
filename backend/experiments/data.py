"""Data loading, filtering and temporal splitting for offline experiments.

Protocol
--------
- Training uses ALL interactions in the dataset (including one-shot users,
  who contribute to item popularity and item-item similarity).
- Evaluation is restricted to users with >= MIN_USER_INTERACTIONS ratings.
  For each such user, interactions are ordered by timestamp and the most
  recent TEST_FRACTION (at least one) are held out as the test set.
- An interaction is a *positive* if its rating >= POSITIVE_THRESHOLD.
  Implicit-feedback models are trained on positives only; a held-out item
  counts as relevant only if it is a positive.
"""
import re
import urllib.parse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "Amazon_Beauty_Recommendation.csv"

POSITIVE_THRESHOLD = 4.0
MIN_USER_INTERACTIONS = 5
MIN_TRAIN_POSITIVES = 2
TEST_FRACTION = 0.2


def _slug_from_url(url: str) -> str:
    """Extract the human-readable product slug from an Amazon URL."""
    m = re.search(r"amazon\.[a-z.]+/([^/]+)/dp/", str(url))
    if not m:
        return ""
    slug = urllib.parse.unquote(m.group(1))
    return slug.replace("-", " ").replace("_", " ")


@dataclass
class Dataset:
    n_users: int
    n_items: int
    train_csr: sparse.csr_matrix          # binary positives, all users
    train_items_csr: sparse.csr_matrix    # ALL train interactions (for masking)
    test_pos: dict                        # user_idx -> np.ndarray of relevant item_idx
    eval_users: np.ndarray                # user indices eligible for evaluation
    item_pop: np.ndarray                  # positive-interaction count per item
    trend_score: np.ndarray               # review_count * avg_rating (paper Eq. 4)
    item_text: list                       # product text per item index
    item_type: np.ndarray                 # integer product-type id per item
    type_names: list
    stats: dict


def thin_training(train_df, spec, seed=0):
    """Remove training interactions to simulate a sparser store.

    Applied to the training frame only, before any derived array is computed,
    so item popularity, the Eq. 4 trend gate and the positive matrix all stay
    consistent with the interactions that remain. The test set is never
    touched, which keeps every sparsity level scored against the same held-out
    purchases.

    spec = {"mode": ..., "level": ...} where mode is one of

      "global"  keep a random `level` fraction of all training interactions;
                thins the user and item margins together
      "user"    keep at most `level` interactions per user, chosen at random;
                thins the user margin, leaving item columns fed by other users
      "item"    keep at most `level` interactions per item; thins the item
                margin, which is the one the paper claims dominates
      "cold"    remove *all* training interactions for a random `level`
                fraction of items, so they enter evaluation with no history at
                all. Their held-out purchases remain, which is what makes this
                a test of the content fallback rather than of sparsity: it
                simulates a catalogue where that share of stock was added
                after the model was last fitted.
    """
    mode, level = spec["mode"], spec["level"]
    rng = np.random.default_rng(seed)
    if mode == "global":
        keep = rng.random(len(train_df)) < level
        return train_df[keep]
    if mode == "cold":
        items = np.unique(train_df["i"].to_numpy())
        n_cold = int(round(level * items.size))
        cold = rng.choice(items, size=n_cold, replace=False)
        return train_df[~train_df["i"].isin(cold)]
    if mode not in ("user", "item"):
        raise ValueError(f"unknown thinning mode {mode!r}")
    key = "u" if mode == "user" else "i"
    # shuffle once, then keep the first `level` rows within each group: a
    # uniform random subsample per user (or per item) without a Python loop
    shuffled = train_df.iloc[rng.permutation(len(train_df))]
    return shuffled[shuffled.groupby(key).cumcount() < level]


_RAW_CACHE = {}


def load_dataset(path=DATA_PATH, verbose=True, validation=False,
                 thin=None, thin_seed=0, cache_raw=False,
                 positive_threshold=None, min_user_interactions=None) -> Dataset:
    """validation=True discards the final test rows and re-splits the remaining
    data with the same protocol, for hyperparameter tuning without touching
    the test set.

    thin={"mode": ..., "level": ...} sparsifies the training data (see
    `thin_training`); the test set and the split itself are unaffected.
    cache_raw=True keeps the parsed CSV in memory so a sweep over many
    thinning levels does not re-read 590 MB each time.

    positive_threshold and min_user_interactions override the module defaults,
    so the sensitivity of the whole protocol to the relevance definition and
    the evaluation-eligibility floor can be measured rather than assumed."""
    pos_thr = POSITIVE_THRESHOLD if positive_threshold is None else positive_threshold
    min_ui = MIN_USER_INTERACTIONS if min_user_interactions is None else min_user_interactions
    if cache_raw and path in _RAW_CACHE:
        df = _RAW_CACHE[path].copy()
    else:
        df = pd.read_csv(path)
        if cache_raw:
            _RAW_CACHE[path] = df.copy()
    raw_rows = len(df)
    df = df.dropna(subset=["UserId", "ProductId", "Rating", "Timestamp"])
    df = df.drop_duplicates(subset=["UserId", "ProductId"], keep="last")

    user_ids = df["UserId"].astype("category")
    item_ids = df["ProductId"].astype("category")
    df = df.assign(u=user_ids.cat.codes.values, i=item_ids.cat.codes.values)
    n_users = int(df["u"].max()) + 1
    n_items = int(df["i"].max()) + 1

    # --- product side information -------------------------------------------------
    prod = df.drop_duplicates("i").sort_values("i")
    types = prod["ProductType"].astype("category")
    item_type = np.zeros(n_items, dtype=np.int32)
    item_type[prod["i"].values] = types.cat.codes.values
    type_names = list(types.cat.categories)
    item_text = [""] * n_items
    for i, ptype, url in prod[["i", "ProductType", "URL"]].itertuples(index=False):
        item_text[i] = f"{ptype} {_slug_from_url(url)}".strip()

    # --- temporal per-user split ----------------------------------------------------
    df = df.sort_values(["u", "Timestamp"], kind="stable")

    def temporal_split(frame):
        counts = frame.groupby("u")["i"].transform("size")
        rank = frame.groupby("u").cumcount()               # 0..n-1 in time order
        n_test = np.ceil(counts * TEST_FRACTION).astype(int)
        is_eligible = counts >= min_ui
        is_test = is_eligible & (rank >= (counts - n_test))
        return frame[~is_test], frame[is_test]

    train_df, test_df = temporal_split(df)
    if validation:
        # tune on a split carved from the training portion only
        train_df, test_df = temporal_split(train_df)

    n_train_full = len(train_df)
    if thin is not None:
        train_df = thin_training(train_df, thin, seed=thin_seed)

    train_pos = train_df[train_df["Rating"] >= pos_thr]
    train_csr = sparse.csr_matrix(
        (np.ones(len(train_pos), dtype=np.float32), (train_pos["u"], train_pos["i"])),
        shape=(n_users, n_items),
    )
    train_items_csr = sparse.csr_matrix(
        (np.ones(len(train_df), dtype=np.float32), (train_df["u"], train_df["i"])),
        shape=(n_users, n_items),
    )

    # relevant test items = held-out positives
    test_pos_df = test_df[test_df["Rating"] >= pos_thr]
    test_pos = {u: grp["i"].values for u, grp in test_pos_df.groupby("u")}

    train_pos_per_user = np.asarray(train_csr.sum(axis=1)).ravel()
    eval_users = np.array(
        [u for u in test_pos if train_pos_per_user[u] >= MIN_TRAIN_POSITIVES],
        dtype=np.int64,
    )

    item_pop = np.asarray(train_csr.sum(axis=0)).ravel()
    # paper Eq. 4: trend = review_count * avg_rating, computed on training data only
    grp = train_df.groupby("i")["Rating"].agg(["count", "mean"])
    trend_score = np.zeros(n_items, dtype=np.float32)
    eligible_items = grp[(grp["count"] >= 10) & (grp["mean"] > pos_thr)]
    trend_score[eligible_items.index.values] = (
        eligible_items["count"] * eligible_items["mean"]
    ).astype(np.float32)

    stats = {
        "raw_rows": int(raw_rows),
        "interactions": int(len(df)),
        "n_users": int(n_users),
        "n_items": int(n_items),
        "sparsity_pct": float(100 * (1 - len(df) / (n_users * n_items))),
        "train_interactions": int(len(train_df)),
        "test_interactions": int(len(test_df)),
        "train_positives": int(len(train_pos)),
        "test_positives": int(len(test_pos_df)),
        "eligible_users": int(test_df["u"].nunique()),
        "validation_split": bool(validation),
        "eval_users": int(len(eval_users)),
        "thin": thin,
        "thin_seed": thin_seed if thin else None,
        "train_interactions_before_thinning": int(n_train_full),
        "train_retained_pct": round(100 * len(train_df) / n_train_full, 2) if n_train_full else 0.0,
        "positive_threshold": pos_thr,
        "min_user_interactions": min_ui,
        "test_fraction": TEST_FRACTION,
    }
    if verbose:
        for k, v in stats.items():
            print(f"  {k}: {v}")
    return Dataset(
        n_users=n_users,
        n_items=n_items,
        train_csr=train_csr,
        train_items_csr=train_items_csr,
        test_pos=test_pos,
        eval_users=eval_users,
        item_pop=item_pop,
        trend_score=trend_score,
        item_text=item_text,
        item_type=item_type,
        type_names=type_names,
        stats=stats,
    )
