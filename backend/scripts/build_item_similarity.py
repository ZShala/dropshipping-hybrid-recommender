"""Precompute item-item cosine similarities (ItemKNN) for the live engine.

Replaces the fixed collaborative score (0.8) with genuine item-based
collaborative filtering: for every product, the top-k most similar products
by cosine similarity over the binary user-item matrix of positive ratings
(rating >= 4.0). Offline evaluation (backend/experiments/) showed this
improves NDCG@10 by ~14x over the fixed-score proxy.

Reads ratings from the database if reachable, otherwise from the raw CSV.
Writes the `item_similarity` table (ProductId, NeighborId, Score, Rank).

Usage:
    python scripts/build_item_similarity.py                  # default MySQL
    python scripts/build_item_similarity.py --db-url sqlite:///dev.db
    python scripts/build_item_similarity.py --csv ../data/Amazon_Beauty_Recommendation.csv
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import normalize
from sqlalchemy import create_engine, text

DEFAULT_DB_URL = os.environ.get(
    "DATABASE_URL", "mysql+mysqlconnector://root:mysqlZ97*@localhost/dataset_db"
)
DEFAULT_CSV = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data", "Amazon_Beauty_Recommendation.csv",
)
POSITIVE_THRESHOLD = 4.0
TOP_K = 200
CHUNK = 2000


def load_ratings(engine, csv_path):
    if engine is not None:
        try:
            df = pd.read_sql(
                text("SELECT UserId, ProductId, Rating FROM amazon_beauty"), engine
            )
            print(f"Loaded {len(df):,} ratings from database")
            return df
        except Exception as e:
            print(f"Database read failed ({e}); falling back to CSV")
    df = pd.read_csv(csv_path, usecols=["UserId", "ProductId", "Rating"])
    print(f"Loaded {len(df):,} ratings from {csv_path}")
    return df


def compute_topk_similarity(df, k=TOP_K):
    pos = df[df["Rating"] >= POSITIVE_THRESHOLD]
    users = pos["UserId"].astype("category")
    items = pos["ProductId"].astype("category")
    item_ids = np.array(items.cat.categories)
    m = sparse.csr_matrix(
        (np.ones(len(pos), dtype=np.float32),
         (users.cat.codes.values, items.cat.codes.values)),
        shape=(users.cat.categories.size, item_ids.size),
    )
    print(f"Positive matrix: {m.shape[0]:,} users x {m.shape[1]:,} items, "
          f"{m.nnz:,} positives")

    mn = normalize(m, axis=0).tocsc()
    mt = mn.T.tocsr()
    rows = []
    for start in range(0, item_ids.size, CHUNK):
        block = (mt[start : start + CHUNK] @ mn).tocsr()
        for r in range(block.shape[0]):
            item_idx = start + r
            s, e = block.indptr[r], block.indptr[r + 1]
            cols, vals = block.indices[s:e], block.data[s:e]
            keep = cols != item_idx                     # drop self-similarity
            cols, vals = cols[keep], vals[keep]
            if len(vals) > k:
                top = np.argpartition(vals, -k)[-k:]
                cols, vals = cols[top], vals[top]
            order = np.argsort(-vals)
            for rank, (c, v) in enumerate(zip(cols[order], vals[order]), start=1):
                rows.append((item_ids[item_idx], item_ids[c], float(v), rank))
    out = pd.DataFrame(rows, columns=["ProductId", "NeighborId", "Score", "Rank"])
    print(f"Computed {len(out):,} similarity pairs "
          f"({len(out)/item_ids.size:.0f} neighbours/item avg)")
    return out


def build_product_stats(engine):
    """Materialise per-product rating aggregates so the live queries never
    re-aggregate the full ratings table per request."""
    with engine.connect() as conn:
        conn.execute(text("DROP TABLE IF EXISTS product_stats"))
        conn.execute(text("""
            CREATE TABLE product_stats AS
            SELECT ProductId,
                   COUNT(DISTINCT UserId) AS review_count,
                   AVG(Rating) AS avg_rating,
                   COUNT(DISTINCT UserId) * AVG(Rating) AS trend_score
            FROM amazon_beauty
            GROUP BY ProductId
        """))
        if engine.dialect.name == "mysql":
            conn.execute(text(
                "ALTER TABLE product_stats ADD INDEX idx_ps_product (ProductId(20)), "
                "ADD INDEX idx_ps_trend (trend_score)"
            ))
        else:
            conn.execute(text(
                "CREATE INDEX IF NOT EXISTS idx_ps_product ON product_stats (ProductId)"
            ))
        conn.commit()
    print("Wrote product_stats table and indexes")


def write_table(engine, sim_df):
    sim_df.to_sql("item_similarity", engine, if_exists="replace", index=False,
                  chunksize=10000)
    with engine.connect() as conn:
        dialect = engine.dialect.name
        if dialect == "mysql":
            conn.execute(text(
                "ALTER TABLE item_similarity ADD INDEX idx_sim_product (ProductId(255))"
            ))
        else:
            conn.execute(text(
                "CREATE INDEX IF NOT EXISTS idx_sim_product ON item_similarity (ProductId)"
            ))
        if hasattr(conn, "commit"):
            conn.commit()
    print("Wrote item_similarity table and index")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db-url", default=DEFAULT_DB_URL)
    ap.add_argument("--csv", default=DEFAULT_CSV)
    ap.add_argument("--k", type=int, default=TOP_K)
    ap.add_argument("--csv-out", default=None,
                    help="also write the similarity pairs to this CSV path")
    args = ap.parse_args()

    engine = None
    try:
        engine = create_engine(args.db_url)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception as e:
        print(f"Cannot connect to {args.db_url}: {e}")
        engine = None

    t0 = time.time()
    df = load_ratings(engine, args.csv)
    sim = compute_topk_similarity(df, k=args.k)
    if args.csv_out:
        sim.to_csv(args.csv_out, index=False)
        print(f"Wrote {args.csv_out}")
    if engine is not None:
        write_table(engine, sim)
        build_product_stats(engine)
    elif not args.csv_out:
        print("No database and no --csv-out given; nothing written", file=sys.stderr)
        sys.exit(1)
    print(f"Done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
