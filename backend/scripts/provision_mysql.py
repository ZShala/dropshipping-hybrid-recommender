"""Provision the production MySQL database from the raw CSV.

Creates and populates:
  - amazon_beauty   (all rating interactions, as in preprocess_and_load.py)
  - products        (one row per product; title derived from the Amazon URL
                     slug; price is a deterministic placeholder unless real
                     supplier prices are loaded separately)
plus the indexes described in the paper (B-tree on ProductId/UserId, composite
on (ProductType, price)).

Usage:  python scripts/provision_mysql.py [--db-url URL] [--csv PATH]
"""
import argparse
import hashlib
import os
import re
import time
import urllib.parse

import pandas as pd
from sqlalchemy import create_engine, text

DEFAULT_DB_URL = os.environ.get(
    "DATABASE_URL", "mysql+mysqlconnector://root:mysqlZ97*@localhost/dataset_db"
)
DEFAULT_CSV = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data", "Amazon_Beauty_Recommendation.csv",
)


def slug(url):
    m = re.search(r"amazon\.[a-z.]+/([^/]+)/dp/", str(url))
    if not m:
        return ""
    return urllib.parse.unquote(m.group(1)).replace("-", " ").replace("_", " ")


def pseudo_price(pid):
    """Deterministic placeholder price in [5.00, 50.00]; replace with real
    supplier prices if a dump of the original products table is available."""
    h = int(hashlib.md5(pid.encode()).hexdigest()[:8], 16)
    return round(5 + (h % 4500) / 100, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db-url", default=DEFAULT_DB_URL)
    ap.add_argument("--csv", default=DEFAULT_CSV)
    args = ap.parse_args()

    t0 = time.time()
    df = pd.read_csv(args.csv)
    print(f"Loaded {len(df):,} rows from CSV")

    engine = create_engine(args.db_url)
    df.to_sql("amazon_beauty", engine, if_exists="replace", index=False,
              chunksize=20000)
    print(f"amazon_beauty written ({time.time() - t0:.0f}s)")

    prod = df.drop_duplicates("ProductId")[["ProductId", "ProductType", "URL"]].copy()
    prod["ProductTitle"] = prod["URL"].map(slug)
    prod["ImageURL"] = None
    prod["price"] = prod["ProductId"].map(pseudo_price)
    prod[["ProductId", "ProductTitle", "ProductType", "ImageURL", "price"]].to_sql(
        "products", engine, if_exists="replace", index=False
    )
    print(f"products written ({len(prod):,} rows)")

    with engine.connect() as conn:
        conn.execute(text(
            "ALTER TABLE amazon_beauty ADD INDEX idx_ab_product (ProductId(20)), "
            "ADD INDEX idx_ab_user (UserId(20)), ADD INDEX idx_ab_rating (Rating)"
        ))
        conn.execute(text(
            "ALTER TABLE products ADD INDEX idx_p_product (ProductId(20)), "
            "ADD INDEX idx_p_type_price (ProductType(50), price)"
        ))
        conn.commit()
    print(f"indexes created; done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
