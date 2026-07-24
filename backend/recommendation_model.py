import os

from sqlalchemy import bindparam, create_engine, text
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from collections import defaultdict
import time

DB_URL = os.environ.get(
    "DATABASE_URL", "mysql+mysqlconnector://root:mysqlZ97*@localhost/dataset_db"
)

# Hybrid component weights (content / collaborative / trend). The 10/70/20
# split is the best three-component configuration found in the offline
# evaluation (backend/experiments/results/tables.md): collaborative ItemKNN
# carries most of the ranking signal, trend acts as a tie-breaker, and a
# small content weight is kept for cold-start coverage on new products.
HYBRID_WEIGHTS = {"content": 0.10, "collaborative": 0.70, "trend": 0.20}

# Content-based filtering thresholds (see paper, Methodology)
COSINE_THRESHOLD = 0.2
PRICE_BAND = 0.5  # candidate price within +/-50% of the reference product

def get_db():
    try:
        engine = create_engine(DB_URL)
        return engine
    except Exception as e:
        print(f"Database connection error: {str(e)}")
        raise e

def get_recommendations(product_id, num_recommendations=4):
    recommender = None
    try:
        print(f"Starting recommendations for product: {product_id}")
        engine = get_db()
        recommender = AdvancedRecommendationEngine(engine)
        recommendations = recommender.get_hybrid_recommendations(product_id, num_recommendations)
        print(f"Found {len(recommendations)} hybrid recommendations")
        return recommendations
    except Exception as e:
        print(f"Error in get_recommendations: {str(e)}")
        import traceback
        traceback.print_exc()
        if recommender is None:
            return []
        return recommender.get_similar_products(product_id, num_recommendations)

def get_recommendations_with_images(product_id):
    try:
        print(f"Getting hybrid recommendations with images for product: {product_id}")
        recommendations = get_recommendations(product_id)

        if not recommendations:
            print("No recommendations found")
            return []

        print(f"Processing images for {len(recommendations)} recommendations")

        for rec in recommendations:
            try:
                if not rec.get('ImageURL') or 'placeholder' in rec['ImageURL']:
                    query = text("""
                        SELECT ImageURL, URL
                        FROM products
                        WHERE ProductId = :product_id
                    """)

                    with get_db().connect() as conn:
                        result = conn.execute(query, {"product_id": rec['ProductId']}).fetchone()

                        if result and result.ImageURL:
                            rec['ImageURL'] = result.ImageURL
                        else:
                            rec['ImageURL'] = "http://localhost:5001/static/images/product-placeholder.jpg"

            except Exception as e:
                print(f"Error processing image for product {rec['ProductId']}: {str(e)}")
                rec['ImageURL'] = "http://localhost:5001/static/images/product-placeholder.jpg"

        return recommendations

    except Exception as e:
        print(f"Error in get_recommendations_with_images: {str(e)}")
        import traceback
        traceback.print_exc()
        return []

class AdvancedRecommendationEngine:
    def __init__(self, engine):
        self.engine = engine
        self.tfidf = TfidfVectorizer(
            analyzer='word',
            ngram_range=(1, 2),
            min_df=0.01,
            max_features=1000,
            stop_words='english'
        )
        self.memory_cache = {}
        self.trend_cache = {}
        self.cache_expiry = 300

    def _build_content_index(self):
        """Fit TF-IDF over product type + title once per engine instance."""
        query = text("SELECT ProductId, ProductTitle, ProductType, price FROM products")
        with self.engine.connect() as conn:
            rows = conn.execute(query).fetchall()
        ids, texts, prices = [], [], []
        for r in rows:
            ids.append(r.ProductId)
            texts.append(f"{r.ProductType or ''} {r.ProductTitle or ''}".strip())
            prices.append(float(r.price) if r.price else np.nan)
        self._content_ids = np.array(ids)
        self._content_prices = np.array(prices, dtype=np.float64)
        self._content_pos = {pid: i for i, pid in enumerate(ids)}
        self._content_matrix = normalize(self.tfidf.fit_transform(texts))
        print(f"Content index built over {len(ids)} products")

    def _fetch_products(self, product_ids):
        """Metadata + rating aggregates for a list of product ids, keyed by id."""
        if not product_ids:
            return {}
        query = text("""
            SELECT
                p.ProductId, p.ProductType, p.ProductTitle, p.ImageURL, p.price,
                COALESCE(ps.avg_rating, 0) as avg_rating,
                COALESCE(ps.review_count, 0) as review_count
            FROM products p
            LEFT JOIN product_stats ps ON ps.ProductId = p.ProductId
            WHERE p.ProductId IN :ids
        """).bindparams(bindparam("ids", expanding=True))
        with self.engine.connect() as conn:
            rows = conn.execute(query, {"ids": list(product_ids)}).fetchall()
        return {
            row.ProductId: {
                "ProductId": row.ProductId,
                "ProductType": row.ProductType,
                "ProductTitle": row.ProductTitle,
                "ImageURL": row.ImageURL or "http://localhost:5001/static/images/product-placeholder.jpg",
                "price": float(row.price) if row.price else 0.0,
                "Rating": float(row.avg_rating),
                "ReviewCount": row.review_count,
            }
            for row in rows
        }

    def get_similar_products(self, product_id, num_recommendations=4):
        """Content-based filtering: TF-IDF cosine similarity (paper Eqs. 1-2)
        with the 0.2 cosine threshold and +/-50% price band."""
        try:
            if not hasattr(self, "_content_matrix"):
                self._build_content_index()
            pos = self._content_pos.get(product_id)
            if pos is None:
                return []

            sims = np.asarray(
                (self._content_matrix[pos] @ self._content_matrix.T).todense()
            ).ravel()
            sims[pos] = 0.0

            ref_price = self._content_prices[pos]
            if np.isfinite(ref_price) and ref_price > 0:
                in_band = (
                    np.isfinite(self._content_prices)
                    & (self._content_prices >= ref_price * (1 - PRICE_BAND))
                    & (self._content_prices <= ref_price * (1 + PRICE_BAND))
                )
                sims = np.where(in_band, sims, 0.0)
            sims = np.where(sims >= COSINE_THRESHOLD, sims, 0.0)

            n_cand = min(num_recommendations * 3, np.count_nonzero(sims))
            if n_cand == 0:
                return []
            top = np.argpartition(sims, -n_cand)[-n_cand:]
            top = top[np.argsort(-sims[top])]
            meta = self._fetch_products(self._content_ids[top].tolist())

            recommended = []
            for idx in top:
                pid = self._content_ids[idx]
                product = meta.get(pid)
                if product is None or product["Rating"] < 4.0:
                    continue
                product = dict(product, similarity_score=float(sims[idx]))
                recommended.append(product)
            return recommended[:num_recommendations]

        except Exception as e:
            print(f"Error in get_similar_products: {str(e)}")
            import traceback
            traceback.print_exc()
            return []

    def get_collaborative_recommendations(self, product_id, num_recommendations=4):
        """Item-based collaborative filtering. Neighbours and their cosine
        similarities are precomputed by scripts/build_item_similarity.py
        (ItemKNN, k=200, over positive ratings). This replaces the earlier
        fixed collaborative score of 0.8, which offline evaluation showed to
        be ~14x weaker in NDCG@10 (backend/experiments/results/tables.md)."""
        try:
            query = text("""
                SELECT
                    p.ProductId, p.ProductType, p.ProductTitle, p.ImageURL, p.price,
                    s.Score as collaborative_score,
                    ps.avg_rating, ps.review_count
                FROM item_similarity s
                JOIN products p ON p.ProductId = s.NeighborId
                JOIN product_stats ps ON ps.ProductId = s.NeighborId
                WHERE s.ProductId = :product_id
                  AND ps.avg_rating >= 4.0
                ORDER BY s.Score DESC
                LIMIT :limit
            """)

            with self.engine.connect() as conn:
                results = conn.execute(query, {
                    "product_id": product_id,
                    "limit": num_recommendations
                }).fetchall()

                return [{
                    "ProductId": row.ProductId,
                    "ProductType": row.ProductType,
                    "ProductTitle": row.ProductTitle,
                    "ImageURL": row.ImageURL or "http://localhost:5001/static/images/product-placeholder.jpg",
                    "price": float(row.price) if row.price else 0.0,
                    "Rating": float(row.avg_rating),
                    "ReviewCount": row.review_count,
                    "similarity_score": float(row.collaborative_score)
                } for row in results]

        except Exception as e:
            print(f"Error in collaborative recommendations: {str(e)}")
            return []

    def get_trending_recommendations(self, product_id, num_recommendations=4):

        try:
            cache_key = f"trend_{product_id}_{num_recommendations}"
            current_time = time.time()

            if cache_key in self.trend_cache:
                cached_data, timestamp = self.trend_cache[cache_key]
                if current_time - timestamp < self.cache_expiry:
                    return cached_data

            query = text("""
                SELECT
                    p.ProductId, p.ProductType, p.ProductTitle, p.ImageURL, p.price,
                    ps.review_count, ps.avg_rating, ps.trend_score
                FROM products p
                JOIN product_stats ps ON ps.ProductId = p.ProductId
                WHERE p.ProductType = (
                    SELECT ProductType FROM products WHERE ProductId = :product_id
                )
                  AND p.ProductId != :product_id
                  AND ps.avg_rating >= 4.0
                  AND ps.review_count >= 10
                ORDER BY ps.trend_score DESC, ps.review_count DESC, ps.avg_rating DESC
                LIMIT :limit
            """)

            with self.engine.connect() as conn:
                results = conn.execute(query, {
                    "product_id": product_id,
                    "limit": num_recommendations
                }).fetchall()

                recommendations = [{
                    "ProductId": row.ProductId,
                    "ProductType": row.ProductType,
                    "ProductTitle": row.ProductTitle,
                    "ImageURL": row.ImageURL or "http://localhost:5001/static/images/product-placeholder.jpg",
                    "price": float(row.price) if row.price else 0.0,
                    "Rating": float(row.avg_rating),
                    "ReviewCount": row.review_count,
                    "TrendScore": float(row.trend_score),
                    "similarity_score": min(float(row.trend_score) / 1000, 1.0)
                } for row in results]

                self.trend_cache[cache_key] = (recommendations, current_time)
                return recommendations

        except Exception as e:
            print(f"Error in trending recommendations: {str(e)}")
            return []

    @staticmethod
    def _normalized_scores(recommendations):
        """Min-max normalise a component's similarity scores to [0, 1] so the
        three components mix on a comparable scale (paper Eq. 5)."""
        if not recommendations:
            return {}
        scores = [r["similarity_score"] for r in recommendations]
        lo, hi = min(scores), max(scores)
        span = (hi - lo) or 1.0
        return {
            r["ProductId"]: (r["similarity_score"] - lo) / span if hi > lo else 1.0
            for r in recommendations
        }

    def get_hybrid_recommendations(self, product_id, num_recommendations=4):
        try:
            pool = num_recommendations * 3
            similar_products = self.get_similar_products(product_id, pool)
            collaborative_recs = self.get_collaborative_recommendations(product_id, pool)
            trending_recs = self.get_trending_recommendations(product_id, pool)

            product_scores = defaultdict(float)
            for component, recs in (
                ("content", similar_products),
                ("collaborative", collaborative_recs),
                ("trend", trending_recs),
            ):
                weight = HYBRID_WEIGHTS[component]
                for pid, score in self._normalized_scores(recs).items():
                    product_scores[pid] += weight * score

            # graceful degradation: fall back to content-based when the
            # behavioural components have no data (e.g. brand-new product)
            if not product_scores:
                return self.get_similar_products(product_id, num_recommendations)

            top_products = sorted(
                product_scores.items(), key=lambda x: x[1], reverse=True
            )[:num_recommendations]

            meta = self._fetch_products([pid for pid, _ in top_products])
            final_recommendations = []
            for prod_id, score in top_products:
                product = meta.get(prod_id)
                if product:
                    final_recommendations.append(
                        dict(product, similarity_score=float(score))
                    )
            return final_recommendations

        except Exception as e:
            print(f"Error in hybrid recommendations: {str(e)}")
            return self.get_similar_products(product_id, num_recommendations)
