"""Recommendation models for the offline evaluation.

Every model implements:
    fit(dataset, seed)                       -> self
    score_users(user_indices) -> np.ndarray  (len(users), n_items) float32

Scores are only used for ranking, so their absolute scale is irrelevant;
the hybrid normalises component scores per user before mixing.
"""
import time

import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


class Popularity:
    """Trend-based baseline: review_count * avg_rating (paper Eq. 4)."""

    name = "Trend (popularity)"
    deterministic = True

    def fit(self, dataset, seed=0):
        self.scores = dataset.trend_score.astype(np.float32)
        self.n_items = dataset.n_items
        return self

    def score_users(self, users):
        return np.tile(self.scores, (len(users), 1))


class ContentBased:
    """TF-IDF over product type + URL slug. Mirrors the system's item-to-item
    similarity: an item's score for a user is its maximum cosine similarity to
    any of the user's training positives ("similar to something you bought")."""

    name = "Content-based (TF-IDF)"
    deterministic = True

    def __init__(self, max_features=5000):
        self.max_features = max_features

    def fit(self, dataset, seed=0):
        vec = TfidfVectorizer(
            stop_words="english", max_features=self.max_features, sublinear_tf=True
        )
        tfidf = vec.fit_transform(dataset.item_text)
        self.item_vecs = normalize(tfidf).astype(np.float32).tocsr()
        self.item_vecs_t = self.item_vecs.T.tocsc()
        self.train_csr = dataset.train_csr
        return self

    def score_users(self, users):
        out = np.zeros((len(users), self.item_vecs.shape[0]), dtype=np.float32)
        for row, u in enumerate(users):
            items = self.train_csr[u].indices
            if len(items):
                sims = self.item_vecs[items] @ self.item_vecs_t   # (n_i, n_items)
                out[row] = sims.max(axis=0).toarray().ravel()
        return out


class ItemKNN:
    """Item-based collaborative filtering with cosine similarity (top-k pruned)."""

    deterministic = True

    def __init__(self, k=100, chunk=2000):
        self.k = k
        self.chunk = chunk
        self.name = f"ItemKNN (k={k})"

    def fit(self, dataset, seed=0):
        # With L2-normalised item columns, the matrix product Mᵀ·M yields the
        # full item-item cosine similarity matrix. Computed in row chunks so
        # the intermediate never materialises all ~24k x 24k entries at once;
        # each chunk is pruned to the k strongest neighbours before stacking.
        m = normalize(dataset.train_csr, axis=0).tocsc()       # L2 per item column
        n_items = dataset.n_items
        blocks = []
        mt = m.T.tocsr()
        for start in range(0, n_items, self.chunk):
            block = (mt[start : start + self.chunk] @ m).tocsr()
            # zero self-similarity (an item is trivially similar to itself);
            # indptr[r]:indptr[r+1] is row r's slice of the CSR data arrays
            for r in range(block.shape[0]):
                item = start + r
                s, e = block.indptr[r], block.indptr[r + 1]
                cols = block.indices[s:e]
                block.data[s:e][cols == item] = 0.0
            block = _prune_topk(block, self.k)
            blocks.append(block)
        self.sim = sparse.vstack(blocks).tocsr()
        self.train_csr = dataset.train_csr
        return self

    def score_users(self, users):
        return np.asarray((self.train_csr[users] @ self.sim).todense(), dtype=np.float32)


def _prune_topk(csr_block, k):
    """Keep the k largest entries per row of a CSR matrix.

    Works directly on the CSR representation: for each row, slice its
    values/column-indices out of the flat data arrays, keep the k largest,
    and rebuild the three CSR arrays (data, indices, indptr) from the kept
    pieces. indptr[r] is where row r starts in the flat arrays."""
    data, indices, indptr = [], [], [0]
    for r in range(csr_block.shape[0]):
        s, e = csr_block.indptr[r], csr_block.indptr[r + 1]
        row_data = csr_block.data[s:e]
        row_idx = csr_block.indices[s:e]
        if len(row_data) > k:
            top = np.argpartition(row_data, -k)[-k:]
            row_data, row_idx = row_data[top], row_idx[top]
        data.append(row_data)
        indices.append(row_idx)
        indptr.append(indptr[-1] + len(row_data))
    return sparse.csr_matrix(
        (np.concatenate(data), np.concatenate(indices), np.array(indptr)),
        shape=csr_block.shape,
    )


class CFProxy:
    """Replication of the paper's original 'collaborative signal': every item
    co-rated (rating >= 4) with any of the user's items receives the fixed
    score 0.8; all other items receive 0."""

    name = "CF proxy (fixed 0.8)"
    deterministic = True

    def fit(self, dataset, seed=0):
        self.train_csr = dataset.train_csr
        return self

    def score_users(self, users):
        # Two sparse matrix products implement "items rated >=4 by anyone who
        # co-rated one of my items":
        #   u @ Rᵀ  -> which other users share a positive item with me
        #   ... @ R -> which items those co-raters rated positively
        # Any item reached this way gets the paper's fixed 0.8; all else 0.
        u = self.train_csr[users]                      # (n, items)
        co_users = u @ self.train_csr.T                # (n, users) co-raters
        co_users.data = np.ones_like(co_users.data)
        counts = co_users @ self.train_csr             # (n, items)
        scores = np.asarray(counts.todense(), dtype=np.float32)
        out = np.where(scores > 0, np.float32(0.8), np.float32(0.0))
        # remove trivial self-hits: user's own items are masked later anyway
        return out


class PureSVD:
    """PureSVD (Cremonesi et al., 2010): truncated SVD of the binary matrix."""

    def __init__(self, factors=64):
        self.factors = factors
        self.name = f"PureSVD (f={factors})"

    deterministic = False

    def fit(self, dataset, seed=0):
        svd = TruncatedSVD(n_components=self.factors, random_state=seed)
        svd.fit(dataset.train_csr)
        self.vt = svd.components_.astype(np.float32)   # (f, items)
        self.train_csr = dataset.train_csr
        return self

    def score_users(self, users):
        proj = self.train_csr[users] @ self.vt.T       # (n, f)
        return np.asarray(proj @ self.vt, dtype=np.float32)


class BPR:
    """Bayesian Personalised Ranking (Rendle et al., 2009) via `implicit`."""

    deterministic = False

    def __init__(self, factors=64, learning_rate=0.01, regularization=0.01, iterations=100):
        self.params = dict(
            factors=factors,
            learning_rate=learning_rate,
            regularization=regularization,
            iterations=iterations,
        )
        self.name = f"BPR (f={factors})"

    def fit(self, dataset, seed=0):
        from implicit.bpr import BayesianPersonalizedRanking

        model = BayesianPersonalizedRanking(random_state=seed, verify_negative_samples=True,
                                            **self.params)
        model.fit(dataset.train_csr, show_progress=False)
        self.user_factors = model.user_factors
        self.item_factors = model.item_factors
        return self

    def score_users(self, users):
        return np.asarray(
            self.user_factors[users] @ self.item_factors.T, dtype=np.float32
        )


def fit_timed(model, dataset, seed):
    t0 = time.perf_counter()
    model.fit(dataset, seed)
    return model, time.perf_counter() - t0
