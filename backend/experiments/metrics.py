"""Ranking-quality and beyond-accuracy metrics.

All accuracy metrics are computed per user at cutoff K so that paired
significance tests (Wilcoxon signed-rank) can be run across users.
"""
import numpy as np

K = 10


def user_accuracy_metrics(topk: np.ndarray, relevant: np.ndarray) -> dict:
    """topk: (K,) recommended item ids, ranked. relevant: ground-truth item ids."""
    rel_set = set(relevant.tolist())
    hits = np.array([1.0 if it in rel_set else 0.0 for it in topk])
    n_rel = len(rel_set)
    n_hits = hits.sum()

    precision = n_hits / K
    recall = n_hits / n_rel if n_rel else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    hr = 1.0 if n_hits > 0 else 0.0

    ranks = np.arange(1, K + 1)
    dcg = (hits / np.log2(ranks + 1)).sum()
    ideal_hits = min(n_rel, K)
    idcg = (1.0 / np.log2(np.arange(1, ideal_hits + 1) + 1)).sum()
    ndcg = dcg / idcg if idcg else 0.0

    # average precision @ K
    cum_hits = np.cumsum(hits)
    ap = (hits * cum_hits / ranks).sum() / min(n_rel, K) if n_rel else 0.0

    return {
        "Precision@10": precision,
        "Recall@10": recall,
        "F1@10": f1,
        "HR@10": hr,
        "NDCG@10": ndcg,
        "MAP@10": ap,
    }


def novelty(topk: np.ndarray, item_pop: np.ndarray, total_interactions: float) -> float:
    """Mean self-information -log2(p(i)) of recommended items (Zhou et al., 2010)."""
    p = np.maximum(item_pop[topk], 1.0) / total_interactions
    return float(np.mean(-np.log2(p)))


def intra_list_diversity(topk: np.ndarray, item_emb: np.ndarray) -> float:
    """1 - mean pairwise cosine similarity of recommended items' content embeddings."""
    e = item_emb[topk]
    norms = np.linalg.norm(e, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    e = e / norms
    sim = e @ e.T
    n = len(topk)
    off_diag = (sim.sum() - np.trace(sim)) / (n * (n - 1))
    return float(1.0 - off_diag)


def category_diversity(topk: np.ndarray, item_type: np.ndarray) -> float:
    """Fraction of distinct product types in the list (paper's diversity measure)."""
    return float(len(set(item_type[topk].tolist())) / len(topk))


def aggregate(per_user: list) -> dict:
    """Mean of each metric over users."""
    keys = per_user[0].keys()
    return {k: float(np.mean([m[k] for m in per_user])) for k in keys}
