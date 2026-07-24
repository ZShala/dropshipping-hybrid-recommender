"""Chunked evaluation engine.

Component models are scored once per user chunk; every evaluated *system*
(standalone model or weighted hybrid over normalised component scores) then
reuses those matrices, which makes large ablation/sensitivity grids cheap.
"""
import numpy as np

from . import metrics


def _minmax_rows(x: np.ndarray) -> np.ndarray:
    lo = x.min(axis=1, keepdims=True)
    hi = x.max(axis=1, keepdims=True)
    rng = np.where(hi - lo > 0, hi - lo, 1.0)
    return (x - lo) / rng


def evaluate_systems(
    dataset,
    users: np.ndarray,
    component_models: dict,
    systems: list,
    item_emb: np.ndarray,
    chunk_size: int = 500,
    k: int = metrics.K,
):
    """
    component_models: {key: fitted model}
    systems: [{"name": str, "model": key} | {"name": str, "weights": {key: w}}]
    Returns {system_name: {"per_user": {metric: np.ndarray}, "summary": {...}}}
    """
    total_interactions = float(dataset.item_pop.sum())
    names = [s["name"] for s in systems]
    per_user = {n: [] for n in names}
    extra = {n: {"novelty": [], "ild": [], "cat_div": []} for n in names}
    rec_items = {n: set() for n in names}

    for start in range(0, len(users), chunk_size):
        cu = users[start : start + chunk_size]
        raw = {key: m.score_users(cu) for key, m in component_models.items()}
        norm = {key: _minmax_rows(v) for key, v in raw.items()}

        # mask training items (both positives and non-positive interactions)
        mask = dataset.train_items_csr[cu]

        for spec in systems:
            if "model" in spec:
                scores = raw[spec["model"]].copy()
            else:
                scores = np.zeros_like(next(iter(norm.values())))
                for key, w in spec["weights"].items():
                    if w:
                        scores += np.float32(w) * norm[key]
            rows, cols = mask.nonzero()
            scores[rows, cols] = -1e9

            # Top-k extraction in two steps: argpartition finds the k best
            # items per row in O(n_items) but returns them unordered, so a
            # second argsort over just those k columns produces the ranking.
            part = np.argpartition(scores, -k, axis=1)[:, -k:]
            row_scores = np.take_along_axis(scores, part, axis=1)
            order = np.argsort(-row_scores, axis=1)
            topk = np.take_along_axis(part, order, axis=1)      # (n, k) ranked

            # vectorised beyond-accuracy metrics
            # Novelty: mean self-information -log2(p(item)) of the list.
            pop = np.maximum(dataset.item_pop[topk], 1.0) / total_interactions
            extra[spec["name"]]["novelty"].append(-np.log2(pop).mean(axis=1))
            # Intra-list diversity = 1 - mean pairwise cosine similarity of
            # the k recommended items' content embeddings. Batched: for each
            # user, e[user] is a (k, dim) matrix, so e @ e^T is that user's
            # (k, k) similarity matrix; subtracting the trace (k ones) and
            # dividing by k(k-1) averages the off-diagonal pairs.
            e = item_emb[topk]
            norms = np.linalg.norm(e, axis=2, keepdims=True)
            e = e / np.where(norms > 0, norms, 1.0)
            sims = e @ e.transpose(0, 2, 1)
            n_ = sims.shape[1]
            ild = 1.0 - (sims.sum(axis=(1, 2)) - n_) / (n_ * (n_ - 1))
            extra[spec["name"]]["ild"].append(ild)
            cats = dataset.item_type[topk]
            cat_div = np.array([len(np.unique(r)) / k for r in cats])
            extra[spec["name"]]["cat_div"].append(cat_div)
            rec_items[spec["name"]].update(np.unique(topk).tolist())

            for row, u in enumerate(cu):
                per_user[spec["name"]].append(
                    metrics.user_accuracy_metrics(topk[row], dataset.test_pos[u])
                )

    results = {}
    for n in names:
        acc_keys = per_user[n][0].keys()
        pu = {key: np.array([m[key] for m in per_user[n]]) for key in acc_keys}
        summary = {key: float(v.mean()) for key, v in pu.items()}
        summary["Novelty"] = float(np.concatenate(extra[n]["novelty"]).mean())
        summary["ILD"] = float(np.concatenate(extra[n]["ild"]).mean())
        summary["CategoryDiversity"] = float(np.concatenate(extra[n]["cat_div"]).mean())
        summary["Coverage"] = float(len(rec_items[n]) / dataset.n_items)
        results[n] = {"per_user": pu, "summary": summary}
    return results
