# Offline Experiment Suite

Reproducible evaluation pipeline for the paper *Enhancing Dropshipping
Performance Through Hybrid Recommendation Engines* (IJITIS submission 570,
major revision).

## Protocol

- **Data**: `data/Amazon_Beauty_Recommendation.csv` — Kaggle
  [satrapankti/amazon-beauty-product-recommendation](https://www.kaggle.com/datasets/satrapankti/amazon-beauty-product-recommendation);
  1,348,246 ratings, 883,753 users, 23,838 products, 22 product types
  (1999–2014). Sparsity 99.994 %.
- **Split**: per-user temporal 80/20 — each user's interactions are ordered by
  timestamp and the most recent 20 % are held out. Training uses all
  interactions from all users; evaluation is restricted to the 23,952 users
  with ≥ 5 interactions, ≥ 2 training positives and ≥ 1 held-out positive.
  A *positive* is a rating ≥ 4.0; only held-out positives count as relevant.
- **Ranking**: full-catalogue ranking (all 23,838 items minus the user's
  training items), cutoff k = 10.
- **Repetition**: 5 independent runs; each samples 5,000 evaluation users and
  reseeds the stochastic models (PureSVD, BPR). Reported as mean ± std.
- **Significance**: per-user paired Wilcoxon signed-rank on pooled runs
  (n = 25,000), Holm-Bonferroni corrected.
- **Hyperparameters**: tuned on a validation split carved from the training
  portion (never the test set): ItemKNN k = 200; PureSVD f = 256;
  BPR f = 64, lr = 0.05, 300 iterations. See `results/tuning.json`.

## Models

| Key | Model |
|---|---|
| T | Trend/popularity: `review_count × avg_rating`, ≥ 10 reviews, avg > 4.0 (paper Eq. 4) |
| C | Content-based: TF-IDF over product type + URL slug, max cosine to any owned item |
| P | Fixed collaborative proxy (0.8) — replication of the originally submitted system |
| CF | ItemKNN — item-based CF, cosine similarity, top-200 neighbours |
| SVD | PureSVD (Cremonesi et al., 2010), 256 factors |
| BPR | Bayesian Personalised Ranking (Rendle et al., 2009) via `implicit` |
| Hybrid c/cf/t | per-user min-max normalised weighted sum of C, CF, T |

## Running

```bash
cd backend
python -m experiments.run_experiments --tune       # hyperparameter search (validation split)
python -m experiments.run_experiments --seeds 5 --users 5000
python -m experiments.make_tables                  # aggregate -> results/tables.md
```

Requires: `pandas`, `numpy`, `scipy`, `scikit-learn`, `implicit`
(Python 3.10; no database needed — runs directly from the CSV).

Outputs in `results/`: `dataset_stats.json`, `tuning.json`,
`summary_seed*.json`, `per_user_seed*.npz` (per-user metric arrays for
significance testing), `tables.md`, `sensitivity_grid.csv`.
