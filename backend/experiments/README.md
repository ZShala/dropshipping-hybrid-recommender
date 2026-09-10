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
  BPR f = 128, lr = 0.05, 300 iterations. See `results/tuning.json`.
- **Reported hybrid**: the tables report the configuration used in the paper,
  10/70/20 (content/CF/trend). The grid maximum is 0/80/20, but the two are
  statistically indistinguishable (Δ NDCG@10 = 0.00002) and 10/70/20 retains a
  content weight for the cold-item fallback, so `make_tables.py` pins it.

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
python -m experiments.analyse_fallback             # catalogue reachability per component
```

### Second-revision analyses

Each answers a specific point from the second review and writes one JSON to
`results/`. They read the outputs of the main run above, so run that first.

```bash
python -m experiments.analyse_effect_sizes         # -> effect_sizes.json
python -m experiments.analyse_ecology              # -> ecology_stats.json
python -m experiments.analyse_metadata_quality     # -> metadata_quality.json
python -m experiments.analyse_sparsity  --seeds 3  # -> sparsity_sweep.json   (~35 min)
python -m experiments.analyse_churn     --seeds 3  # -> churn_stress.json     (~10 min)
python -m experiments.analyse_protocol  --seeds 3  # -> protocol_sensitivity.json (~12 min)
```

| Script | Question it answers | Feeds |
|---|---|---|
| `analyse_effect_sizes` | Is the hybrid's edge over ItemKNN large enough to matter? Cohen's dz, rank-biserial, bootstrap CI, win/tie/loss, and an operational rate | Practical Significance |
| `analyse_sparsity` | How does the model ordering change as data thins? Random, per-user and per-item thinning, plus a no-refit stratification by user profile length | Table 6 |
| `analyse_churn` | Does the content component actually rescue new products? Removes all history from a share of items and scores the cold ones separately | Table 7 |
| `analyse_protocol` | Do the results survive the two arbitrary constants (relevance threshold, eligibility floor)? | Table 8 |
| `analyse_ecology` | How far does the dataset reproduce dropshipping conditions? Catalogue turnover, item lifespan, demand concentration | Dropshipping-Specific Conditions |
| `analyse_metadata_quality` | The dataset's `URL` column does not describe the row's item; this quantifies the resulting degeneracy and re-tests H2 on the reliable field alone | Preprocessing, Results |

The three sweeps depend on `load_dataset(thin=...)` in `data.py`, which
removes training interactions before any derived quantity is computed, so item
popularity and the Eq. 4 trend gate stay consistent with what remains. The
test set is never touched.

Requires: `pandas`, `numpy`, `scipy`, `scikit-learn`, `implicit`
(Python 3.10; no database needed — runs directly from the CSV).

Outputs in `results/`: `dataset_stats.json`, `tuning.json`,
`summary_seed*.json`, `per_user_seed*.npz` (per-user metric arrays for
significance testing), `tables.md`, `sensitivity_grid.csv`,
`fallback_stats.json` (how much of the catalogue each component can rank),
the six second-revision files listed above, and `FINDINGS.md` (dated working
notes, not tracked).
