# Running this project — short manual

Three independently runnable parts, all driven from the same dataset:

| Part | What it is | Needs MySQL? |
|---|---|---|
| A. Offline experiments | Reproduces every number in the paper (baselines, ablation, sensitivity, statistics) | No — reads the CSV directly |
| B. Web application | Flask API + React storefront with the live hybrid recommender | Yes |
| C. Paper figures | Weight-sensitivity heatmap + conceptual framework diagram | No |

**Prerequisites:** Python 3.10+, and for part B: MySQL 8/9 and Node.js 16+.
The dataset must be at `data/Amazon_Beauty_Recommendation.csv` (Kaggle:
`satrapankti/amazon-beauty-product-recommendation`, ~590 MB).

Install Python dependencies once:

```bash
cd backend
pip install -r requirements.txt
```

---

## A. Offline experiments (no database needed)

Run from the `backend/` directory:

```bash
# 1. sanity check (~1 min: 1 seed, 300 users)
python -m experiments.run_experiments --smoke

# 2. hyperparameter search on the validation split (~5 min; optional —
#    the chosen values are already the defaults)
python -m experiments.run_experiments --tune

# 3. full run: 5 seeds x 5,000 users x 78 systems (~15-20 min)
python -m experiments.run_experiments --seeds 5 --users 5000

# 4. aggregate into publication tables + significance tests
python -m experiments.make_tables

# 5. catalogue reachability per component (the fallback figures in the paper)
python -m experiments.analyse_fallback
```

Outputs land in `backend/experiments/results/`:
`tables.md` (all paper tables), `sensitivity_grid.csv`,
`summary_seed*.json`, `per_user_seed*.npz`, `dataset_stats.json`,
`fallback_stats.json`, and `live_performance.md` (from part B).
Protocol details: `backend/experiments/README.md`.

---

## B. Web application

### B1. One-time database setup

MySQL must be running on `localhost:3306`. The code falls back to a local
`root` account on database `dataset_db`; set `DATABASE_URL` to point at your
own credentials instead (recognised by all backend code):

```bash
# PowerShell:  $env:DATABASE_URL = "mysql+mysqlconnector://USER:PASS@localhost/dataset_db"
export DATABASE_URL="mysql+mysqlconnector://USER:PASS@localhost/dataset_db"
```

Then, from `backend/` (the database `dataset_db` must already exist —
`CREATE DATABASE dataset_db;` in a MySQL shell if not):

```bash
# load ratings + products from the CSV, create indexes  (~5 min)
python scripts/provision_mysql.py

# build the ItemKNN similarity table + materialised product stats (~25 s)
# -> REQUIRED before the app starts; rerun whenever the catalogue changes
python scripts/build_item_similarity.py
```

Note: `provision_mysql.py` generates deterministic placeholder prices
(real supplier prices are not part of the public dataset), and product
titles are derived from the product URL slugs. The dataset contains no
product images, so the storefront shows a placeholder image for every
product.

### B2. Start the backend (port 5001)

```bash
cd backend
python app.py
# ->  http://localhost:5001  (Flask dev server)
```

Quick smoke test without the frontend:

```bash
curl "http://localhost:5001/recommendations?product_id=B00LLPT4HI"
```

### B3. Start the frontend (port 3000)

```bash
cd frontend
npm install        # first time only
npm start
# ->  http://localhost:3000
```

The frontend calls the backend at `http://localhost:5001` via absolute URLs
(hard-coded in `frontend/src/components/**`), so both must run on their
default ports.

### B4. Measure live performance (optional; used for the paper's Table 6)

With MySQL provisioned (app itself doesn't need to be running):

```bash
cd backend
python -m experiments.measure_live
# -> experiments/results/live_performance.{md,json}
```

---

## C. Paper figures

```bash
cd ../revision/figures        # relative to the repo root's parent
python make_figures.py
# -> weight_sensitivity.{png,pdf,svg}, conceptual_framework.{png,pdf,svg}
```

Requires part A's `sensitivity_grid.csv` to exist (already committed with
results, or regenerate via A).

---

## Troubleshooting

- **`Can't connect to MySQL server`** — MySQL isn't running, or credentials
  differ from the default; set `DATABASE_URL` (see B1).
- **`Table 'item_similarity' doesn't exist`** — run
  `python scripts/build_item_similarity.py` (B1, second step).
- **Trend/hybrid endpoints slow (seconds)** — the `product_stats` table is
  missing; it is created by `build_item_similarity.py`. Without it the trend
  query re-aggregates 1.35 M rows per request.
- **`ModuleNotFoundError` in experiments** — run with `python -m
  experiments....` *from the `backend/` directory*, not from repo root.
- **First recommendation request after startup is slow (~1-2 s)** — the
  TF-IDF content index builds lazily on first use, then stays in memory.
- **`ImportError: cannot import name 'url_quote' from 'werkzeug.urls'`** — a
  too-new Werkzeug for Flask 2.0.1; run `pip install "werkzeug<2.1"` (already
  pinned in `requirements.txt`, so a fresh install avoids this).
- **Windows: MySQL not running** — start the service before Part B, e.g.
  `net start MySQL97` in an Administrator terminal (substitute your service
  name, seen via `services.msc`), then create the DB once: `CREATE DATABASE
  dataset_db;`.
- **Product images show a placeholder** — expected; the public dataset has no
  product images, so every product uses `static/images/product-placeholder.jpg`.
