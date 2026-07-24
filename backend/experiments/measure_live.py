"""Measure live latency and memory of the recommendation engine on MySQL.

Each method runs in its own subprocess so memory (peak RSS) is attributable
to that method alone. Latency is reported as a distribution (median / mean /
p95) over distinct product requests, after one warm-up call, plus a simple
4-thread concurrent burst for throughput under load.

Usage (from backend/):
    python -m experiments.measure_live                # driver, all methods
    python -m experiments.measure_live --method hybrid --json  # single method
"""
import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

RESULTS_DIR = Path(__file__).resolve().parent / "results"
METHODS = ["content", "collaborative", "trend", "hybrid"]
N_REQUESTS = 100
N_CONCURRENT = 50
THREADS = 4


def sample_products(engine, n):
    from sqlalchemy import text
    with engine.connect() as conn:
        rows = conn.execute(text(
            "SELECT DISTINCT ProductId FROM item_similarity LIMIT :n"
        ), {"n": n * 3}).fetchall()
    ids = [r[0] for r in rows]
    rng = np.random.default_rng(0)
    return list(rng.choice(ids, size=min(n, len(ids)), replace=False))


def run_method(method):
    import psutil
    from recommendation_model import AdvancedRecommendationEngine, get_db

    engine = AdvancedRecommendationEngine(get_db())
    fn = {
        "content": engine.get_similar_products,
        "collaborative": engine.get_collaborative_recommendations,
        "trend": engine.get_trending_recommendations,
        "hybrid": engine.get_hybrid_recommendations,
    }[method]

    products = sample_products(engine.engine, N_REQUESTS + N_CONCURRENT)
    fn(products[0], 4)  # warm-up (builds content index where relevant)

    times, n_results = [], []
    for pid in products[:N_REQUESTS]:
        t0 = time.perf_counter()
        recs = fn(pid, 4)
        times.append((time.perf_counter() - t0) * 1000)
        n_results.append(len(recs))

    # concurrent burst
    burst = products[N_REQUESTS : N_REQUESTS + N_CONCURRENT]
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=THREADS) as ex:
        list(ex.map(lambda p: fn(p, 4), burst))
    burst_wall = time.perf_counter() - t0

    rss_mb = psutil.Process().memory_info().rss / 1024 / 1024
    return {
        "method": method,
        "n_requests": len(times),
        "latency_ms": {
            "median": float(np.median(times)),
            "mean": float(np.mean(times)),
            "p95": float(np.percentile(times, 95)),
            "min": float(np.min(times)),
            "max": float(np.max(times)),
        },
        "concurrent": {
            "requests": N_CONCURRENT,
            "threads": THREADS,
            "wall_sec": burst_wall,
            "req_per_sec": N_CONCURRENT / burst_wall,
        },
        "avg_results_returned": float(np.mean(n_results)),
        "empty_result_rate": float(np.mean([n == 0 for n in n_results])),
        "process_rss_mb": rss_mb,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=METHODS)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    if args.method:
        result = run_method(args.method)
        print(json.dumps(result) if args.json else json.dumps(result, indent=2))
        return

    results = {}
    for method in METHODS:
        print(f"Measuring {method} ...", flush=True)
        out = subprocess.run(
            [sys.executable, "-m", "experiments.measure_live",
             "--method", method, "--json"],
            capture_output=True, text=True, cwd=Path(__file__).resolve().parents[1],
        )
        if out.returncode != 0:
            print(out.stderr[-2000:])
            raise SystemExit(f"{method} failed")
        results[method] = json.loads(out.stdout.strip().splitlines()[-1])

    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "live_performance.json", "w") as f:
        json.dump(results, f, indent=2)

    lines = [
        "### Live performance of the upgraded engine (Flask stack, MySQL)",
        "",
        f"{N_REQUESTS} sequential requests per method (after warm-up) + "
        f"{N_CONCURRENT}-request burst on {THREADS} threads. Isolated process per method.",
        "",
        "| Method | Median (ms) | Mean (ms) | p95 (ms) | Throughput (req/s) | RSS (MB) | Empty results |",
        "|---|---|---|---|---|---|---|",
    ]
    for method in METHODS:
        r = results[method]
        L = r["latency_ms"]
        lines.append(
            f"| {method} | {L['median']:.0f} | {L['mean']:.0f} | {L['p95']:.0f} "
            f"| {r['concurrent']['req_per_sec']:.1f} | {r['process_rss_mb']:.0f} "
            f"| {100*r['empty_result_rate']:.0f}% |"
        )
    (RESULTS_DIR / "live_performance.md").write_text("\n".join(lines) + "\n",
                                                     encoding="utf-8")
    print("\n".join(lines))
    print("\nWrote", RESULTS_DIR / "live_performance.json", "and .md")


if __name__ == "__main__":
    main()
