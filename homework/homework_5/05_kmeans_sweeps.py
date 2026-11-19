"""
05 — K-Means extras: the two "things to try" from the report.

Reuses the finished functions from 05_kmeans.py without changing them.

    uv run --with numpy --with matplotlib 05_kmeans_sweeps.py

1. Seed sweep: run k = 3 from seeds 0..99 and bucket the outcomes. Shows that
   random initialisation sometimes lands in the wrong valley, and that inertia
   (which needs no labels) exposes those runs.
2. k sweep: final inertia for k = 1..8, best of several restarts per k. Shows the
   elbow, and that inertia alone can never choose k because it always falls.
"""

import importlib.util
import pathlib
import numpy as np

# The filename starts with a digit, so a normal `import` will not work.
_path = pathlib.Path(__file__).with_name("05_kmeans.py")
_spec = importlib.util.spec_from_file_location("kmeans05", _path)
km = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(km)

N_SEEDS = 100
K_RANGE = range(1, 9)
RESTARTS = 10


def seed_sweep():
    rows = []
    for seed in range(N_SEEDS):
        rng = np.random.default_rng(seed)
        X, true_ids = km.make_data(rng)
        centroids, labels, history = km.fit_kmeans(X, km.DEFAULT_K, rng)
        rows.append((seed, history[-1], km.purity(true_ids, labels, km.DEFAULT_K)))
    return rows


def best_of(X, k, rng, restarts):
    """Run k-means several times and keep the lowest-inertia result."""
    best = None
    for _ in range(restarts):
        centroids, labels, history = km.fit_kmeans(X, k, rng)
        if best is None or history[-1] < best:
            best = history[-1]
    return best


def k_sweep():
    rng = np.random.default_rng(km.SEED)
    X, _ = km.make_data(rng)
    return [(k, best_of(X, k, rng, RESTARTS)) for k in K_RANGE]


def summarise_seeds(rows):
    buckets = {
        "purity 1.00": [r for r in rows if r[2] >= 0.9999],
        "purity 0.80-0.99": [r for r in rows if 0.80 <= r[2] < 0.9999],
        "purity under 0.80": [r for r in rows if r[2] < 0.80],
    }
    lines = [f"  {'bucket':<20}{'runs':>6}{'inertia range':>24}"]
    for name, rs in buckets.items():
        if rs:
            lo, hi = min(r[1] for r in rs), max(r[1] for r in rs)
            lines.append(f"  {name:<20}{len(rs):>6}{lo:>14.0f} ..{hi:>6.0f}")
        else:
            lines.append(f"  {name:<20}{0:>6}{'-':>24}")
    return buckets, "\n".join(lines)


def save_report(rows, buckets, ks, path="05_kmeans_sweeps_report.md"):
    bucket_rows = []
    for name, rs in buckets.items():
        if rs:
            rng_txt = f"{min(r[1] for r in rs):.0f} – {max(r[1] for r in rs):.0f}"
        else:
            rng_txt = "—"
        bucket_rows.append(f"| {name} | {len(rs)} | {rng_txt} |")
    bucket_rows = "\n".join(bucket_rows)

    bad = [r[0] for r in buckets["purity under 0.80"]]
    good_hi = max([r[1] for r in rows if r[2] >= 0.80])
    bad_lo = min([r[1] for r in rows if r[2] < 0.80], default=float("nan"))

    k_rows, prev = [], None
    for k, val in ks:
        drop = "—" if prev is None else f"{prev - val:.1f}"
        k_rows.append(f"| {k} | {val:.2f} | {drop} |")
        prev = val
    k_rows = "\n".join(k_rows)

    text = f"""# Homework 05 — K-Means extras

The two "things to try" from the main report, run with `05_kmeans_sweeps.py`. It
reuses the finished functions from `05_kmeans.py` unchanged.

## 1. Seed sweep — k = {km.DEFAULT_K}, seeds 0 to {len(rows) - 1}

Each seed generates its own dataset and its own random starting centroids.

| outcome | runs | final inertia range |
| --- | --- | --- |
{bucket_rows}

Collapsed seeds: {bad}.

The collapsed runs have inertia {bad_lo / good_hi:.1f}x or more above the worst good
run ({bad_lo:.0f} vs {good_hi:.0f}), so inertia separates them cleanly without
needing any labels. Inertia cannot catch the near misses, where a few boundary
points sit in the wrong cluster. Those look normal.

![seed sweep](05_kmeans_sweeps.png)

## 2. k sweep — k = 1 to {max(K_RANGE)}, best of {RESTARTS} restarts per k

| k | final inertia | drop from k-1 |
| --- | --- | --- |
{k_rows}

The elbow is at k = {len(km.TRUE_CENTERS)}, the true number of blobs: the big drops
stop there and every later step is small. Inertia still falls at every k, because
more clusters always fit better, so it cannot be minimised to choose k. The elbow
is read off the shape of the curve instead. Taking the best of several restarts at
each k keeps a bad initialisation from putting a kink in that curve.

## Takeaway

K-means can converge, correctly obeying its own rules, to the wrong answer, and a
single run never says so. Restarts plus inertia are how you catch it on real data.
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"  [report saved -> {path}]")


def main():
    print("=" * 64)
    print(f"  SEED SWEEP — k = {km.DEFAULT_K}, seeds 0..{N_SEEDS - 1}")
    print("=" * 64)
    rows = seed_sweep()
    buckets, table = summarise_seeds(rows)
    print(table)
    bad = [r[0] for r in buckets["purity under 0.80"]]
    print(f"\n  collapsed seeds: {bad}")
    print(f"  seed 31 purity : {rows[31][2]:.4f}  (the file's known failure)")

    print("\n" + "=" * 64)
    print(f"  K SWEEP — k = 1..{max(K_RANGE)}, best of {RESTARTS} restarts")
    print("=" * 64)
    ks = k_sweep()
    print(f"  {'k':>3}{'inertia':>12}{'drop from k-1':>16}")
    prev = None
    for k, val in ks:
        drop = "" if prev is None else f"{prev - val:.1f}"
        print(f"  {k:>3}{val:>12.2f}{drop:>16}")
        prev = val

    save_report(rows, buckets, ks)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [plots skipped — pip install matplotlib]")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))
    colors = {"purity 1.00": "tab:green", "purity 0.80-0.99": "tab:orange",
              "purity under 0.80": "tab:red"}
    for name, rs in buckets.items():
        if rs:
            ax1.scatter([r[1] for r in rs], [r[2] for r in rs], s=22, alpha=0.7,
                        c=colors[name], label=f"{name} ({len(rs)})")
    ax1.set_xlabel("final inertia"); ax1.set_ylabel("purity")
    ax1.set_title(f"{N_SEEDS} seeds, k = {km.DEFAULT_K}")
    ax1.legend(fontsize=8)

    ax2.plot([k for k, _ in ks], [v for _, v in ks], marker="o", lw=1.6)
    ax2.axvline(len(km.TRUE_CENTERS), ls="--", c="gray", lw=1)
    ax2.set_xlabel("k"); ax2.set_ylabel("final inertia")
    ax2.set_title("Elbow plot (dashed = true number of blobs)")

    fig.tight_layout(); fig.savefig("05_kmeans_sweeps.png", dpi=130); plt.close(fig)
    print("\n  [plot saved -> 05_kmeans_sweeps.png]")


if __name__ == "__main__":
    main()
