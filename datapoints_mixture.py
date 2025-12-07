#!/usr/bin/env python3
"""
datapoints_mixture.py

Generate samples from a 2-D Gaussian mixture (two independent Gaussian components),
optionally run K-means (k=2), soft-K-means variants, and plot results.

Usage examples:
    # default: show mixture scatter
    python datapoints_mixture.py

    # run hard K-means and show K-means plot
    python datapoints_mixture.py --run-kmeans --plot-kmeans

    # run soft K-means v1 with beta=1.0 and show plot
    python datapoints_mixture.py --run-softk1 --beta 1.0 --plot-softk1

    # run soft K-means v3 (annealing) and save overlay
    python datapoints_mixture.py --run-softk3 --beta-start 0.1 --beta-mult 1.5 --plot-softk3 --out-plot overlay.png --no-show
"""
from __future__ import annotations
import argparse
import csv
import sys
from typing import Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt


# ------------------------
# Data generation
# ------------------------
def datapoints(n: int,
               mu1x: float, mu1y: float, sigma1x2: float, sigma1y2: float, pi1: float,
               mu2x: float, mu2y: float, sigma2x2: float, sigma2y2: float, pi2: float,
               random_seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate n samples from a 2-component 2D Gaussian mixture.
    Each component has independent x and y Gaussians (diagonal covariance).
    Returns:
      samples: (n,2) array of (x,y)
      labels: (n,) array of ints {0,1} indicating which true mixture component generated each sample
    """
    if random_seed is not None:
        np.random.seed(int(random_seed))

    if not (0.0 <= pi1 <= 1.0 and 0.0 <= pi2 <= 1.0):
        raise ValueError("Mixing weights must be between 0 and 1")
    if abs((pi1 + pi2) - 1.0) > 1e-8:
        raise ValueError("pi1 and pi2 must sum to 1")

    # draw which component each sample comes from (0 -> component1, 1 -> component2)
    comps = np.random.choice([0, 1], size=n, p=[pi1, pi2])

    # allocate array
    samples = np.zeros((n, 2), dtype=float)

    # convert variances to standard deviations
    s1x = float(np.sqrt(sigma1x2))
    s1y = float(np.sqrt(sigma1y2))
    s2x = float(np.sqrt(sigma2x2))
    s2y = float(np.sqrt(sigma2y2))

    # sample component 1
    idx1 = (comps == 0)
    m1 = idx1.sum()
    if m1 > 0:
        samples[idx1, 0] = np.random.normal(loc=mu1x, scale=s1x, size=m1)
        samples[idx1, 1] = np.random.normal(loc=mu1y, scale=s1y, size=m1)

    # sample component 2
    idx2 = (comps == 1)
    m2 = idx2.sum()
    if m2 > 0:
        samples[idx2, 0] = np.random.normal(loc=mu2x, scale=s2x, size=m2)
        samples[idx2, 1] = np.random.normal(loc=mu2y, scale=s2y, size=m2)

    return samples, comps


# ------------------------
# Hard K-means (k=2)
# ------------------------
def kmeans_2clusters(samples: np.ndarray,
                     seed: Optional[int] = None,
                     max_iters: int = 100) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Simple K-means for k=2 (Euclidean distance). Returns:
      labels: (n,) ints {0,1}
      centroids: (2,2) array
      iters: number of iterations performed
    Initialization: pick two distinct samples at random as initial centroids (deterministic if seed set).
    """
    n, d = samples.shape
    if d != 2:
        raise ValueError("This kmeans implementation expects 2D samples")

    rng = np.random.RandomState(seed)

    # pick two distinct indices
    idxs = rng.choice(n, size=2, replace=False)
    centroids = samples[idxs].astype(float)  # shape (2,2)

    labels = np.full(n, -1, dtype=int)
    for it in range(1, max_iters + 1):
        # assign labels by nearest centroid
        diff0 = samples - centroids[0:1, :]
        dist0 = np.sum(diff0 * diff0, axis=1)
        diff1 = samples - centroids[1:2, :]
        dist1 = np.sum(diff1 * diff1, axis=1)

        new_labels = (dist1 < dist0).astype(int)  # 0 if closer to centroid 0, 1 if closer to centroid 1

        # if no label changes, converged
        if np.array_equal(new_labels, labels):
            return labels, centroids, it - 1

        labels = new_labels

        # update centroids
        for k in (0, 1):
            idxk = (labels == k)
            if np.any(idxk):
                centroids[k] = samples[idxk].mean(axis=0)
            else:
                # cluster empty: reinitialize that centroid to a random sample
                centroids[k] = samples[rng.choice(n)]

    return labels, centroids, max_iters


# ------------------------
# Soft K-means (v1: fixed beta)
# ------------------------
def soft_kmeans_v1(samples: np.ndarray,
                   beta: float = 1.0,
                   seed: Optional[int] = None,
                   max_iters: int = 200,
                   tol: float = 1e-6) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Soft K-means (2 clusters) with fixed temperature beta.
    responsibilities r[n,k] = exp(-beta * dist_nk^2) / sum_k exp(-beta * dist_nk^2)
    Returns:
      r: (n,2) responsibilities (soft assignments)
      centroids: (2,2)
      hard_labels: (n,) argmax of r
      iters: number of iterations
    """
    n, d = samples.shape
    if d != 2:
        raise ValueError("Expected 2D samples")
    rng = np.random.RandomState(seed)
    # init centroids randomly from samples
    idxs = rng.choice(n, size=2, replace=False)
    centroids = samples[idxs].astype(float)

    prev_centroids = centroids.copy()
    for it in range(1, max_iters + 1):
        # compute squared distances n x 2
        diff0 = samples - centroids[0:1, :]
        diff1 = samples - centroids[1:2, :]
        d0 = np.sum(diff0 * diff0, axis=1)
        d1 = np.sum(diff1 * diff1, axis=1)
        # responsibilities (avoid overflow by subtracting min)
        # exponent = -beta * d
        e0 = np.exp(-beta * d0)
        e1 = np.exp(-beta * d1)
        denom = e0 + e1
        # avoid division by zero
        denom = np.where(denom == 0.0, 1e-16, denom)
        r0 = e0 / denom
        r1 = e1 / denom
        r = np.vstack([r0, r1]).T  # shape (n,2)

        # update centroids (weighted means)
        r0_sum = r0.sum()
        r1_sum = r1.sum()
        # prevent division by zero: if cluster weight is zero, reinit centroid
        if r0_sum > 0:
            centroids[0] = (r0[:, None] * samples).sum(axis=0) / r0_sum
        else:
            centroids[0] = samples[rng.choice(n)]
        if r1_sum > 0:
            centroids[1] = (r1[:, None] * samples).sum(axis=0) / r1_sum
        else:
            centroids[1] = samples[rng.choice(n)]

        # check movement
        move = np.max(np.abs(centroids - prev_centroids))
        if move < tol:
            hard_labels = np.argmax(r, axis=1)
            return r, centroids, hard_labels, it
        prev_centroids[:] = centroids

    hard_labels = np.argmax(r, axis=1)
    return r, centroids, hard_labels, max_iters


# ------------------------
# Soft K-means v3 (annealing):
# start with beta_start and multiply by beta_mult each iteration
# ------------------------
def soft_kmeans_v3(samples: np.ndarray,
                   beta_start: float = 0.1,
                   beta_mult: float = 1.5,
                   seed: Optional[int] = None,
                   max_iters: int = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Soft K-means variant with annealing on beta.
    Returns the final responsibilities, centroids, hard labels, and iterations.
    """
    n, d = samples.shape
    rng = np.random.RandomState(seed)
    idxs = rng.choice(n, size=2, replace=False)
    centroids = samples[idxs].astype(float)

    beta = float(beta_start)
    prev_centroids = centroids.copy()

    for it in range(1, max_iters + 1):
        # compute squared distances
        diff0 = samples - centroids[0:1, :]
        diff1 = samples - centroids[1:2, :]
        d0 = np.sum(diff0 * diff0, axis=1)
        d1 = np.sum(diff1 * diff1, axis=1)

        e0 = np.exp(-beta * d0)
        e1 = np.exp(-beta * d1)
        denom = e0 + e1
        denom = np.where(denom == 0.0, 1e-16, denom)
        r0 = e0 / denom
        r1 = e1 / denom
        r = np.vstack([r0, r1]).T

        # update centroids
        r0_sum = r0.sum()
        r1_sum = r1.sum()
        if r0_sum > 0:
            centroids[0] = (r0[:, None] * samples).sum(axis=0) / r0_sum
        else:
            centroids[0] = samples[rng.choice(n)]
        if r1_sum > 0:
            centroids[1] = (r1[:, None] * samples).sum(axis=0) / r1_sum
        else:
            centroids[1] = samples[rng.choice(n)]

        # convergence by centroid movement
        move = np.max(np.abs(centroids - prev_centroids))
        prev_centroids[:] = centroids

        # anneal beta
        beta *= float(beta_mult)

        if move < 1e-6:
            hard_labels = np.argmax(r, axis=1)
            return r, centroids, hard_labels, it

    hard_labels = np.argmax(r, axis=1)
    return r, centroids, hard_labels, max_iters


# ------------------------
# Plotting helpers
# ------------------------
def plot_mixture(samples: np.ndarray, labels: np.ndarray,
                 title: str = "Mixture samples",
                 show: bool = True, out_path: Optional[str] = None):
    plt.figure(figsize=(8, 7))
    plt.scatter(samples[labels == 0, 0], samples[labels == 0, 1],
                marker='o', alpha=0.7, label='mixture comp 1 (label=0)')
    plt.scatter(samples[labels == 1, 0], samples[labels == 1, 1],
                marker='x', alpha=0.7, label='mixture comp 2 (label=1)')
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend()
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=200)
        print(f"Saved plot to: {out_path}")
    if show:
        plt.show()
    else:
        plt.close()


def plot_kmeans_result(samples: np.ndarray, labels: np.ndarray, centroids: np.ndarray,
                       title: str = "K-means clusters (k=2)",
                       show: bool = True, out_path: Optional[str] = None):
    plt.figure(figsize=(8, 7))
    plt.scatter(samples[labels == 0, 0], samples[labels == 0, 1],
                marker='s', alpha=0.7, label='cluster 0 (square)')
    plt.scatter(samples[labels == 1, 0], samples[labels == 1, 1],
                marker='x', alpha=0.7, label='cluster 1 (cross)')
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='D', s=120, edgecolor='k',
                linewidth=1.0, label='centroids')
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend()
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=200)
        print(f"Saved plot to: {out_path}")
    if show:
        plt.show()
    else:
        plt.close()


def plot_overlay(samples: np.ndarray, mixture_labels: np.ndarray, algo_labels: np.ndarray, centroids: np.ndarray,
                 title: str = "Overlay: mixture vs algorithm",
                 show: bool = True, out_path: Optional[str] = None):
    plt.figure(figsize=(8, 7))
    # mixture faint
    plt.scatter(samples[mixture_labels == 0, 0], samples[mixture_labels == 0, 1],
                marker='o', alpha=0.25, label='mixture comp 0 (faint)')
    plt.scatter(samples[mixture_labels == 1, 0], samples[mixture_labels == 1, 1],
                marker='o', alpha=0.25, label='mixture comp 1 (faint)')
    # algorithm result
    plt.scatter(samples[algo_labels == 0, 0], samples[algo_labels == 0, 1],
                marker='s', alpha=0.85, label='algo cluster 0 (square)')
    plt.scatter(samples[algo_labels == 1, 0], samples[algo_labels == 1, 1],
                marker='x', alpha=0.85, label='algo cluster 1 (cross)')
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='D', s=140, edgecolor='k',
                linewidth=1.2, label='centroids')
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend()
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=200)
        print(f"Saved plot to: {out_path}")
    if show:
        plt.show()
    else:
        plt.close()


# ------------------------
# IO helpers
# ------------------------
def save_csv(samples: np.ndarray, labels: np.ndarray, path: str):
    with open(path, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y", "label"])
        for (x, y), lab in zip(samples, labels):
            writer.writerow([f"{x:.8g}", f"{y:.8g}", int(lab)])
    print(f"Saved samples CSV to: {path}")


def save_npy(samples: np.ndarray, labels: np.ndarray, path: str):
    np.savez_compressed(path, samples=samples, labels=labels)
    print(f"Saved numpy archive to: {path}")


# ------------------------
# Evaluation: percent incorrect (compares two labelings with best permutation)
# ------------------------
def percent_incorrect(true_labels: np.ndarray, pred_labels: np.ndarray) -> float:
    """
    For two clusters only: compute % of points incorrectly assigned, allowing for label swap.
    Returns percentage (0..100).
    """
    if true_labels.shape != pred_labels.shape:
        raise ValueError("Label arrays must have same shape")
    # mapping: either pred as-is or invert predictions
    mismatch_1 = np.sum(true_labels != pred_labels)
    mismatch_2 = np.sum(true_labels != (1 - pred_labels))
    best_mismatch = min(mismatch_1, mismatch_2)
    pct = 100.0 * best_mismatch / true_labels.size
    return pct


# ------------------------
# CLI parsing and main
# ------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(prog="datapoints_mixture.py",
                                description="Generate 2D Gaussian mixture samples and run clustering.")
    p.add_argument("--n", type=int, default=500, help="number of samples to generate")
    p.add_argument("--mu1x", type=float, default=2.0)
    p.add_argument("--mu1y", type=float, default=2.0)
    p.add_argument("--sigma1x2", type=float, default=1.0, help="variance of comp1 x")
    p.add_argument("--sigma1y2", type=float, default=4.0, help="variance of comp1 y")
    p.add_argument("--pi1", type=float, default=0.2, help="mixing weight for component 1")
    p.add_argument("--mu2x", type=float, default=-2.0)
    p.add_argument("--mu2y", type=float, default=-2.0)
    p.add_argument("--sigma2x2", type=float, default=4.0, help="variance of comp2 x")
    p.add_argument("--sigma2y2", type=float, default=2.0, help="variance of comp2 y")
    p.add_argument("--pi2", type=float, default=0.8, help="mixing weight for component 2")
    p.add_argument("--seed", type=int, default=None, help="random seed for generating mixture")
    p.add_argument("--out-csv", type=str, default=None, help="path to save samples as CSV")
    p.add_argument("--out-npz", type=str, default=None, help="path to save samples as npz (numpy)")
    p.add_argument("--out-plot", type=str, default=None, help="base path to save plots (png/pdf)")

    # plotting / kmeans controls
    p.add_argument("--plot-mix", action="store_true", help="plot the mixture (true component labels)")
    p.add_argument("--run-kmeans", action="store_true", help="run hard K-means (k=2) on samples")
    p.add_argument("--plot-kmeans", action="store_true", help="plot the hard K-means result (squares / crosses)")

    # soft-kmeans v1 (fixed beta)
    p.add_argument("--run-softk1", action="store_true", help="run soft K-means v1 (fixed beta)")
    p.add_argument("--plot-softk1", action="store_true", help="plot soft K-means v1 result")
    p.add_argument("--beta", type=float, default=1.0, help="beta (inverse temperature) for soft-kmeans v1")

    # soft-kmeans v3 (annealing)
    p.add_argument("--run-softk3", action="store_true", help="run soft K-means v3 (annealing)")
    p.add_argument("--plot-softk3", action="store_true", help="plot soft K-means v3 result")
    p.add_argument("--beta-start", type=float, default=0.1, help="beta start for annealing (soft-kmeans v3)")
    p.add_argument("--beta-mult", type=float, default=1.5, help="beta multiplier per iteration (soft-kmeans v3)")

    p.add_argument("--no-show", action="store_true", help="do not display interactive plots")
    p.add_argument("--k-seed", type=int, default=None, help="seed used for K-means/softK init")
    p.add_argument("--k-max-iters", type=int, default=200, help="max iterations for K-means / soft-kmeans")
    return p.parse_args(argv)

def main(argv=None):
    args = parse_args(argv)

    # default behavior: if no plot/run flags provided, show the mixture plot
    # (check all flags that control plotting or running algorithms)
    if not (
        args.plot_mix
        or args.plot_kmeans
        or args.run_kmeans
        or args.run_softk1
        or args.run_softk3
        or args.plot_softk1
        or args.plot_softk3
    ):
        args.plot_mix = True

    # validate mixing weights
    if abs((args.pi1 + args.pi2) - 1.0) > 1e-8:
        print("Error: pi1 and pi2 must sum to 1.0", file=sys.stderr)
        sys.exit(2)

    samples, mixture_labels = datapoints(
        n=args.n,
        mu1x=args.mu1x, mu1y=args.mu1y, sigma1x2=args.sigma1x2, sigma1y2=args.sigma1y2, pi1=args.pi1,
        mu2x=args.mu2x, mu2y=args.mu2y, sigma2x2=args.sigma2x2, sigma2y2=args.sigma2y2, pi2=args.pi2,
        random_seed=args.seed
    )

    print(f"Generated {args.n} samples.")
    u, counts = np.unique(mixture_labels, return_counts=True)
    print("True mixture component counts:", dict(zip(u.tolist(), counts.tolist())))

    if args.out_csv:
        save_csv(samples, mixture_labels, args.out_csv)
    if getattr(args, "out_npy", None):
        # support either out_npy or out-npz naming depending on previous edits
        save_npy(samples, mixture_labels, args.out_npy)

    # Hard K-means
    klabels = None
    kcentroids = None
    if args.run_kmeans:
        klabels, kcentroids, iters = kmeans_2clusters(samples, seed=args.k_seed, max_iters=args.k_max_iters)
        print(f"K-means finished in {iters} iterations.")
        u, cnts = np.unique(klabels, return_counts=True)
        print("K-means cluster counts:", dict(zip(u.tolist(), cnts.tolist())))
        print("K-means centroids:\n", kcentroids)
        pct_wrong = percent_incorrect(mixture_labels, klabels)
        print(f"K-means percent incorrectly clustered: {pct_wrong:.2f}%")

    # Soft K-means v1 (fixed beta)
    soft1_r = None
    soft1_centroids = None
    soft1_labels = None
    if args.run_softk1:
        soft1_r, soft1_centroids, soft1_labels, iters = soft_kmeans_v1(
            samples, beta=args.beta, seed=args.k_seed, max_iters=args.k_max_iters)
        print(f"Soft-K-means v1 (beta={args.beta}) finished in {iters} iterations.")
        u, cnts = np.unique(soft1_labels, return_counts=True)
        print("Soft-k1 hard-label counts:", dict(zip(u.tolist(), cnts.tolist())))
        print("Soft-k1 centroids:\n", soft1_centroids)
        pct_wrong = percent_incorrect(mixture_labels, soft1_labels)
        print(f"Soft-k1 percent incorrectly clustered: {pct_wrong:.2f}%")

    # Soft K-means v3 (annealing)
    soft3_r = None
    soft3_centroids = None
    soft3_labels = None
    if args.run_softk3:
        soft3_r, soft3_centroids, soft3_labels, iters = soft_kmeans_v3(
            samples, beta_start=args.beta_start, beta_mult=args.beta_mult, seed=args.k_seed, max_iters=args.k_max_iters)
        print(f"Soft-K-means v3 (annealing start={args.beta_start}, mult={args.beta_mult}) finished in {iters} iterations.")
        u, cnts = np.unique(soft3_labels, return_counts=True)
        print("Soft-k3 hard-label counts:", dict(zip(u.tolist(), cnts.tolist())))
        print("Soft-k3 centroids:\n", soft3_centroids)
        pct_wrong = percent_incorrect(mixture_labels, soft3_labels)
        print(f"Soft-k3 percent incorrectly clustered: {pct_wrong:.2f}%")

    # Plotting
    base_out = args.out_plot

    # plot mixture
    if args.plot_mix:
        outpath = None
        if base_out:
            outpath = base_out.replace('.png', '_mix.png').replace('.pdf', '_mix.pdf')
            if outpath == base_out:
                outpath = base_out + '_mix.png'
        plot_mixture(samples, mixture_labels, title=f"Mixture (n={args.n})", show=(not args.no_show), out_path=outpath)

    # plot hard kmeans
    if args.plot_kmeans:
        if klabels is None:
            klabels, kcentroids, iters = kmeans_2clusters(samples, seed=args.k_seed, max_iters=args.k_max_iters)
        outpath = None
        if base_out:
            outpath = base_out.replace('.png', '_kmeans.png').replace('.pdf', '_kmeans.pdf')
            if outpath == base_out:
                outpath = base_out + '_kmeans.png'
        plot_kmeans_result(samples, klabels, kcentroids, title=f"K-means (n={args.n})", show=(not args.no_show), out_path=outpath)

    # plot softk1
    if args.plot_softk1:
        if soft1_labels is None:
            soft1_r, soft1_centroids, soft1_labels, iters = soft_kmeans_v1(samples, beta=args.beta, seed=args.k_seed, max_iters=args.k_max_iters)
        outpath = None
        if base_out:
            outpath = base_out.replace('.png', '_softk1.png').replace('.pdf', '_softk1.pdf')
            if outpath == base_out:
                outpath = base_out + '_softk1.png'
        plot_kmeans_result(samples, soft1_labels, soft1_centroids, title=f"Soft-K-means v1 (beta={args.beta})", show=(not args.no_show), out_path=outpath)

    # plot softk3
    if args.plot_softk3:
        if soft3_labels is None:
            soft3_r, soft3_centroids, soft3_labels, iters = soft_kmeans_v3(samples, beta_start=args.beta_start, beta_mult=args.beta_mult, seed=args.k_seed, max_iters=args.k_max_iters)
        outpath = None
        if base_out:
            outpath = base_out.replace('.png', '_softk3.png').replace('.pdf', '_softk3.pdf')
            if outpath == base_out:
                outpath = base_out + '_softk3.png'
        plot_kmeans_result(samples, soft3_labels, soft3_centroids, title=f"Soft-K-means v3 (anneal start={args.beta_start})", show=(not args.no_show), out_path=outpath)


if __name__ == "__main__":
    main()
