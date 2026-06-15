"""Benchmark for the persistent q-communities core."""
import argparse
import resource
import sys
import time

import numpy as np


def make_dense_matrix(n, seed):
    rng = np.random.default_rng(seed)
    m = rng.random((n, n))
    m = (m + m.T) / 2
    np.fill_diagonal(m, np.inf)
    return m


def make_sparse_edges(n, avg_degree, seed):
    rng = np.random.default_rng(seed)
    num_edges = n * avg_degree // 2
    pairs = set()
    while len(pairs) < num_edges:
        u, v = rng.integers(0, n, 2)
        if u != v:
            pairs.add((min(u, v), max(u, v)))
    pairs = np.array(sorted(pairs), dtype=np.float64)
    weights = rng.random(len(pairs))[:, None]
    return np.hstack([pairs, weights])


def peak_rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)


def run_new(matrix, max_q):
    from q_analysis.persistence import persistent_q_communities
    return persistent_q_communities(matrix, max_q=max_q)


def run_new_edges(edges, max_q):
    from q_analysis.persistence import persistent_q_communities_from_edges
    return persistent_q_communities_from_edges(edges, max_q=max_q)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=30)
    parser.add_argument('--max-q', type=int, default=None)
    parser.add_argument('--sparse', action='store_true')
    parser.add_argument('--avg-degree', type=int, default=10)
    parser.add_argument('--round', type=int, default=None,
                        help='quantize weights to this many decimals (bigger threshold batches)')
    args = parser.parse_args()

    if args.sparse:
        data = make_sparse_edges(args.n, args.avg_degree, seed=7)
        label = f"sparse n={args.n} deg={args.avg_degree}"
        if args.round is not None:
            data[:, 2] = data[:, 2].round(args.round)
    else:
        data = make_dense_matrix(args.n, seed=7)
        label = f"dense n={args.n}"
        if args.round is not None:
            data = data.round(args.round)
    if args.round is not None:
        label += f" round={args.round}"

    start = time.perf_counter()
    if args.sparse:
        result = run_new_edges(data, args.max_q)
    else:
        result = run_new(data, args.max_q)
    elapsed = time.perf_counter() - start

    print(f"{label} max_q={args.max_q}: "
          f"{elapsed:.3f}s, {len(result)} intervals, peak RSS {peak_rss_mb():.1f} MB")


if __name__ == '__main__':
    main()
