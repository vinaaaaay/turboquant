import numpy as np
import argparse
import os


def lloyd_max_1d(samples, k, max_iter=100, tol=1e-6):
    """
    1D k-means (Lloyd-Max) for scalar quantization
    """

    # initialize centroids (percentiles = stable)
    percentiles = np.linspace(0, 100, k)
    centroids = np.percentile(samples, percentiles)

    for _ in range(max_iter):
        # assign
        distances = np.abs(samples[:, None] - centroids[None, :])
        labels = np.argmin(distances, axis=1)

        new_centroids = np.array([
            samples[labels == i].mean() if np.any(labels == i) else centroids[i]
            for i in range(k)
        ])

        # convergence check
        if np.max(np.abs(new_centroids - centroids)) < tol:
            break

        centroids = new_centroids

    return np.sort(centroids)


def generate_codebook(d, b, num_samples=1_000_000, save_dir="config/codebooks"):
    """
    Generate codebook for TurboQuant
    """

    print(f"Generating codebook for d={d}, b={b}")

    # sample from Gaussian approx
    std = 1.0 / np.sqrt(d)
    samples = np.random.randn(num_samples) * std

    k = 2 ** b
    centroids = lloyd_max_1d(samples, k)

    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"codebook_d{d}_b{b}.npy")

    np.save(path, centroids)

    print(f"Saved codebook → {path}")
    print("Centroids:", centroids)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--d", type=int, required=True)
    parser.add_argument("--b", type=int, required=True)
    parser.add_argument("--samples", type=int, default=1_000_000)

    args = parser.parse_args()

    generate_codebook(args.d, args.b, args.samples)