#!/usr/bin/env python3
"""Spatial-EnBaSe example using MNIST.

Python: >= 3.12

This script:
1) Downloads MNIST from OpenML.
2) Applies Spatial-EnBaSe (per class, keep samples with score >= class median).
3) Prints selection ratio and class distribution comparison.

Usage:
    python examples/spatial_enbase.py --max-train 5000 --window 3
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from typing import Final

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.datasets import fetch_openml

MNIST_NAME: Final[str] = "mnist_784"
N_CLASSES: Final[int] = 10
N_BINS: Final[int] = 256
IMAGE_SIDE: Final[int] = 28


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Spatial-EnBaSe on MNIST.")
    parser.add_argument(
        "--max-train",
        type=int,
        default=5_000,
        help="Maximum number of training samples to use (default: 5000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for sub-sampling.",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=3,
        help="Spatial window size W (must be odd and >= 1).",
    )
    return parser.parse_args()


def load_mnist(max_train: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    X, y = fetch_openml(name=MNIST_NAME, version=1, return_X_y=True, as_frame=False)

    X = X.astype(np.uint8, copy=False)
    y = y.astype(np.int64, copy=False)

    X_train = X[:60_000]
    y_train = y[:60_000]

    return subsample(X_train, y_train, max_train, seed)


def subsample(X: np.ndarray, y: np.ndarray, limit: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if limit <= 0:
        raise ValueError("limit must be > 0")
    if limit >= len(X):
        return X, y

    rng = np.random.default_rng(seed)
    indices = rng.choice(len(X), size=limit, replace=False)
    return X[indices], y[indices]


def entropy_from_patch(patch: np.ndarray, bins: int = N_BINS) -> float:
    counts = np.bincount(patch, minlength=bins)
    probs = counts[counts > 0] / patch.size
    return float(-np.sum(probs * np.log2(probs)))


def compute_local_entropy(image_2d: np.ndarray, window: int) -> np.ndarray:
    if window < 1 or window % 2 == 0:
        raise ValueError("window must be odd and >= 1")

    # ComputeLocalEntropy(image, W): build E_map with local entropy per pixel.
    radius = window // 2
    padded = np.pad(image_2d, pad_width=radius, mode="edge")
    patches = sliding_window_view(padded, (window, window))
    flat_patches = patches.reshape(image_2d.shape[0] * image_2d.shape[1], window * window)

    values = np.fromiter((entropy_from_patch(patch) for patch in flat_patches), dtype=np.float64)
    return values.reshape(image_2d.shape)


def spatial_score(image_flat: np.ndarray, window: int) -> float:
    # H_score <- Mean(E_map)
    image_2d = image_flat.reshape(IMAGE_SIDE, IMAGE_SIDE)
    entropy_map = compute_local_entropy(image_2d, window)
    return float(np.mean(entropy_map))


def select_indices_by_spatial_enbase(
    X_train: np.ndarray,
    y_train: np.ndarray,
    window: int,
    n_classes: int = N_CLASSES,
) -> np.ndarray:
    # X_selected / Y_selected are represented by selected global indices.
    selected_indices: list[np.ndarray] = []

    for label in range(n_classes):
        # C <- indices belonging to class label.
        class_indices = np.flatnonzero(y_train == label)
        if class_indices.size == 0:
            continue

        # M_Score <- {(i, H_score)} for each sample in C.
        class_scores = np.array([spatial_score(X_train[idx], window) for idx in class_indices], dtype=np.float64)

        # Sort M_Score in descending order by H_score.
        sorted_order = np.argsort(-class_scores)
        sorted_indices = class_indices[sorted_order]
        sorted_scores = class_scores[sorted_order]

        # median <- median(scores in M_Score)
        median_score = float(np.median(sorted_scores))

        # IQualified <- {key.index | key.score >= median}
        keep_mask = sorted_scores >= median_score
        selected_indices.append(sorted_indices[keep_mask])

    if not selected_indices:
        raise RuntimeError("No samples selected by Spatial-EnBaSe.")

    return np.concatenate(selected_indices)


def run_spatial_enbase(X_train: np.ndarray, y_train: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    # Return X_selected, Y_selected exactly as in algorithm output.
    selected_idx = select_indices_by_spatial_enbase(X_train, y_train, window=window, n_classes=N_CLASSES)
    return X_train[selected_idx], y_train[selected_idx]


def summarize_distribution(values: Iterable[int]) -> dict[int, int]:
    unique, counts = np.unique(np.fromiter(values, dtype=np.int64), return_counts=True)
    return {int(k): int(v) for k, v in zip(unique, counts, strict=True)}


def main() -> None:
    args = parse_args()

    X_train, y_train = load_mnist(max_train=args.max_train, seed=args.seed)
    X_selected, y_selected = run_spatial_enbase(X_train, y_train, window=args.window)

    train_size = len(y_train)
    selected_size = len(y_selected)
    ratio = selected_size / train_size

    print("=== Spatial-EnBaSe on MNIST ===")
    print(f"Window size:            {args.window}")
    print(f"Train size (full):      {train_size}")
    print(f"Train size (selected):  {selected_size}")
    print(f"Selection ratio:        {ratio:.2%}")
    print()
    print("Class distribution (full):")
    print(summarize_distribution(y_train))
    print("Class distribution (selected):")
    print(summarize_distribution(y_selected))


if __name__ == "__main__":
    main()