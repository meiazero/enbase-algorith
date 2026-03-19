#!/usr/bin/env python3
"""EnBaSe (Entropy-Based Selection) example using MNIST.

Python: >= 3.12

This script:
1) Downloads MNIST from OpenML.
2) Applies EnBaSe (per class, keep samples with entropy <= class median).
3) Trains a simple classifier on full data vs EnBaSe-selected data.
4) Prints selection ratio and test accuracy comparison.

Usage:
    python examples/enbase_mnist.py --max-train 20000 --max-test 5000
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from typing import Final

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

MNIST_NAME: Final[str] = "mnist_784"
N_CLASSES: Final[int] = 10
N_BINS: Final[int] = 256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run EnBaSe on MNIST.")
    parser.add_argument(
        "--max-train",
        type=int,
        default=20_000,
        help="Maximum number of training samples to use (default: 20000).",
    )
    parser.add_argument(
        "--max-test",
        type=int,
        default=5_000,
        help="Maximum number of test samples to use (default: 5000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for sub-sampling.",
    )
    return parser.parse_args()


def load_mnist(max_train: int, max_test: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Load canonical MNIST from OpenML to keep replication simple and portable.
    X, y = fetch_openml(name=MNIST_NAME, version=1, return_X_y=True, as_frame=False)

    # OpenML returns float64 values in [0, 255] and labels as strings.
    X = X.astype(np.uint8, copy=False)
    y = y.astype(np.int64, copy=False)

    # Preserve the canonical split ordering from the original MNIST protocol.
    X_train, X_test = X[:60_000], X[60_000:]
    y_train, y_test = y[:60_000], y[60_000:]

    X_train, y_train = subsample(X_train, y_train, max_train, seed)
    X_test, y_test = subsample(X_test, y_test, max_test, seed + 1)

    return X_train, y_train, X_test, y_test


def subsample(X: np.ndarray, y: np.ndarray, limit: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if limit <= 0:
        raise ValueError("limit must be > 0")
    if limit >= len(X):
        return X, y

    rng = np.random.default_rng(seed)
    indices = rng.choice(len(X), size=limit, replace=False)
    return X[indices], y[indices]


def image_entropy(image: np.ndarray, bins: int = N_BINS) -> float:
    # Entropy H = -sum(p * log2(p)) over pixel-intensity distribution.
    counts = np.bincount(image, minlength=bins)
    probs = counts[counts > 0] / image.size
    return float(-np.sum(probs * np.log2(probs)))


def select_indices_by_enbase(X_train: np.ndarray, y_train: np.ndarray, n_classes: int = N_CLASSES) -> np.ndarray:
    # This function is the direct implementation of EnBaSe.
    selected_indices: list[np.ndarray] = []

    for label in range(n_classes):
        # C <- indices of samples belonging to the current class.
        class_indices = np.flatnonzero(y_train == label)
        if class_indices.size == 0:
            continue

        # MEntropy <- (sample_index, entropy) map for the current class.
        class_entropies = np.array([image_entropy(X_train[idx]) for idx in class_indices], dtype=np.float64)

        # median <- class entropy median.
        median_entropy = float(np.median(class_entropies))

        # IQualified <- samples with entropy <= median.
        keep_mask = class_entropies <= median_entropy
        selected_indices.append(class_indices[keep_mask])

    if not selected_indices:
        raise RuntimeError("No samples selected by EnBaSe.")

    return np.concatenate(selected_indices)


def run_enbase(X_train: np.ndarray, y_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Return X_selected and Y_selected as defined in the paper pseudocode.
    selected_idx = select_indices_by_enbase(X_train, y_train, n_classes=N_CLASSES)
    return X_train[selected_idx], y_train[selected_idx]


def train_and_eval(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> float:
    # A simple baseline model to compare full training vs EnBaSe-selected training.
    model = LogisticRegression(
        solver="saga",
        multi_class="multinomial",
        max_iter=120,
        n_jobs=-1,
        random_state=42,
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    return float(accuracy_score(y_test, pred))


def scale_pixels(X: np.ndarray) -> np.ndarray:
    return X.astype(np.float32) / 255.0


def summarize_distribution(values: Iterable[int]) -> dict[int, int]:
    unique, counts = np.unique(np.fromiter(values, dtype=np.int64), return_counts=True)
    return {int(k): int(v) for k, v in zip(unique, counts, strict=True)}


def main() -> None:
    args = parse_args()

    X_train, y_train, X_test, y_test = load_mnist(
        max_train=args.max_train,
        max_test=args.max_test,
        seed=args.seed,
    )

    X_train_scaled = scale_pixels(X_train)
    X_test_scaled = scale_pixels(X_test)

    # Apply EnBaSe on raw pixel values (uint8) to match entropy computation assumptions.
    X_selected, y_selected = run_enbase(X_train, y_train)
    X_selected_scaled = scale_pixels(X_selected)

    # Evaluate both training strategies on the same test split.
    full_acc = train_and_eval(X_train_scaled, y_train, X_test_scaled, y_test)
    enbase_acc = train_and_eval(X_selected_scaled, y_selected, X_test_scaled, y_test)

    train_size = len(y_train)
    selected_size = len(y_selected)
    ratio = selected_size / train_size

    print("=== EnBaSe on MNIST ===")
    print(f"Train size (full):      {train_size}")
    print(f"Train size (EnBaSe):    {selected_size}")
    print(f"Selection ratio:        {ratio:.2%}")
    print()
    print("Class distribution (full):")
    print(summarize_distribution(y_train))
    print("Class distribution (EnBaSe):")
    print(summarize_distribution(y_selected))
    print()
    print(f"Accuracy full train:    {full_acc:.4f}")
    print(f"Accuracy EnBaSe train:  {enbase_acc:.4f}")


if __name__ == "__main__":
    main()
