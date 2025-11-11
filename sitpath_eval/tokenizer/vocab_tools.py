from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def save_vocab(path: str | Path, vocab: Dict[str, int]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(vocab, fh, indent=2)


def load_vocab(path: str | Path) -> Dict[str, int]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def vocab_frequencies(tokens: Iterable[int]) -> Counter:
    return Counter(tokens)


def plot_zipf(frequencies: Counter, *, ax=None) -> None:
    ax = ax if ax is not None else plt.gca()
    freqs = np.array(sorted(frequencies.values(), reverse=True), dtype=float)
    ranks = np.arange(1, len(freqs) + 1)
    ax.loglog(ranks, freqs)
    ax.set_xlabel("Rank")
    ax.set_ylabel("Frequency")
    ax.set_title("Zipf plot")


def mutual_information(tokens: Iterable[int], labels: Iterable[int]) -> float:
    tokens = np.asarray(list(tokens))
    labels = np.asarray(list(labels))
    joint = Counter(zip(tokens, labels))
    token_counts = Counter(tokens)
    label_counts = Counter(labels)
    total = len(tokens)
    mi = 0.0
    for (token, label), joint_count in joint.items():
        p_joint = joint_count / total
        p_token = token_counts[token] / total
        p_label = label_counts[label] / total
        mi += p_joint * np.log2(p_joint / (p_token * p_label))
    return float(mi)


def linear_probe(features: np.ndarray, labels: np.ndarray, *, test_size: float = 0.2, random_state: int = 0) -> Tuple[float, LogisticRegression]:
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=test_size, random_state=random_state)
    clf = LogisticRegression(max_iter=200)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    return acc, clf


__all__ = [
    "save_vocab",
    "load_vocab",
    "vocab_frequencies",
    "plot_zipf",
    "mutual_information",
    "linear_probe",
]
