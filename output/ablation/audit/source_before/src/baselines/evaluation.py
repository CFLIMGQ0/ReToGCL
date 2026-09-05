"""分层五折线性评测及分类/聚类一致性指标。"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from tqdm import tqdm


METRIC_KEYS = ("accuracy", "nmi", "ari", "macro_f1")


def classification_metrics(labels: Tensor, predictions: Tensor) -> dict[str, float]:
    """返回 0--1 量纲的 Accuracy、NMI、ARI 和 Macro-F1。"""
    truth = labels.detach().cpu().numpy().reshape(-1)
    predicted = predictions.detach().cpu().numpy().reshape(-1)
    if truth.size == 0 or truth.size != predicted.size:
        raise ValueError("标签和预测必须是等长的非空向量")

    truth_values, truth_inverse = np.unique(truth, return_inverse=True)
    predicted_values, predicted_inverse = np.unique(predicted, return_inverse=True)
    contingency = np.zeros((truth_values.size, predicted_values.size), dtype=np.int64)
    np.add.at(contingency, (truth_inverse, predicted_inverse), 1)

    sample_count = truth.size
    probabilities = contingency / sample_count
    truth_probability = probabilities.sum(axis=1)
    predicted_probability = probabilities.sum(axis=0)
    nonzero = probabilities > 0
    independent = truth_probability[:, None] * predicted_probability[None, :]
    mutual_information = float(
        np.sum(probabilities[nonzero] * np.log(probabilities[nonzero] / independent[nonzero]))
    )
    truth_entropy = float(-np.sum(truth_probability[truth_probability > 0] * np.log(
        truth_probability[truth_probability > 0]
    )))
    predicted_entropy = float(-np.sum(predicted_probability[predicted_probability > 0] * np.log(
        predicted_probability[predicted_probability > 0]
    )))
    entropy_mean = (truth_entropy + predicted_entropy) / 2
    nmi = 1.0 if entropy_mean == 0 else mutual_information / entropy_mean

    choose_two = lambda values: np.sum(values * (values - 1) / 2)
    total_pairs = sample_count * (sample_count - 1) / 2
    cell_pairs = float(choose_two(contingency))
    truth_pairs = float(choose_two(contingency.sum(axis=1)))
    predicted_pairs = float(choose_two(contingency.sum(axis=0)))
    expected_pairs = 0.0 if total_pairs == 0 else truth_pairs * predicted_pairs / total_pairs
    maximum_pairs = (truth_pairs + predicted_pairs) / 2
    ari_denominator = maximum_pairs - expected_pairs
    ari = 1.0 if ari_denominator == 0 else (cell_pairs - expected_pairs) / ari_denominator

    class_f1 = []
    for label in np.union1d(truth_values, predicted_values):
        true_positive = np.sum((truth == label) & (predicted == label))
        false_positive = np.sum((truth != label) & (predicted == label))
        false_negative = np.sum((truth == label) & (predicted != label))
        denominator = 2 * true_positive + false_positive + false_negative
        class_f1.append(0.0 if denominator == 0 else 2 * true_positive / denominator)
    return {
        "accuracy": float(np.mean(truth == predicted)),
        "nmi": float(np.clip(nmi, 0.0, 1.0)),
        "ari": float(np.clip(ari, -1.0, 1.0)),
        "macro_f1": float(np.mean(class_f1)),
    }


def stratified_folds(labels: Tensor, folds: int, seed: int) -> list[Tensor]:
    generator = torch.Generator().manual_seed(seed)
    fold_indices: list[list[Tensor]] = [[] for _ in range(folds)]
    for label in labels.unique(sorted=True):
        indices = (labels == label).nonzero(as_tuple=False).flatten()
        indices = indices[torch.randperm(indices.numel(), generator=generator)]
        for offset, index in enumerate(indices):
            fold_indices[offset % folds].append(index.view(1))
    return [torch.cat(parts).sort().values for parts in fold_indices]


def linear_probe_five_fold_metrics(
    embeddings: Tensor,
    labels: Tensor,
    device: torch.device,
    seed: int = 42,
    folds: int = 5,
    epochs: int = 200,
    split_seed: int | None = None,
) -> dict[str, list[float]]:
    embeddings = embeddings.float().cpu()
    labels = labels.long().cpu()
    split = stratified_folds(labels, folds, seed if split_seed is None else split_seed)
    results = {key: [] for key in METRIC_KEYS}

    for fold, test_indices in enumerate(tqdm(split, desc="五折线性评测", unit="折"), start=1):
        train_mask = torch.ones(labels.numel(), dtype=torch.bool)
        train_mask[test_indices] = False
        train_x, test_x = embeddings[train_mask], embeddings[test_indices]
        train_y, test_y = labels[train_mask], labels[test_indices]

        mean = train_x.mean(dim=0, keepdim=True)
        std = train_x.std(dim=0, keepdim=True).clamp_min(1e-6)
        train_x = ((train_x - mean) / std).to(device)
        test_x = ((test_x - mean) / std).to(device)
        train_y, test_y = train_y.to(device), test_y.to(device)

        torch.manual_seed(seed + fold)
        classifier = nn.Linear(embeddings.size(1), int(labels.max()) + 1).to(device)
        optimizer = torch.optim.Adam(classifier.parameters(), lr=0.05, weight_decay=1e-4)
        for _ in range(epochs):
            optimizer.zero_grad()
            loss = F.cross_entropy(classifier(train_x), train_y)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            predictions = classifier(test_x).argmax(dim=-1)
        fold_metrics = classification_metrics(test_y, predictions)
        for key in METRIC_KEYS:
            results[key].append(fold_metrics[key])
    return results


def linear_probe_five_fold(
    embeddings: Tensor,
    labels: Tensor,
    device: torch.device,
    seed: int = 42,
    folds: int = 5,
    epochs: int = 200,
    split_seed: int | None = None,
) -> list[float]:
    """兼容旧调用：只返回五折 Accuracy。"""
    return linear_probe_five_fold_metrics(
        embeddings, labels, device=device, seed=seed, split_seed=split_seed,
        folds=folds, epochs=epochs
    )["accuracy"]


def summarize(values: list[float]) -> dict[str, float | list[float]]:
    array = np.asarray(values, dtype=np.float64) * 100
    return {
        "folds": [round(float(value), 4) for value in array],
        "mean": round(float(array.mean()), 4),
        "std": round(float(array.std(ddof=1)), 4),
    }
