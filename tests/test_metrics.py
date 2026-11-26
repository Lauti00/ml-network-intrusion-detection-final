
import numpy as np
import pytest

from src.metrics import select_binary_threshold


def test_threshold_respects_minimum_recall():
    """El umbral elegido debe alcanzar el recall mínimo."""
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.4, 0.35, 0.8])

    threshold = select_binary_threshold(
        y_true,
        y_score,
        min_recall=0.90
    )

    y_pred = (y_score >= threshold).astype(int)

    true_positives = np.sum((y_true == 1) & (y_pred == 1))
    actual_positives = np.sum(y_true == 1)
    recall = true_positives / actual_positives

    assert recall >= 0.90


def test_threshold_selects_best_f1():
    """Entre los umbrales válidos debe elegir el de mejor F1."""
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.4, 0.35, 0.8])

    threshold = select_binary_threshold(
        y_true,
        y_score,
        min_recall=0.90
    )

    assert threshold == pytest.approx(0.35)


def test_threshold_tie_selects_higher_threshold():
    """Si dos umbrales tienen el mismo F1, debe elegir el más alto."""
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([2, 3, 1, 4])

    threshold = select_binary_threshold(
        y_true,
        y_score,
        min_recall=0.50
    )

    assert threshold == pytest.approx(4)


def test_threshold_rejects_single_class():
    """Debe rechazar datos que contienen una sola clase."""
    y_true = np.array([1, 1, 1])
    y_score = np.array([0.2, 0.5, 0.9])

    with pytest.raises(ValueError):
        select_binary_threshold(y_true, y_score)


def test_threshold_rejects_invalid_min_recall():
    """El recall mínimo debe estar entre 0 y 1."""
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.2, 0.7, 0.9])

    with pytest.raises(ValueError, match="min_recall"):
        select_binary_threshold(
            y_true,
            y_score,
            min_recall=1.5
        )
