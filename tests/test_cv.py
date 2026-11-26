
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from src.config import SEED
from src.train import N_FOLDS, build_groups


def test_build_groups_identifies_duplicate_rows():
    """Las filas con características idénticas deben pertenecer al mismo grupo."""
    X = pd.DataFrame({
        "proto": ["tcp", "udp", "tcp", "icmp"],
        "service": ["http", "dns", "http", "-"],
        "state": ["FIN", "CON", "FIN", "INT"],
    })

    groups = build_groups(X)

    assert groups.iloc[0] == groups.iloc[2]


def test_build_groups_separates_different_rows():
    """Las filas con características diferentes deben pertenecer a grupos distintos."""
    X = pd.DataFrame({
        "proto": ["tcp", "udp", "icmp"],
        "service": ["http", "dns", "-"],
        "state": ["FIN", "CON", "INT"],
    })

    groups = build_groups(X)

    assert groups.nunique() == len(X)


def test_cross_validation_keeps_groups_together():
    """Ningún grupo debe aparecer simultáneamente en train y validación."""
    X = pd.DataFrame({
        "proto": ["tcp", "udp", "tcp", "icmp", "udp", "tcp",
                  "icmp", "udp", "tcp", "icmp", "udp", "tcp"],
        "service": ["http", "dns", "http", "-", "ftp", "ftp",
                    "-", "dns", "http", "-", "ftp", "ftp"],
        "state": ["FIN", "CON", "FIN", "INT", "CON", "FIN",
                  "INT", "CON", "FIN", "INT", "CON", "FIN"],
    })

    y = pd.Series([0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0])

    groups = build_groups(X)

    cv = StratifiedGroupKFold(
        n_splits=N_FOLDS,
        shuffle=True,
        random_state=SEED
    )

    for train_idx, val_idx in cv.split(X, y, groups):
        train_groups = set(groups.iloc[train_idx])
        val_groups = set(groups.iloc[val_idx])

        assert train_groups.isdisjoint(val_groups)
