
import pandas as pd
import pytest

from src.data import get_X_y


@pytest.fixture
def sample_df():
    """DataFrame pequeño para probar la separación entre X e y."""
    return pd.DataFrame({
        "id": [1, 2, 3],
        "proto": ["tcp", "udp", "tcp"],
        "service": ["http", "-", "ftp"],
        "state": ["FIN", "CON", "INT"],
        "label": [0, 1, 1],
        "attack_cat": ["Normal", "Generic", "Exploits"],
    })


def test_get_X_y_binary(sample_df):
    """La tarea binaria debe utilizar label como target."""
    X, y = get_X_y(sample_df, "binary")

    assert y.equals(sample_df["label"])
    assert "label" not in X.columns
    assert "attack_cat" not in X.columns
    assert "id" not in X.columns


def test_get_X_y_multiclass(sample_df):
    """La tarea multiclase debe utilizar attack_cat como target."""
    X, y = get_X_y(sample_df, "multiclass")

    assert y.equals(sample_df["attack_cat"])
    assert "attack_cat" not in X.columns
    assert "label" not in X.columns
    assert "id" not in X.columns


def test_get_X_y_preserves_input_features(sample_df):
    """Las variables que no son targets ni identificadores deben conservarse."""
    X, _ = get_X_y(sample_df, "binary")

    expected_columns = ["proto", "service", "state"]

    assert list(X.columns) == expected_columns


def test_get_X_y_invalid_task(sample_df):
    """Una tarea inexistente debe generar ValueError."""
    with pytest.raises(ValueError):
        get_X_y(sample_df, "regression")
