
import pandas as pd
import pytest

from src.inference import validate_input


def test_validate_input_correct_columns():
    """Devuelve las columnas esperadas."""
    df = pd.DataFrame({
        "proto": ["tcp", "udp"],
        "service": ["http", "ftp"],
        "state": ["FIN", "CON"],
    })

    expected = ["proto", "service", "state"]

    result = validate_input(df, expected)

    assert list(result.columns) == expected
    pd.testing.assert_frame_equal(result, df)


def test_validate_input_reorders_columns():
    """Devuelve las columnas en el orden que espera el modelo."""
    df = pd.DataFrame({
        "state": ["FIN", "CON"],
        "proto": ["tcp", "udp"],
        "service": ["http", "ftp"],
    })

    expected = ["proto", "service", "state"]

    result = validate_input(df, expected)

    assert list(result.columns) == expected


def test_validate_input_missing_columns():
    """Genera ValueError cuando faltan columnas necesarias."""
    df = pd.DataFrame({
        "proto": ["tcp", "udp"],
    })

    expected = ["proto", "service", "state"]

    with pytest.raises(ValueError, match="Faltan columnas"):
        validate_input(df, expected)


def test_validate_input_ignores_extra_columns(capsys):
    """Ignora columnas adicionales y muestra un aviso."""
    df = pd.DataFrame({
        "proto": ["tcp"],
        "service": ["http"],
        "state": ["FIN"],
        "extra_column": [123],
    })

    expected = ["proto", "service", "state"]

    result = validate_input(df, expected)

    captured = capsys.readouterr()

    assert list(result.columns) == expected
    assert "extra_column" in captured.out


def test_validate_input_ignores_id_without_warning(capsys):
    """Permite recibir id sin incluirlo en las variables del modelo."""
    df = pd.DataFrame({
        "id": [101, 102],
        "proto": ["tcp", "udp"],
        "service": ["http", "ftp"],
        "state": ["FIN", "CON"],
    })

    expected = ["proto", "service", "state"]

    result = validate_input(df, expected)

    captured = capsys.readouterr()

    assert list(result.columns) == expected
    assert "id" not in result.columns
    assert captured.out == ""
