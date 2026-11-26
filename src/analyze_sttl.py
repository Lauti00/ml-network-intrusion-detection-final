
"""Compara falsos positivos y verdaderos negativos con sttl=254."""

from pathlib import Path

import pandas as pd

from src.config import ROOT


PREDICTIONS_PATH = (
    ROOT
    / "reports"
    / "predictions"
    / "binary_xgb_oof_predictions.csv"
)

OUTPUT_DIR = ROOT / "reports" / "predictions"

# Columnas que no son caracteristicas originales del trafico.
EXCLUDED_COLUMNS = {
    "id",
    "y_true",
    "probability_attack",
    "predicted_label",
    "is_false_positive",
    "is_false_negative",
    "threshold",
    "sttl",
}


def load_data() -> pd.DataFrame:
    """Carga las predicciones OOF y separa los dos grupos a comparar."""

    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(
            f"No se encontro el archivo: {PREDICTIONS_PATH}"
        )

    df = pd.read_csv(PREDICTIONS_PATH)

    required_columns = {
        "y_true",
        "sttl",
        "is_false_positive",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Faltan columnas necesarias: {sorted(missing)}"
        )

    # Nos quedamos solamente con trafico realmente normal
    # y con sttl igual a 254.
    normal_sttl = df[
        (df["y_true"] == 0)
        & (df["sttl"] == 254)
    ].copy()

    # Grupo 1: el modelo confundio trafico normal con un ataque.
    false_positives = normal_sttl[
        normal_sttl["is_false_positive"].astype(bool)
    ].copy()

    # Grupo 2: el modelo reconocio correctamente el trafico normal.
    true_negatives = normal_sttl[
        ~normal_sttl["is_false_positive"].astype(bool)
    ].copy()

    print("\n=== GRUPOS A COMPARAR ===")
    print(f"Falsos positivos: {len(false_positives)}")
    print(f"Verdaderos negativos: {len(true_negatives)}")

    if false_positives.empty or true_negatives.empty:
        raise ValueError(
            "Uno de los grupos esta vacio. No se puede comparar."
        )

    return normal_sttl


def analyze_numeric(
    false_positives: pd.DataFrame,
    true_negatives: pd.DataFrame,
    features: list[str],
) -> pd.DataFrame:
    """Compara distribuciones de caracteristicas numericas."""

    rows = []

    for feature in features:
        fp = pd.to_numeric(
            false_positives[feature],
            errors="coerce",
        ).dropna()

        tn = pd.to_numeric(
            true_negatives[feature],
            errors="coerce",
        ).dropna()

        if fp.empty or tn.empty:
            continue

        fp_median = fp.median()
        tn_median = tn.median()

        fp_mean = fp.mean()
        tn_mean = tn.mean()

        rows.append({
            "feature": feature,
            "fp_count": len(fp),
            "tn_count": len(tn),
            "fp_mean": fp_mean,
            "tn_mean": tn_mean,
            "difference_mean": fp_mean - tn_mean,
            "fp_median": fp_median,
            "tn_median": tn_median,
            "difference_median": fp_median - tn_median,
            "fp_q25": fp.quantile(0.25),
            "fp_q75": fp.quantile(0.75),
            "tn_q25": tn.quantile(0.25),
            "tn_q75": tn.quantile(0.75),
        })

    result = pd.DataFrame(rows)

    if not result.empty:
        result["absolute_median_difference"] = (
            result["difference_median"].abs()
        )

        result = result.sort_values(
            "absolute_median_difference",
            ascending=False,
        )

    return result


def analyze_categorical(
    false_positives: pd.DataFrame,
    true_negatives: pd.DataFrame,
    features: list[str],
) -> pd.DataFrame:
    """Compara la frecuencia de categorias entre ambos grupos."""

    rows = []

    for feature in features:
        fp_values = false_positives[feature].fillna(
            "<NULO>"
        ).astype(str)

        tn_values = true_negatives[feature].fillna(
            "<NULO>"
        ).astype(str)

        fp_counts = fp_values.value_counts()
        tn_counts = tn_values.value_counts()

        # Consideramos las diez categorias mas frecuentes
        # sumando ambos grupos.
        combined_counts = fp_counts.add(
            tn_counts,
            fill_value=0,
        )

        top_categories = combined_counts.nlargest(10).index

        for category in top_categories:
            fp_count = int(fp_counts.get(category, 0))
            tn_count = int(tn_counts.get(category, 0))

            fp_percentage = (
                fp_count / len(fp_values) * 100
            )

            tn_percentage = (
                tn_count / len(tn_values) * 100
            )

            rows.append({
                "feature": feature,
                "category": category,
                "fp_count": fp_count,
                "fp_percentage": fp_percentage,
                "tn_count": tn_count,
                "tn_percentage": tn_percentage,
                "difference_percentage_points": (
                    fp_percentage - tn_percentage
                ),
            })

    result = pd.DataFrame(rows)

    if not result.empty:
        result["absolute_difference"] = (
            result["difference_percentage_points"].abs()
        )

        result = result.sort_values(
            "absolute_difference",
            ascending=False,
        )

    return result


def main() -> None:
    """Ejecuta el analisis comparativo."""

    print("\n=== ANALISIS DE TRAFICO CON STTL=254 ===")

    data = load_data()

    false_positives = data[
        data["is_false_positive"].astype(bool)
    ]

    true_negatives = data[
        ~data["is_false_positive"].astype(bool)
    ]

    features = [
        column
        for column in data.columns
        if column not in EXCLUDED_COLUMNS
    ]

    # Separamos variables numericas y categoricas.
    numeric_features = []
    categorical_features = []

    for feature in features:
        if pd.api.types.is_numeric_dtype(data[feature]):
            unique_count = data[feature].nunique(dropna=False)

            # Variables numericas con pocos valores distintos
            # se analizan como categorias.
            if unique_count <= 20:
                categorical_features.append(feature)
            else:
                numeric_features.append(feature)
        else:
            categorical_features.append(feature)

    print("\n=== CARACTERISTICAS DETECTADAS ===")
    print(f"Numericas: {len(numeric_features)}")
    print(f"Categoricas: {len(categorical_features)}")

    # Analisis numerico.
    numeric_result = analyze_numeric(
        false_positives,
        true_negatives,
        numeric_features,
    )

    # Analisis categorico.
    categorical_result = analyze_categorical(
        false_positives,
        true_negatives,
        categorical_features,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    numeric_path = (
        OUTPUT_DIR / "binary_xgb_sttl254_numeric.csv"
    )

    categorical_path = (
        OUTPUT_DIR / "binary_xgb_sttl254_categorical.csv"
    )

    numeric_result.to_csv(
        numeric_path,
        index=False,
    )

    categorical_result.to_csv(
        categorical_path,
        index=False,
    )

    # Mostrar los resultados mas relevantes.
    print("\n=== DIFERENCIAS NUMERICAS ===")

    if numeric_result.empty:
        print("No se encontraron variables numericas para comparar.")
    else:
        print(
            numeric_result.head(15).to_string(
                index=False,
                float_format=lambda value: f"{value:.4f}",
            )
        )

    print("\n=== DIFERENCIAS CATEGORICAS ===")

    if categorical_result.empty:
        print("No se encontraron variables categoricas para comparar.")
    else:
        print(
            categorical_result.head(20).to_string(
                index=False,
                float_format=lambda value: f"{value:.2f}",
            )
        )

    print("\n=== ARCHIVOS GENERADOS ===")
    print(f"Analisis numerico: {numeric_path}")
    print(f"Analisis categorico: {categorical_path}")


if __name__ == "__main__":
    main()
