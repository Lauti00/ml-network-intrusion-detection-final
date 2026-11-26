
"""Compara caracteristicas numericas dentro de grupos problematicos."""

import pandas as pd

from src.config import ROOT


PREDICTIONS_PATH = (
    ROOT
    / "reports"
    / "predictions"
    / "binary_xgb_oof_predictions.csv"
)

OUTPUT_DIR = ROOT / "reports" / "predictions"

EXCLUDED_COLUMNS = {
    "id",
    "y_true",
    "sttl",
    "probability_attack",
    "predicted_label",
    "is_false_positive",
    "is_false_negative",
    "threshold",
}

GROUPS = {
    "TCP_FIN_SIN_SERVICIO": {
        "proto": "tcp",
        "state": "FIN",
        "service": "-",
    },
    "UDP_INT_SIN_SERVICIO": {
        "proto": "udp",
        "state": "INT",
        "service": "-",
    },
    "TCP_FIN_HTTP": {
        "proto": "tcp",
        "state": "FIN",
        "service": "http",
    },
}


def load_data() -> pd.DataFrame:
    """Carga las predicciones y conserva trafico normal con sttl=254."""

    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(
            f"No se encontro el archivo: {PREDICTIONS_PATH}"
        )

    df = pd.read_csv(PREDICTIONS_PATH)

    required = {
        "y_true",
        "sttl",
        "proto",
        "state",
        "service",
        "is_false_positive",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Faltan columnas: {sorted(missing)}"
        )

    data = df[
        (df["y_true"] == 0)
        & (df["sttl"] == 254)
    ].copy()

    data["is_false_positive"] = (
        data["is_false_positive"].astype(bool)
    )

    return data


def compare_features(
    false_positives: pd.DataFrame,
    true_negatives: pd.DataFrame,
    features: list[str],
    group_name: str,
) -> pd.DataFrame:
    """Compara distribuciones numericas entre ambos grupos."""

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

        fp_iqr = fp.quantile(0.75) - fp.quantile(0.25)
        tn_iqr = tn.quantile(0.75) - tn.quantile(0.25)

        pooled_iqr = (fp_iqr + tn_iqr) / 2

        median_difference = fp_median - tn_median

        # Diferencia de medianas normalizada por la escala
        # de dispersion de ambos grupos.
        normalized_difference = (
            abs(median_difference) / pooled_iqr
            if pooled_iqr > 0
            else 0
        )

        rows.append({
            "group": group_name,
            "feature": feature,
            "fp_count": len(fp),
            "tn_count": len(tn),
            "fp_mean": fp.mean(),
            "tn_mean": tn.mean(),
            "fp_median": fp_median,
            "tn_median": tn_median,
            "median_difference": median_difference,
            "fp_q25": fp.quantile(0.25),
            "fp_q75": fp.quantile(0.75),
            "tn_q25": tn.quantile(0.25),
            "tn_q75": tn.quantile(0.75),
            "normalized_difference": normalized_difference,
        })

    result = pd.DataFrame(rows)

    if not result.empty:
        result = result.sort_values(
            "normalized_difference",
            ascending=False,
        )

    return result


def main() -> None:
    """Ejecuta las comparaciones por grupo."""

    print("\n=== ANALISIS DE PERFILES OOF ===")

    data = load_data()

    # Seleccionamos variables numericas con suficiente variedad.
    numeric_features = [
        column
        for column in data.columns
        if (
            column not in EXCLUDED_COLUMNS
            and pd.api.types.is_numeric_dtype(data[column])
            and data[column].nunique(dropna=False) > 20
        )
    ]

    print(f"\nVariables numericas analizadas: {len(numeric_features)}")

    all_results = []

    for group_name, conditions in GROUPS.items():
        mask = pd.Series(True, index=data.index)

        for feature, value in conditions.items():
            mask &= data[feature] == value

        group = data[mask].copy()

        false_positives = group[
            group["is_false_positive"]
        ]

        true_negatives = group[
            ~group["is_false_positive"]
        ]

        print("\n" + "=" * 70)
        print(group_name)
        print("=" * 70)

        print(f"Registros normales: {len(group)}")
        print(f"Falsos positivos: {len(false_positives)}")
        print(f"Verdaderos negativos: {len(true_negatives)}")

        if false_positives.empty or true_negatives.empty:
            print("No hay suficientes datos en ambos grupos.")
            continue

        result = compare_features(
            false_positives,
            true_negatives,
            numeric_features,
            group_name,
        )

        if result.empty:
            print("No hay variables numericas comparables.")
            continue

        all_results.append(result)

        print("\nDiferencias mas marcadas:")

        print(
            result.head(15).to_string(
                index=False,
                float_format=lambda value: f"{value:.4f}",
            )
        )

    if not all_results:
        raise ValueError("No se generaron resultados.")

    final_result = pd.concat(
        all_results,
        ignore_index=True,
    )

    output_path = (
        OUTPUT_DIR / "binary_xgb_oof_profiles.csv"
    )

    final_result.to_csv(
        output_path,
        index=False,
    )

    print("\n=== ANALISIS FINALIZADO ===")
    print(f"Archivo generado: {output_path}")


if __name__ == "__main__":
    main()
