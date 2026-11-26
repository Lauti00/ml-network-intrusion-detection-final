
"""Analiza combinaciones de protocolo, estado y servicio en sttl=254."""

import pandas as pd

from src.config import ROOT


PREDICTIONS_PATH = (
    ROOT
    / "reports"
    / "predictions"
    / "binary_xgb_oof_predictions.csv"
)

OUTPUT_DIR = ROOT / "reports" / "predictions"

MIN_RECORDS = 30

FEATURE_GROUPS = [
    ["proto"],
    ["state"],
    ["service"],
    ["proto", "state"],
    ["proto", "service"],
    ["state", "service"],
    ["proto", "state", "service"],
]


def load_data() -> pd.DataFrame:
    """Carga los registros normales con sttl=254."""

    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(
            f"No se encontro el archivo: {PREDICTIONS_PATH}"
        )

    df = pd.read_csv(PREDICTIONS_PATH)

    required_columns = {
        "y_true",
        "sttl",
        "is_false_positive",
        "proto",
        "state",
        "service",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Faltan columnas necesarias: {sorted(missing)}"
        )

    # Filtramos exclusivamente trafico realmente normal
    # con sttl igual a 254.
    data = df[
        (df["y_true"] == 0)
        & (df["sttl"] == 254)
    ].copy()

    if data.empty:
        raise ValueError(
            "No se encontraron registros normales con sttl=254."
        )

    # Normalizamos la columna para trabajar con valores booleanos.
    data["is_false_positive"] = (
        data["is_false_positive"]
        .astype(bool)
    )

    print("\n=== DATOS DEL ANALISIS ===")
    print(f"Registros normales con sttl=254: {len(data)}")
    print(
        "Falsos positivos:",
        int(data["is_false_positive"].sum()),
    )
    print(
        "Verdaderos negativos:",
        int((~data["is_false_positive"]).sum()),
    )

    return data


def analyze_group(
    data: pd.DataFrame,
    features: list[str],
) -> pd.DataFrame:
    """Calcula la tasa de falsos positivos para cada combinacion."""

    result = (
        data.groupby(
            features,
            dropna=False,
        )
        .agg(
            total_records=("is_false_positive", "size"),
            false_positives=("is_false_positive", "sum"),
        )
        .reset_index()
    )

    result["true_negatives"] = (
        result["total_records"]
        - result["false_positives"]
    )

    # FPR = falsos positivos / total de registros normales.
    result["fpr"] = (
        result["false_positives"]
        / result["total_records"]
    )

    # Porcentaje de todos los falsos positivos que pertenece
    # a esta combinacion.
    total_fp = int(data["is_false_positive"].sum())

    result["fp_share"] = (
        result["false_positives"] / total_fp
        if total_fp > 0
        else 0
    )

    # Excluimos grupos con muy pocos registros.
    result = result[
        result["total_records"] >= MIN_RECORDS
    ].copy()

    # Ordenamos por FPR, pero conservamos los conteos
    # para poder interpretar el soporte de cada grupo.
    result = result.sort_values(
        ["fpr", "total_records"],
        ascending=[False, False],
    )

    return result


def main() -> None:
    """Ejecuta el analisis de combinaciones."""

    print("\n=== ANALISIS DE COMBINACIONES: STTL=254 ===")

    data = load_data()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_results = []

    for features in FEATURE_GROUPS:
        print("\n" + "=" * 65)
        print("COMBINACION:", " + ".join(features))
        print("=" * 65)

        result = analyze_group(data, features)

        if result.empty:
            print("No hay combinaciones con suficientes registros.")
            continue

        # Agregamos informacion sobre que variables se analizaron.
        result.insert(
            0,
            "analysis",
            " + ".join(features),
        )

        all_results.append(result)

        # Mostramos las 15 combinaciones con mayor FPR.
        display_columns = [
            *features,
            "total_records",
            "false_positives",
            "true_negatives",
            "fpr",
            "fp_share",
        ]

        print(
            result[display_columns]
            .head(15)
            .to_string(
                index=False,
                formatters={
                    "fpr": lambda value: f"{value * 100:.2f}%",
                    "fp_share": lambda value: f"{value * 100:.2f}%",
                },
            )
        )

    if not all_results:
        raise ValueError(
            "No se encontraron combinaciones para analizar."
        )

    final_result = pd.concat(
        all_results,
        ignore_index=True,
    )

    output_path = (
        OUTPUT_DIR
        / "binary_xgb_sttl254_combinations.csv"
    )

    final_result.to_csv(
        output_path,
        index=False,
    )

    print("\n=== ANALISIS FINALIZADO ===")
    print(f"Archivo generado: {output_path}")
    print(f"Combinaciones analizadas: {len(final_result)}")
    print(f"Minimo de registros por grupo: {MIN_RECORDS}")


if __name__ == "__main__":
    main()
