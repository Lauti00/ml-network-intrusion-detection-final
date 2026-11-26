
"""Analisis de predicciones OOF para estudiar los falsos positivos."""

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import METRICS_DIR
from src.data import get_X_y, load_split
from src.metrics import select_binary_threshold
from src.train import cross_validate


def main() -> None:
    """Reconstruye predicciones OOF y analiza falsos positivos."""

    print("\n=== ANALISIS OOF: BINARY / XGBOOST ===")

    # 1. Cargar el conjunto oficial de entrenamiento.
    df = load_split("train")
    X, y = get_X_y(df, "binary")

    print(f"Registros cargados: {len(df)}")

    # 2. Reconstruir las predicciones OOF.
    # Se utilizan los mismos folds, semilla y parametros de train.py.
    _, oof_scores, _, y_cv, _ = cross_validate(
        task="binary",
        model="xgb",
        drop_features=None,
    )

    if not y.index.equals(y_cv.index):
        raise ValueError("Los indices de las etiquetas no coinciden.")

    if np.any(oof_scores < 0) or np.any(oof_scores > 1):
        raise ValueError("Se encontraron probabilidades fuera de [0, 1].")

    # 3. Seleccionar el mismo umbral utilizado en el entrenamiento.
    threshold = select_binary_threshold(
        y,
        oof_scores,
        min_recall=0.90,
    )

    y_pred = (oof_scores >= threshold).astype(int)

    print(f"\nUmbral OOF: {threshold:.10f}")

    # 4. Construir el archivo con las predicciones y caracteristicas.
    predictions = X.copy()

    predictions.insert(0, "id", df["id"].to_numpy())

    predictions["y_true"] = y.to_numpy()
    predictions["probability_attack"] = oof_scores
    predictions["predicted_label"] = y_pred
    predictions["is_false_positive"] = (
        (y.to_numpy() == 0) & (y_pred == 1)
    )
    predictions["is_false_negative"] = (
        (y.to_numpy() == 1) & (y_pred == 0)
    )
    predictions["threshold"] = threshold

    # 5. Crear directorio de salida.
    output_dir = METRICS_DIR.parent / "predictions"
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions_path = (
        output_dir / "binary_xgb_oof_predictions.csv"
    )

    predictions.to_csv(
        predictions_path,
        index=False,
    )

    # 6. Separar los registros normales.
    # Los falsos positivos son registros normales clasificados como ataque.
    normal = predictions[predictions["y_true"] == 0].copy()

    false_positives = normal[
        normal["is_false_positive"]
    ]

    true_negatives = normal[
        ~normal["is_false_positive"]
    ]

    total_fp = len(false_positives)

    if total_fp == 0:
        raise ValueError("No se encontraron falsos positivos.")

    # 7. Analizar especificamente sttl = 254.
    fp_sttl_254 = (
        false_positives["sttl"] == 254
    ).sum()

    normal_sttl_254 = (
        normal["sttl"] == 254
    ).sum()

    tn_sttl_254 = (
        true_negatives["sttl"] == 254
    ).sum()

    percentage_fp_254 = (
        fp_sttl_254 / total_fp
    ) * 100

    fpr_sttl_254 = (
        fp_sttl_254 / normal_sttl_254 * 100
        if normal_sttl_254 > 0
        else 0
    )

    normal_other = normal[
        normal["sttl"] != 254
    ]

    fp_other = normal_other[
        normal_other["is_false_positive"]
    ]

    fpr_other = (
        len(fp_other) / len(normal_other) * 100
        if len(normal_other) > 0
        else 0
    )

    # 8. Agrupar los registros normales por valor de sttl.
    sttl_summary = (
        normal.groupby("sttl", dropna=False)
        .agg(
            normal_records=("y_true", "size"),
            false_positives=("is_false_positive", "sum"),
        )
        .reset_index()
    )

    sttl_summary["fpr"] = (
        sttl_summary["false_positives"]
        / sttl_summary["normal_records"]
    )

    sttl_summary = sttl_summary.sort_values(
        "false_positives",
        ascending=False,
    )

    summary_path = (
        output_dir / "binary_xgb_oof_fpr_by_sttl.csv"
    )

    sttl_summary.to_csv(
        summary_path,
        index=False,
    )

    # 9. Mostrar resultados.
    print("\n=== RESULTADOS GENERALES ===")
    print(f"Registros normales: {len(normal)}")
    print(f"Verdaderos negativos: {len(true_negatives)}")
    print(f"Falsos positivos: {total_fp}")
    print(f"Registros con sttl=254 entre los normales: {normal_sttl_254}")
    print(f"Verdaderos negativos con sttl=254: {tn_sttl_254}")

    print("\n=== ANALISIS DE STTL=254 ===")
    print(
        f"Falsos positivos con sttl=254: "
        f"{fp_sttl_254}"
    )
    print(
        f"Porcentaje de falsos positivos con sttl=254: "
        f"{percentage_fp_254:.2f}%"
    )
    print(
        f"FPR para registros normales con sttl=254: "
        f"{fpr_sttl_254:.2f}%"
    )
    print(
        f"FPR para registros normales con sttl distinto de 254: "
        f"{fpr_other:.2f}%"
    )

    print("\n=== ARCHIVOS GENERADOS ===")
    print(f"Predicciones completas: {predictions_path}")
    print(f"Resumen por sttl: {summary_path}")


if __name__ == "__main__":
    main()
