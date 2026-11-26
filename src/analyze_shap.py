
"""Explicación SHAP OOF para tráfico normal con sttl=254.

Cada registro se explica con el modelo del fold en cuya validación quedó.
Se comparan falsos positivos y verdaderos negativos dentro de tres grupos.

El archivo de detalle conserva:
- El ID original del dataset.
- La posición de la fila en el CSV.
- Los valores originales de las características.
- Las contribuciones SHAP de cada característica.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import StratifiedGroupKFold

from src.config import CATEGORICAL_COLS, SEED
from src.data import load_split
from src.train import (
    N_FOLDS,
    build_groups,
    build_model,
    fit_model,
    load_xy,
)


# ---------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

PREDICTIONS_DIR = ROOT / "reports" / "predictions"

OOF_PATH = PREDICTIONS_DIR / "binary_xgb_oof_predictions.csv"

DETAIL_PATH = (
    PREDICTIONS_DIR / "binary_xgb_sttl254_shap_details.csv"
)

SUMMARY_PATH = (
    PREDICTIONS_DIR / "binary_xgb_sttl254_shap_summary.csv"
)

MAX_PER_CLASS_PER_FOLD = 20
TARGET_STTL = 254

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


# ---------------------------------------------------------------------
# Selección de registros
# ---------------------------------------------------------------------

def get_group_mask(
    X: pd.DataFrame,
    group: dict,
) -> pd.Series:
    """Filtra por sttl=254 y por la combinación indicada."""

    mask = X["sttl"].eq(TARGET_STTL)

    for column, value in group.items():
        mask &= X[column].astype(str).eq(value)

    return mask


def select_positions(
    X_val: pd.DataFrame,
    y_val: pd.Series,
    probabilities: np.ndarray,
    threshold: float,
    group: dict,
    rng: np.random.Generator,
) -> dict:
    """Selecciona falsos positivos y verdaderos negativos del grupo."""

    predictions = (probabilities >= threshold).astype(int)

    group_mask = get_group_mask(
        X_val,
        group,
    ).to_numpy()

    normal_mask = y_val.to_numpy() == 0

    fp_mask = group_mask & normal_mask & (predictions == 1)
    tn_mask = group_mask & normal_mask & (predictions == 0)

    fp_positions = np.flatnonzero(fp_mask)
    tn_positions = np.flatnonzero(tn_mask)

    if len(fp_positions) > MAX_PER_CLASS_PER_FOLD:
        fp_positions = rng.choice(
            fp_positions,
            size=MAX_PER_CLASS_PER_FOLD,
            replace=False,
        )

    if len(tn_positions) > MAX_PER_CLASS_PER_FOLD:
        tn_positions = rng.choice(
            tn_positions,
            size=MAX_PER_CLASS_PER_FOLD,
            replace=False,
        )

    return {
        "FP": np.sort(fp_positions),
        "TN": np.sort(tn_positions),
    }


# ---------------------------------------------------------------------
# Agrupación de columnas SHAP transformadas
# ---------------------------------------------------------------------

def build_feature_mapping(
    original_features: list[str],
    transformed_features: list[str],
) -> dict[str, list[int]]:
    """Relaciona columnas transformadas con variables originales."""

    mapping = {}

    for feature in original_features:

        if feature in CATEGORICAL_COLS:
            prefix = f"cat__{feature}_"

            positions = [
                i
                for i, transformed in enumerate(transformed_features)
                if transformed.startswith(prefix)
            ]

        else:
            expected_name = f"num__{feature}"

            positions = [
                i
                for i, transformed in enumerate(transformed_features)
                if transformed == expected_name
            ]

        if not positions:
            raise ValueError(
                f"No se encontró la columna transformada "
                f"correspondiente a {feature!r}."
            )

        mapping[feature] = positions

    return mapping


def group_shap_values(
    shap_values: np.ndarray,
    feature_mapping: dict[str, list[int]],
) -> dict[str, np.ndarray]:
    """Suma las contribuciones de columnas transformadas."""

    grouped = {}

    for feature, positions in feature_mapping.items():
        grouped[feature] = shap_values[:, positions].sum(axis=1)

    return grouped


# ---------------------------------------------------------------------
# Ejecución principal
# ---------------------------------------------------------------------

def main() -> None:

    print("\n=== ANALISIS SHAP: STTL=254 ===")

    if not OOF_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo OOF: {OOF_PATH}"
        )

    # Leer el umbral utilizado en el análisis OOF anterior.
    oof = pd.read_csv(OOF_PATH, nrows=1)

    if "threshold" not in oof.columns:
        raise ValueError(
            "El archivo OOF no contiene la columna threshold."
        )

    threshold = float(oof["threshold"].iloc[0])

    print(f"Versión SHAP: {shap.__version__}")
    print(f"STTL analizado: {TARGET_STTL}")
    print(f"Umbral utilizado: {threshold:.6f}")
    print(
        "Máximo por clase y grupo en cada fold:",
        MAX_PER_CLASS_PER_FOLD,
    )

    # Cargar el dataset original.
    train_df = load_split("train")

    # Recuperar X e y mediante la función habitual del proyecto.
    X, y, _ = load_xy("binary")

    # Comprobar que los datos originales y las características
    # conservan el mismo orden y cantidad de filas.
    if len(train_df) != len(X):
        raise ValueError(
            "El dataset original y X tienen distinta cantidad de filas."
        )

    if not train_df.index.equals(X.index):
        raise ValueError(
            "Los índices del dataset original y X no coinciden."
        )

    if "id" not in train_df.columns:
        raise ValueError(
            "El dataset original no contiene la columna id."
        )

    if "sttl" not in X.columns:
        raise ValueError(
            "No existe la característica sttl."
        )

    required_columns = set()

    for group_definition in GROUPS.values():
        required_columns.update(group_definition.keys())

    if not required_columns.issubset(X.columns):
        missing = required_columns - set(X.columns)

        raise ValueError(
            f"Faltan columnas necesarias para definir los grupos: {missing}"
        )

    groups = build_groups(X)

    print("\nCaracterísticas originales:", len(X.columns))
    print("Registros de entrenamiento:", len(X))

    print(
        "Registros con sttl=254:",
        int(X["sttl"].eq(TARGET_STTL).sum()),
    )

    splitter = StratifiedGroupKFold(
        n_splits=N_FOLDS,
        shuffle=True,
        random_state=SEED,
    )

    rng = np.random.default_rng(SEED)

    detail_rows = []

    # Reconstruir los cinco folds.
    for fold, (train_idx, val_idx) in enumerate(
        splitter.split(X, y, groups),
        start=1,
    ):

        print(f"\n--- FOLD {fold}/{N_FOLDS} ---")

        X_train = X.iloc[train_idx]
        X_val = X.iloc[val_idx]

        y_train = y.iloc[train_idx]
        y_val = y.iloc[val_idx]

        # Entrenar exclusivamente con el train del fold.
        clf = fit_model(
            build_model("binary", "xgb"),
            "xgb",
            X_train,
            y_train,
        )

        probabilities = clf.predict_proba(X_val)[:, 1]

        # Seleccionar registros de cada grupo.
        selected = {}

        for group_name, group_definition in GROUPS.items():

            selected[group_name] = select_positions(
                X_val,
                y_val,
                probabilities,
                threshold,
                group_definition,
                rng,
            )

        # Preparar los registros seleccionados.
        fold_records = []

        for group_name, class_positions in selected.items():

            for class_name, positions in class_positions.items():

                for position in positions:

                    fold_records.append({
                        "position": int(position),
                        "group": group_name,
                        "prediction_type": class_name,
                    })

        if not fold_records:
            print(
                "No se encontraron registros para explicar en este fold."
            )
            continue

        selected_positions = [
            record["position"]
            for record in fold_records
        ]

        X_selected = X_val.iloc[selected_positions]

        # Aplicar el preprocesador ajustado solamente con X_train.
        preprocessor = clf.named_steps["prep"]
        estimator = clf.named_steps["model"]

        X_transformed = preprocessor.transform(X_selected)

        transformed_names = (
            preprocessor.get_feature_names_out().tolist()
        )

        feature_mapping = build_feature_mapping(
            X.columns.tolist(),
            transformed_names,
        )

        # Calcular SHAP.
        explainer = shap.TreeExplainer(estimator)
        explanation = explainer(X_transformed)

        shap_values = np.asarray(explanation.values)

        # Compatibilidad con diferentes formas de salida.
        if shap_values.ndim == 3:
            shap_values = shap_values[:, :, 1]

        if shap_values.ndim != 2:
            raise ValueError(
                f"Forma SHAP inesperada: {shap_values.shape}"
            )

        grouped_values = group_shap_values(
            shap_values,
            feature_mapping,
        )

        base_values = np.asarray(explanation.base_values)

        if base_values.ndim == 0:
            base_values = np.full(
                len(selected_positions),
                float(base_values),
            )

        elif base_values.ndim > 1:
            base_values = base_values[:, -1]

        # Guardar una fila por registro explicado.
        for i, record in enumerate(fold_records):

            local_position = record["position"]

            global_position = int(
                val_idx[local_position]
            )

            original_row = X.iloc[global_position]
            raw_row = train_df.iloc[global_position]

            row = {
                "fold": fold,
                "id": int(raw_row["id"]),
                "row_position": global_position,
                "group": record["group"],
                "prediction_type": record["prediction_type"],
                "y_true": int(y.iloc[global_position]),
                "probability_attack": float(
                    probabilities[local_position]
                ),
                "threshold": threshold,
                "base_value": float(base_values[i]),
            }

            # Guardar los valores originales de las 42 características.
            for feature in X.columns:
                row[feature] = original_row[feature]

            # Guardar las contribuciones SHAP agrupadas por
            # característica original.
            for feature, values in grouped_values.items():
                row[f"shap_{feature}"] = float(values[i])

            detail_rows.append(row)

        print(
            f"Registros explicados en fold: {len(fold_records)}"
        )

        for group_name, class_positions in selected.items():

            print(
                f"  {group_name}: "
                f"FP={len(class_positions['FP'])}, "
                f"TN={len(class_positions['TN'])}"
            )

    if not detail_rows:
        raise RuntimeError(
            "No se generaron explicaciones SHAP."
        )

    # -----------------------------------------------------------------
    # Guardar detalle
    # -----------------------------------------------------------------

    details = pd.DataFrame(detail_rows)

    PREDICTIONS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    details.to_csv(
        DETAIL_PATH,
        index=False,
        encoding="utf-8",
    )

    print("\nDetalle SHAP guardado:")
    print(DETAIL_PATH)

    # -----------------------------------------------------------------
    # Crear resumen
    # -----------------------------------------------------------------

    shap_columns = [
        column
        for column in details.columns
        if column.startswith("shap_")
    ]

    summary_rows = []

    for (group_name, prediction_type), subset in details.groupby(
        ["group", "prediction_type"],
        sort=True,
    ):

        for column in shap_columns:

            values = subset[column].to_numpy()

            summary_rows.append({
                "group": group_name,
                "prediction_type": prediction_type,
                "feature": column.replace("shap_", "", 1),
                "n_records": len(subset),
                "mean_abs_shap": float(
                    np.mean(np.abs(values))
                ),
                "mean_shap": float(
                    np.mean(values)
                ),
            })

    summary = pd.DataFrame(summary_rows)

    summary = summary.sort_values(
        [
            "group",
            "prediction_type",
            "mean_abs_shap",
        ],
        ascending=[True, True, False],
    )

    summary.to_csv(
        SUMMARY_PATH,
        index=False,
        encoding="utf-8",
    )

    print("\nResumen SHAP guardado:")
    print(SUMMARY_PATH)

    # -----------------------------------------------------------------
    # Mostrar resultados
    # -----------------------------------------------------------------

    print("\n=== CANTIDAD DE REGISTROS EXPLICADOS ===")

    counts = (
        details.groupby(
            ["group", "prediction_type"]
        )
        .size()
        .unstack(fill_value=0)
    )

    print(counts.to_string())

    print("\n=== VERIFICACIÓN STTL ===")

    print(
        details.groupby(
            ["group", "sttl"]
        ).size().to_string()
    )

    print("\n=== VERIFICACIÓN DE VALORES ORIGINALES ===")

    for feature in ["ct_srv_src", "ct_srv_dst", "sbytes"]:
        print(
            f"{feature}: "
            f"{'OK' if feature in details.columns else 'FALTA'}"
        )

    print("\n=== TOP 10 CARACTERÍSTICAS ===")

    for (group_name, prediction_type), subset in summary.groupby(
        ["group", "prediction_type"],
        sort=True,
    ):

        print(
            f"\n{group_name} / {prediction_type}"
        )

        print(
            subset.head(10)[
                [
                    "feature",
                    "mean_abs_shap",
                    "mean_shap",
                ]
            ].to_string(index=False)
        )

    print("\nAnálisis SHAP con sttl=254 finalizado.")


if __name__ == "__main__":
    main()
