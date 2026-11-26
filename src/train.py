
"""Entrenamiento con cross-validation estratificada sobre el train oficial."""

import argparse
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from src.config import (
    FIGURES_DIR,
    METRICS_DIR,
    MODELS_DIR,
    LR_PARAMS,
    SEED,
    XGB_PARAMS,
)
from src.metrics import select_binary_threshold, compute_metrics
from src.baselines import make_baseline
from src.data import get_X_y, load_split
from src.features import build_preprocessor
from src.plots import save_confusion_matrix, save_roc_curve


N_FOLDS = 5

# Características que se pueden excluir en los experimentos.
EXPERIMENTAL_FEATURES = [
    "sttl",
    "dttl",
    "ct_state_ttl",
]


def build_model(task: str, model: str):
    """Devuelve el modelo sin entrenar: regla sin IA o Pipeline."""

    if model == "heuristic":
        return make_baseline(task)

    if model == "lr":
        estimator = LogisticRegression(**LR_PARAMS)
    elif model == "xgb":
        estimator = XGBClassifier(**XGB_PARAMS)
    else:
        raise ValueError(
            f"model debe ser 'heuristic', 'lr' o 'xgb', no {model!r}"
        )

    return Pipeline([
        ("prep", build_preprocessor(model)),
        ("model", estimator),
    ])


def fit_model(clf, model: str, X, y):
    """Entrena el modelo aplicando pesos balanceados cuando corresponde."""

    if model == "xgb":
        weights = compute_sample_weight("balanced", y)
        clf.fit(X, y, model__sample_weight=weights)
    else:
        clf.fit(X, y)

    return clf


def load_xy(task: str):
    """Carga el train y codifica las etiquetas si la tarea es multiclase."""

    X, y = get_X_y(load_split("train"), task)

    encoder = None

    if task == "multiclass":
        encoder = LabelEncoder().fit(y)
        y = pd.Series(
            encoder.transform(y),
            index=y.index,
            name=y.name
        )

    return X, y, encoder


def build_groups(X: pd.DataFrame) -> pd.Series:
    """Asigna el mismo identificador a filas con características idénticas."""

    return X.groupby(
        list(X.columns),
        sort=False,
        dropna=False
    ).ngroup()


def prepare_experiment_features(
    X: pd.DataFrame,
    drop_features: list[str] | None = None
) -> pd.DataFrame:
    """Excluye características seleccionadas para un experimento."""

    if not drop_features:
        return X.copy()

    missing = set(drop_features) - set(X.columns)

    if missing:
        raise ValueError(
            f"Las características indicadas no existen: {sorted(missing)}"
        )

    if len(drop_features) >= len(X.columns):
        raise ValueError(
            "No se pueden eliminar todas las características."
        )

    return X.drop(columns=drop_features)


def experiment_suffix(drop_features: list[str] | None) -> str:
    """Genera un sufijo para guardar cada experimento por separado."""

    if not drop_features:
        return ""

    return "_sin_" + "_".join(drop_features)


def cross_validate(
    task: str,
    model: str,
    drop_features: list[str] | None = None
):
    """
    Ejecuta la CV de 5 folds.

    Los grupos se construyen con las características originales para
    mantener la misma separación entre train y validación en todos
    los experimentos.
    """

    X, y, encoder = load_xy(task)

    # IMPORTANTE:
    # Los grupos se construyen antes de excluir características.
    # Así las variantes mantienen la misma definición de filas duplicadas.
    groups = build_groups(X)

    # Las características se excluyen solamente de la entrada del modelo.
    X = prepare_experiment_features(X, drop_features)

    sgkf = StratifiedGroupKFold(
        n_splits=N_FOLDS,
        shuffle=True,
        random_state=SEED
    )

    fold_metrics = []
    oof_scores = np.zeros(len(y))
    oof_labels = np.full(len(y), -1, dtype=int)

    for k, (train_idx, val_idx) in enumerate(
        sgkf.split(X, y, groups),
        start=1
    ):

        assert not (
            set(groups.iloc[train_idx])
            & set(groups.iloc[val_idx])
        ), "un grupo quedó en train y validación"

        X_tr = X.iloc[train_idx]
        X_val = X.iloc[val_idx]

        y_tr = y.iloc[train_idx]
        y_val = y.iloc[val_idx]

        clf = fit_model(
            build_model(task, model),
            model,
            X_tr,
            y_tr
        )

        y_pred = clf.predict(X_val)
        oof_labels[val_idx] = y_pred

        y_score = (
            clf.predict_proba(X_val)[:, 1]
            if task == "binary"
            else None
        )

        if task == "binary":
            oof_scores[val_idx] = y_score

            m = compute_metrics(
                task,
                y_val,
                y_pred,
                y_score=y_score
            )

            print(
                f"fold {k}: "
                f"f1={m['f1']:.3f}  "
                f"recall={m['recall']:.3f}  "
                f"fpr={m['fpr']:.3f}"
            )

        else:
            m = compute_metrics(
                task,
                encoder.inverse_transform(y_val),
                encoder.inverse_transform(y_pred),
                labels=encoder.classes_
            )

            print(
                f"fold {k}: "
                f"macro_f1={m['macro_f1']:.3f}  "
                f"weighted_f1={m['weighted_f1']:.3f}"
            )

        fold_metrics.append(m)

    return fold_metrics, oof_scores, oof_labels, y, encoder


SUMMARY_KEYS = {
    "binary": ["f1", "recall", "precision", "fpr", "roc_auc", "pr_auc"],
    "multiclass": ["macro_f1", "weighted_f1", "balanced_accuracy"],
}


def run(
    task: str,
    model: str,
    drop_features: list[str] | None = None
) -> None:
    """Ejecuta CV, selecciona umbral y entrena el modelo final."""

    drop_features = list(dict.fromkeys(drop_features or []))
    suffix = experiment_suffix(drop_features)

    print(f"\n=== {task} / {model} ===")

    if drop_features:
        print(f"Características excluidas: {drop_features}")
    else:
        print("Experimento original: se utilizan todas las características.")

    fold_metrics, oof_scores, oof_labels, y, encoder = cross_validate(
        task,
        model,
        drop_features
    )

    # 1. Media y desviación estándar de las métricas de los 5 folds.
    summary = {}

    for key in SUMMARY_KEYS[task]:
        values = [
            m[key]
            for m in fold_metrics
            if m[key] is not None
        ]

        summary[key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values))
        }

        print(
            f"  {key:10s} "
            f"{summary[key]['mean']:.4f} "
            f"± {summary[key]['std']:.4f}"
        )

    # 2. El umbral se elige exclusivamente con predicciones OOF del train.
    threshold = None
    oof_at_threshold = None

    if task == "binary" and model != "heuristic":
        threshold = select_binary_threshold(
            y,
            oof_scores,
            min_recall=0.90
        )

        oof_pred = (oof_scores >= threshold).astype(int)

        oof_at_threshold = compute_metrics(
            "binary",
            y,
            oof_pred,
            y_score=oof_scores
        )

        print(
            f"  umbral OOF = {threshold:.4f} -> "
            f"recall {oof_at_threshold['recall']:.4f}, "
            f"fpr {oof_at_threshold['fpr']:.4f}, "
            f"f1 {oof_at_threshold['f1']:.4f}"
        )

    # 3. Guardado de gráficos OOF con nombres independientes.
    if model != "heuristic":
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)

        if task == "binary":
            plot_threshold = (
                threshold if threshold is not None else 0.5
            )

            oof_pred = (oof_scores >= plot_threshold).astype(int)

            save_confusion_matrix(
                y,
                oof_pred,
                FIGURES_DIR / f"{task}_{model}{suffix}_cm.png",
                f"Matriz de confusión OOF - {model}{suffix}",
                labels=["Normal", "Ataque"]
            )

            save_roc_curve(
                y,
                oof_scores,
                FIGURES_DIR / f"{task}_{model}{suffix}_roc.png",
                f"Curva ROC OOF - {model}{suffix}"
            )

        else:
            y_true_names = encoder.inverse_transform(
                np.asarray(y, dtype=int)
            )

            y_pred_names = encoder.inverse_transform(oof_labels)

            class_names = [
                str(name)
                for name in encoder.classes_
            ]

            save_confusion_matrix(
                y_true_names,
                y_pred_names,
                FIGURES_DIR / f"{task}_{model}{suffix}_cm.png",
                f"Matriz de confusión OOF - {model}{suffix}",
                labels=class_names
            )

    # 4. Entrenamiento final con todo el train.
    # Se aplica la misma exclusión de características del experimento.
    X_full, y_full, encoder = load_xy(task)

    X_full = prepare_experiment_features(
        X_full,
        drop_features
    )

    final = fit_model(
        build_model(task, model),
        model,
        X_full,
        y_full
    )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODELS_DIR / f"{task}_{model}{suffix}.joblib"

    joblib.dump(
        {
            "pipeline": final,
            "threshold": threshold,
            "label_encoder": encoder,
            "dropped_features": drop_features,
        },
        model_path
    )

    print(f"\nModelo guardado en: {model_path}")

    # 5. Guardado de métricas CV en un JSON independiente.
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    result = {
        "task": task,
        "model": model,
        "dropped_features": drop_features,
        "n_folds": N_FOLDS,
        "seed": SEED,
        "cv_summary": summary,
        "threshold": threshold,
        "oof_at_threshold": oof_at_threshold,
        "folds": fold_metrics,
    }

    metrics_path = (
        METRICS_DIR / f"{task}_{model}{suffix}_cv.json"
    )

    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(
            result,
            file,
            indent=2,
            ensure_ascii=False
        )

    print(f"Métricas guardadas en: {metrics_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Cross-validation y entrenamiento experimental"
    )

    parser.add_argument(
        "--task",
        choices=["binary", "multiclass"],
        default="binary"
    )

    parser.add_argument(
        "--model",
        choices=["heuristic", "lr", "xgb"],
        required=True
    )

    parser.add_argument(
        "--drop-features",
        nargs="+",
        choices=EXPERIMENTAL_FEATURES,
        default=None,
        help=(
            "Características que se excluirán del modelo. "
            "Se pueden indicar una o varias."
        )
    )

    args = parser.parse_args()

    run(
        task=args.task,
        model=args.model,
        drop_features=args.drop_features
    )
