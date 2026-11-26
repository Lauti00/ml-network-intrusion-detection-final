
"""Analiza la confianza de XGBoost en grupos con sttl=254."""

import pandas as pd

from src.config import ROOT


PREDICTIONS_PATH = (
    ROOT
    / "reports"
    / "predictions"
    / "binary_xgb_oof_predictions.csv"
)

OUTPUT_DIR = ROOT / "reports" / "predictions"


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
        "probability_attack",
        "is_false_positive",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Faltan columnas necesarias: {sorted(missing)}"
        )

    data = df[
        (df["y_true"] == 0)
        & (df["sttl"] == 254)
    ].copy()

    data["is_false_positive"] = (
        data["is_false_positive"].astype(bool)
    )

    data["probability_attack"] = pd.to_numeric(
        data["probability_attack"],
        errors="coerce",
    )

    data = data.dropna(
        subset=["probability_attack"]
    )

    return data


def define_groups(data: pd.DataFrame) -> dict[str, pd.Series]:
    """Define los grupos principales encontrados en el analisis."""

    return {
        "TCP + FIN + servicio -": (
            (data["proto"] == "tcp")
            & (data["state"] == "FIN")
            & (data["service"] == "-")
        ),
        "UDP + INT + servicio -": (
            (data["proto"] == "udp")
            & (data["state"] == "INT")
            & (data["service"] == "-")
        ),
        "TCP + FIN + HTTP": (
            (data["proto"] == "tcp")
            & (data["state"] == "FIN")
            & (data["service"] == "http")
        ),
        "TCP + FIN + FTP": (
            (data["proto"] == "tcp")
            & (data["state"] == "FIN")
            & (data["service"] == "ftp")
        ),
        "TCP + REQ + servicio -": (
            (data["proto"] == "tcp")
            & (data["state"] == "REQ")
            & (data["service"] == "-")
        ),
    }


def summarize_probabilities(
    group: pd.DataFrame,
    group_name: str,
    prediction_type: str,
) -> dict:
    """Calcula estadisticas de probabilidad para un subgrupo."""

    probabilities = group["probability_attack"]

    if probabilities.empty:
        return {}

    return {
        "group": group_name,
        "prediction": prediction_type,
        "records": len(probabilities),
        "mean_probability": probabilities.mean(),
        "median_probability": probabilities.median(),
        "q25": probabilities.quantile(0.25),
        "q75": probabilities.quantile(0.75),
        "minimum": probabilities.min(),
        "maximum": probabilities.max(),
        "above_50_percent": (
            probabilities >= 0.50
        ).mean() * 100,
        "above_80_percent": (
            probabilities >= 0.80
        ).mean() * 100,
        "above_90_percent": (
            probabilities >= 0.90
        ).mean() * 100,
    }


def main() -> None:
    """Ejecuta el analisis de confianza."""

    print("\n=== ANALISIS DE CONFIANZA OOF ===")

    data = load_data()
    groups = define_groups(data)

    print(
        f"Registros normales con sttl=254: {len(data)}"
    )

    results = []

    for group_name, mask in groups.items():
        selected = data[mask]

        false_positives = selected[
            selected["is_false_positive"]
        ]

        true_negatives = selected[
            ~selected["is_false_positive"]
        ]

        print("\n" + "=" * 65)
        print(group_name)
        print("=" * 65)

        print(f"Registros normales: {len(selected)}")
        print(f"Falsos positivos: {len(false_positives)}")
        print(f"Verdaderos negativos: {len(true_negatives)}")

        for prediction_type, subset in [
            ("Falso positivo", false_positives),
            ("Verdadero negativo", true_negatives),
        ]:
            result = summarize_probabilities(
                subset,
                group_name,
                prediction_type,
            )

            if not result:
                print(f"\n{prediction_type}: sin registros")
                continue

            results.append(result)

            print(f"\n{prediction_type}")
            print(f"Cantidad: {result['records']}")
            print(
                f"Probabilidad media: "
                f"{result['mean_probability']:.4f}"
            )
            print(
                f"Probabilidad mediana: "
                f"{result['median_probability']:.4f}"
            )
            print(
                f"Percentil 25: {result['q25']:.4f}"
            )
            print(
                f"Percentil 75: {result['q75']:.4f}"
            )
            print(
                f"Probabilidad >= 0.50: "
                f"{result['above_50_percent']:.2f}%"
            )
            print(
                f"Probabilidad >= 0.80: "
                f"{result['above_80_percent']:.2f}%"
            )
            print(
                f"Probabilidad >= 0.90: "
                f"{result['above_90_percent']:.2f}%"
            )

    if not results:
        raise ValueError("No se generaron resultados.")

    result_df = pd.DataFrame(results)

    output_path = (
        OUTPUT_DIR / "binary_xgb_oof_confidence.csv"
    )

    result_df.to_csv(
        output_path,
        index=False,
    )

    print("\n=== ARCHIVO GENERADO ===")
    print(output_path)


if __name__ == "__main__":
    main()
