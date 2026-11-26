
"""Analiza distribuciones de falsos positivos y verdaderos negativos."""

import pandas as pd

from src.config import ROOT


PREDICTIONS_PATH = (
    ROOT
    / "reports"
    / "predictions"
    / "binary_xgb_oof_predictions.csv"
)

OUTPUT_DIR = ROOT / "reports" / "predictions"

GROUPS = {
    "TCP_FIN_SIN_SERVICIO": {
        "conditions": {
            "proto": "tcp",
            "state": "FIN",
            "service": "-",
        },
        "features": [
            "dpkts",
            "dbytes",
            "sbytes",
            "dur",
            "ct_srv_dst",
            "smean",
        ],
    },
    "UDP_INT_SIN_SERVICIO": {
        "conditions": {
            "proto": "udp",
            "state": "INT",
            "service": "-",
        },
        "features": [
            "ct_srv_dst",
            "smean",
            "sbytes",
            "spkts",
            "ct_dst_src_ltm",
            "ct_srv_src",
        ],
    },
    "TCP_FIN_HTTP": {
        "conditions": {
            "proto": "tcp",
            "state": "FIN",
            "service": "http",
        },
        "features": [
            "sbytes",
            "sloss",
            "spkts",
            "smean",
            "sload",
        ],
    },
}


def load_data() -> pd.DataFrame:
    """Carga las predicciones OOF y filtra trafico normal con sttl=254."""

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
            f"Faltan columnas necesarias: {sorted(missing)}"
        )

    data = df[
        (df["y_true"] == 0)
        & (df["sttl"] == 254)
    ].copy()

    data["is_false_positive"] = (
        data["is_false_positive"].astype(bool)
    )

    return data


def get_group(
    data: pd.DataFrame,
    conditions: dict,
) -> pd.DataFrame:
    """Filtra registros que cumplen todas las condiciones."""

    mask = pd.Series(True, index=data.index)

    for feature, value in conditions.items():
        mask &= data[feature] == value

    return data[mask].copy()


def calculate_statistics(
    values: pd.Series,
    prefix: str,
) -> dict:
    """Calcula estadisticas descriptivas de una variable."""

    values = pd.to_numeric(
        values,
        errors="coerce",
    ).dropna()

    if values.empty:
        return {}

    return {
        f"{prefix}_count": len(values),
        f"{prefix}_mean": values.mean(),
        f"{prefix}_std": values.std(),
        f"{prefix}_min": values.min(),
        f"{prefix}_q10": values.quantile(0.10),
        f"{prefix}_q25": values.quantile(0.25),
        f"{prefix}_median": values.median(),
        f"{prefix}_q75": values.quantile(0.75),
        f"{prefix}_q90": values.quantile(0.90),
        f"{prefix}_max": values.max(),
        f"{prefix}_zero_pct": (
            (values == 0).mean() * 100
        ),
    }


def calculate_iqr_overlap(
    fp: pd.Series,
    tn: pd.Series,
) -> float:
    """Calcula el solapamiento entre los intervalos intercuartilicos."""

    fp_q25 = fp.quantile(0.25)
    fp_q75 = fp.quantile(0.75)

    tn_q25 = tn.quantile(0.25)
    tn_q75 = tn.quantile(0.75)

    intersection = max(
        0,
        min(fp_q75, tn_q75) - max(fp_q25, tn_q25),
    )

    union = max(fp_q75, tn_q75) - min(fp_q25, tn_q25)

    if union == 0:
        return 100.0 if fp_q25 == tn_q25 else 0.0

    return intersection / union * 100


def analyze_feature(
    group_name: str,
    feature: str,
    false_positives: pd.DataFrame,
    true_negatives: pd.DataFrame,
) -> tuple[dict, list[dict]]:
    """Compara estadisticas y distribuciones de una caracteristica."""

    fp = pd.to_numeric(
        false_positives[feature],
        errors="coerce",
    ).dropna()

    tn = pd.to_numeric(
        true_negatives[feature],
        errors="coerce",
    ).dropna()

    if fp.empty or tn.empty:
        return {}, []

    summary = {
        "group": group_name,
        "feature": feature,
    }

    summary.update(
        calculate_statistics(fp, "fp")
    )

    summary.update(
        calculate_statistics(tn, "tn")
    )

    summary["median_difference"] = (
        fp.median() - tn.median()
    )

    summary["iqr_overlap_pct"] = calculate_iqr_overlap(
        fp,
        tn,
    )

    combined = pd.concat(
        [fp, tn],
        ignore_index=True,
    )

    bins = pd.qcut(
        combined,
        q=10,
        duplicates="drop",
    )

    fp_bins = bins.iloc[:len(fp)]
    tn_bins = bins.iloc[len(fp):]

    fp_distribution = fp_bins.value_counts(
        sort=False,
        normalize=True,
    )

    tn_distribution = tn_bins.value_counts(
        sort=False,
        normalize=True,
    )

    distribution_rows = []

    for interval in fp_distribution.index:
        fp_pct = fp_distribution.get(interval, 0) * 100
        tn_pct = tn_distribution.get(interval, 0) * 100

        distribution_rows.append({
            "group": group_name,
            "feature": feature,
            "interval": str(interval),
            "fp_percentage": fp_pct,
            "tn_percentage": tn_pct,
            "difference_percentage_points": fp_pct - tn_pct,
        })

    return summary, distribution_rows


def main() -> None:
    """Ejecuta el analisis de distribuciones."""

    print("\n=== ANALISIS DE DISTRIBUCIONES OOF ===")

    data = load_data()

    summary_rows = []
    distribution_rows = []

    for group_name, configuration in GROUPS.items():
        group = get_group(
            data,
            configuration["conditions"],
        )

        false_positives = group[
            group["is_false_positive"]
        ]

        true_negatives = group[
            ~group["is_false_positive"]
        ]

        print("\n" + "=" * 65)
        print(group_name)
        print("=" * 65)

        print(f"Registros normales: {len(group)}")
        print(f"Falsos positivos: {len(false_positives)}")
        print(f"Verdaderos negativos: {len(true_negatives)}")

        if false_positives.empty or true_negatives.empty:
            print("No hay datos suficientes para comparar.")
            continue

        for feature in configuration["features"]:
            summary, distribution = analyze_feature(
                group_name,
                feature,
                false_positives,
                true_negatives,
            )

            if not summary:
                continue

            summary_rows.append(summary)
            distribution_rows.extend(distribution)

            print(f"\n--- {feature} ---")
            print(
                f"Mediana FP: {summary['fp_median']:.4f} | "
                f"Mediana TN: {summary['tn_median']:.4f}"
            )
            print(
                f"Q25-Q75 FP: "
                f"{summary['fp_q25']:.4f} - "
                f"{summary['fp_q75']:.4f}"
            )
            print(
                f"Q25-Q75 TN: "
                f"{summary['tn_q25']:.4f} - "
                f"{summary['tn_q75']:.4f}"
            )
            print(
                f"Ceros FP: {summary['fp_zero_pct']:.2f}% | "
                f"Ceros TN: {summary['tn_zero_pct']:.2f}%"
            )
            print(
                f"Solapamiento de intervalos intercuartilicos: "
                f"{summary['iqr_overlap_pct']:.2f}%"
            )

    if not summary_rows:
        raise ValueError("No se generaron resultados.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    summary_df = pd.DataFrame(summary_rows)
    distribution_df = pd.DataFrame(distribution_rows)

    summary_path = (
        OUTPUT_DIR / "binary_xgb_oof_distribution_summary.csv"
    )

    distribution_path = (
        OUTPUT_DIR / "binary_xgb_oof_distribution_bins.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    distribution_df.to_csv(
        distribution_path,
        index=False,
    )

    print("\n=== ARCHIVOS GENERADOS ===")
    print(f"Resumen estadistico: {summary_path}")
    print(f"Distribuciones por intervalos: {distribution_path}")


if __name__ == "__main__":
    main()
