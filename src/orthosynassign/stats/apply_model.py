"""
Apply the calibrated logistic regression model to true edge genes.

Reads the calibration JSON produced by :mod:`orthosynassign.stats.calibrate`
and applies it to edge genes (``edge_type != "internal"``) in
``sog_gene_edge_long.csv``.

Outputs ``sog_gene_edge_scored.csv`` with three additional columns:

``split_probability``
    Predicted probability that the SOG assignment is an artifact split.

``split_confidence``
    ``1 - split_probability``: higher values indicate a more reliable assignment.

``rescue_flag``
    Boolean flag (0/1): 1 when ``split_probability < threshold`` — the gene is
    classified as a potential rescue candidate (the split looks like an artifact).

Usage::

    python -m orthosynassign.stats.apply_model \\
        --table sog_gene_edge_long.csv \\
        --calibration calibration.json \\
        --output sog_gene_edge_scored.csv \\
        --threshold_type f1
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_EPSILON = 1e-9
_THRESHOLD_TYPES = ("f1", "recall")


def apply_model(
    table_path: str | Path,
    calibration_path: str | Path,
    output_path: str | Path = "sog_gene_edge_scored.csv",
    threshold_type: str = "f1",
) -> None:
    """Score edge genes using the calibrated logistic regression model.

    Args:
        table_path: Path to ``sog_gene_edge_long.csv``.
        calibration_path: Path to ``calibration.json`` from
            :func:`orthosynassign.stats.calibrate.run_calibration`.
        output_path: Path for the scored output CSV.
        threshold_type: Which threshold to use — ``"f1"`` (F1-maximising) or
            ``"recall"`` (high-recall threshold; recall ≥ target set at
            calibration time).

    Raises:
        ValueError: If *threshold_type* is not ``"f1"`` or ``"recall"``.
        FileNotFoundError: If either input file does not exist.
    """
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("pandas is required. Install with: pip install 'orthosynassign[stats]'") from exc

    if threshold_type not in _THRESHOLD_TYPES:
        raise ValueError(f"threshold_type must be one of {_THRESHOLD_TYPES}; got {threshold_type!r}")

    table_path = Path(table_path)
    calibration_path = Path(calibration_path)
    output_path = Path(output_path)

    if not table_path.exists():
        raise FileNotFoundError(f"Training table not found: {table_path}")
    if not calibration_path.exists():
        raise FileNotFoundError(f"Calibration file not found: {calibration_path}")

    logger.info("Loading calibration model from %s", calibration_path)
    with open(calibration_path, encoding="utf-8") as fh:
        calib = json.load(fh)

    feature_names: list[str] = calib["feature_names"]
    scaler_mean = np.array(calib["scaler_mean"])
    scaler_scale = np.array(calib["scaler_scale"])
    coef = np.array(calib["coefficients"])
    intercept = float(calib["intercept"])
    threshold_key = "threshold_f1" if threshold_type == "f1" else "threshold_recall"
    threshold = float(calib[threshold_key])

    logger.info(
        "Using %s threshold=%.4f  (AUC-ROC=%.4f)",
        threshold_key,
        threshold,
        calib.get("auc_roc", float("nan")),
    )

    logger.info("Loading training table from %s", table_path)
    df = pd.read_csv(table_path)

    edge_df = df[df["edge_type"] != "internal"].copy()
    if edge_df.empty:
        logger.warning("No edge genes found in the training table.  Nothing to score.")
        df.to_csv(output_path, index=False)
        return

    logger.info("Scoring %d edge genes…", len(edge_df))

    # Build feature matrix matching training features
    X_raw = _build_feature_matrix(edge_df, feature_names)

    # Standardise using training scaler parameters
    X_scaled = (X_raw - scaler_mean) / (scaler_scale + _EPSILON)

    # Logistic sigmoid
    log_odds = X_scaled @ coef + intercept
    split_probability = 1.0 / (1.0 + np.exp(-log_odds))

    edge_df = edge_df.copy()
    edge_df["split_probability"] = split_probability
    edge_df["split_confidence"] = 1.0 - split_probability
    edge_df["rescue_flag"] = (split_probability < threshold).astype(int)

    # Merge back with the rest of the table (internal genes are not scored)
    internal_df = df[df["edge_type"] == "internal"].copy()
    internal_df["split_probability"] = np.nan
    internal_df["split_confidence"] = np.nan
    internal_df["rescue_flag"] = -1  # sentinel: not applicable

    scored_df = pd.concat([internal_df, edge_df], ignore_index=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    scored_df.to_csv(output_path, index=False)
    logger.info("Wrote scored table to %s", output_path)

    n_rescue = int(edge_df["rescue_flag"].sum())
    logger.info(
        "%d / %d edge genes flagged for rescue (split_probability < %.4f)",
        n_rescue,
        len(edge_df),
        threshold,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_feature_matrix(df, feature_names: list[str]) -> np.ndarray:
    """Construct the numpy feature matrix from a DataFrame."""
    import pandas as pd

    columns: list[np.ndarray] = []
    for feat in feature_names:
        if feat == "sim_flank_score":
            # At inference time we use the actual (non-masked) flank_score
            columns.append(df["flank_score"].values.astype(float))
        elif feat == "log_flank_completeness":
            fc = df["flank_completeness"].clip(lower=_EPSILON).values.astype(float)
            columns.append(np.log(fc + _EPSILON))
        elif feat == "og_size":
            columns.append(df["og_size"].values.astype(float))
        elif feat == "genome_completeness":
            columns.append(df["genome_completeness"].fillna(1.0).values.astype(float))
        else:
            raise ValueError(f"Unrecognised feature: {feat!r}")
    return np.column_stack(columns)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--table", required=True, type=Path, help="Path to sog_gene_edge_long.csv")
    parser.add_argument("--calibration", required=True, type=Path, help="Path to calibration.json")
    parser.add_argument("--output", type=Path, default=Path("sog_gene_edge_scored.csv"), help="Output CSV path")
    parser.add_argument(
        "--threshold_type",
        choices=_THRESHOLD_TYPES,
        default="f1",
        help="Which calibration threshold to use (default: f1)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    import logging as _logging
    args = _parse_args()
    _logging.basicConfig(level=_logging.DEBUG if args.verbose else _logging.INFO, format="%(levelname)s: %(message)s")
    apply_model(args.table, args.calibration, args.output, args.threshold_type)
