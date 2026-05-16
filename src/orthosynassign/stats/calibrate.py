"""
Logistic regression calibration using interior genes as held-out validation.

Interior genes (``edge_type == "internal"``) have a known ground-truth split
status from orthoSynAssign.  This module:

1. Loads ``sog_gene_edge_long.csv`` and filters to internal genes.
2. Simulates edge behaviour by masking one randomly-chosen flank.
3. Fits a :class:`sklearn.linear_model.LogisticRegressionCV` with
   genome-stratified folds.
4. Selects two operating thresholds:

   * **F1-maximising threshold** — best overall balance.
   * **High-recall threshold** — recall ≥ 0.95 (rescuing splits is worse than
     missing a rescued gene).

5. Saves model coefficients and thresholds to ``calibration.json``.
6. Saves ROC and PR curve plots.

Usage::

    python -m orthosynassign.stats.calibrate \\
        --table sog_gene_edge_long.csv \\
        --output_dir results/
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_EPSILON = 1e-9


def run_calibration(
    table_path: str | Path,
    output_dir: str | Path = ".",
    seed: int = 42,
    cv_folds: int = 5,
    high_recall_threshold: float = 0.95,
) -> dict:
    """Fit a calibrated logistic regression on interior genes.

    Args:
        table_path: Path to ``sog_gene_edge_long.csv``.
        output_dir: Directory for ``calibration.json`` and plots.
        seed: Random seed.
        cv_folds: Number of cross-validation folds (genome-stratified).
        high_recall_threshold: Minimum recall for the second threshold.

    Returns:
        Dict with keys ``coefficients``, ``intercept``, ``feature_names``,
        ``threshold_f1``, ``threshold_recall``, ``auc_roc``, ``auc_pr``.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import pandas as pd
        from sklearn.linear_model import LogisticRegressionCV
        from sklearn.metrics import (
            average_precision_score,
            f1_score,
            precision_recall_curve,
            roc_auc_score,
            roc_curve,
        )
        from sklearn.model_selection import GroupKFold
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise ImportError(
            "scikit-learn, numpy, and matplotlib are required. "
            "Install with: pip install 'orthosynassign[stats]'"
        ) from exc

    table_path = Path(table_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading training table from %s", table_path)
    df = pd.read_csv(table_path)

    internal = df[df["edge_type"] == "internal"].copy()
    if internal.empty:
        raise ValueError("No internal-edge genes found.")

    logger.info("Simulating edge behaviour for %d internal genes…", len(internal))
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    # Simulate single-flank: randomly zero out one flank per gene
    def _mask_one_flank(row):
        if rng.random() < 0.5:
            # mask left flank → pretend only right flank available
            left = frozenset()
            right = frozenset(row["flank_right_hogs"].split(";")) if isinstance(row["flank_right_hogs"], str) and row["flank_right_hogs"] else frozenset()
        else:
            left = frozenset(row["flank_left_hogs"].split(";")) if isinstance(row["flank_left_hogs"], str) and row["flank_left_hogs"] else frozenset()
            right = frozenset()
        return left | right

    internal["masked_hogs"] = internal.apply(_mask_one_flank, axis=1)

    # Recompute flank_score using masked flank
    og_ref: dict[str, list[frozenset]] = {}
    for og_id, grp in internal.groupby("og_id"):
        og_ref[og_id] = [
            (frozenset(r["flank_left_hogs"].split(";")) if isinstance(r["flank_left_hogs"], str) and r["flank_left_hogs"] else frozenset())
            | (frozenset(r["flank_right_hogs"].split(";")) if isinstance(r["flank_right_hogs"], str) and r["flank_right_hogs"] else frozenset())
            for _, r in grp.iterrows()
        ]

    def _masked_score(row):
        focal = row["masked_hogs"]
        refs = [r for r in og_ref.get(row["og_id"], []) if r != focal]
        if not refs:
            return 0.0
        scores = []
        for ref in refs:
            union = focal | ref
            scores.append(len(focal & ref) / len(union) if union else 0.0)
        return sum(scores) / len(scores)

    internal["sim_flank_score"] = internal.apply(_masked_score, axis=1)

    # Feature matrix
    X = np.column_stack([
        internal["sim_flank_score"].values,
        np.log(internal["flank_completeness"].clip(lower=_EPSILON).values + _EPSILON),
        internal["og_size"].values,
        internal["genome_completeness"].fillna(1.0).values,
    ])
    y = internal["is_split"].values
    groups = internal["genome"].values

    if y.sum() == 0:
        raise ValueError("No positive (is_split=1) examples found — cannot train classifier.")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    feature_names = ["sim_flank_score", "log_flank_completeness", "og_size", "genome_completeness"]

    # Genome-stratified CV
    n_unique_genomes = len(set(groups))
    actual_folds = min(cv_folds, n_unique_genomes)
    cv_splitter = GroupKFold(n_splits=actual_folds)

    logger.info("Fitting LogisticRegressionCV with %d genome-stratified folds…", actual_folds)
    clf = LogisticRegressionCV(
        cv=list(cv_splitter.split(X_scaled, y, groups=groups)),
        solver="lbfgs",
        max_iter=1000,
        random_state=seed,
        scoring="roc_auc",
        n_jobs=-1,
    )
    clf.fit(X_scaled, y)

    y_prob = clf.predict_proba(X_scaled)[:, 1]
    auc_roc = float(roc_auc_score(y, y_prob))
    auc_pr = float(average_precision_score(y, y_prob))
    logger.info("Training AUC-ROC=%.4f  AUC-PR=%.4f", auc_roc, auc_pr)

    # Select thresholds
    fpr, tpr, roc_thresholds = roc_curve(y, y_prob)
    precision, recall, pr_thresholds = precision_recall_curve(y, y_prob)

    # F1-maximising threshold
    f1_scores = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + _EPSILON)
    best_f1_idx = int(np.argmax(f1_scores))
    threshold_f1 = float(pr_thresholds[best_f1_idx])

    # High-recall threshold (recall ≥ high_recall_threshold)
    valid_recall_mask = recall[:-1] >= high_recall_threshold
    if valid_recall_mask.any():
        # Among thresholds with sufficient recall, pick the one with highest precision
        best_recall_idx = int(np.argmax(precision[:-1][valid_recall_mask]))
        threshold_recall = float(pr_thresholds[valid_recall_mask][best_recall_idx])
    else:
        logger.warning("Cannot achieve recall ≥ %.2f; using lowest threshold.", high_recall_threshold)
        threshold_recall = float(pr_thresholds[-1])

    logger.info("Threshold (F1): %.4f | Threshold (recall≥%.2f): %.4f", threshold_f1, high_recall_threshold, threshold_recall)

    # Save plots
    _plot_roc(fpr, tpr, auc_roc, output_dir / "calibration_roc.png")
    _plot_pr(recall, precision, auc_pr, output_dir / "calibration_pr.png")

    # Save model
    result = {
        "feature_names": feature_names,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "coefficients": clf.coef_[0].tolist(),
        "intercept": float(clf.intercept_[0]),
        "threshold_f1": threshold_f1,
        "threshold_recall": threshold_recall,
        "high_recall_target": high_recall_threshold,
        "auc_roc": auc_roc,
        "auc_pr": auc_pr,
    }
    calib_path = output_dir / "calibration.json"
    with open(calib_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    logger.info("Saved calibration model to %s", calib_path)

    return result


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------


def _plot_roc(fpr, tpr, auc, path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, lw=2, label=f"ROC (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Calibration ROC curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_pr(recall, precision, auc, path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, lw=2, color="darkorange", label=f"PR (AUC={auc:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Calibration Precision-Recall curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--table", required=True, type=Path, help="Path to sog_gene_edge_long.csv")
    parser.add_argument("--output_dir", type=Path, default=Path("."), help="Output directory (default: .)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cv_folds", type=int, default=5)
    parser.add_argument("--high_recall", type=float, default=0.95, dest="high_recall_threshold")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s: %(message)s")
    run_calibration(args.table, args.output_dir, args.seed, args.cv_folds, args.high_recall_threshold)
