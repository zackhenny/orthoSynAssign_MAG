"""
Genome-stratified k-fold cross-validation with ROC/PR analysis.

Uses :class:`sklearn.model_selection.GroupKFold` with genomes as groups to
prevent data leakage (genes from the same genome share systematic fragmentation
patterns).

Outputs
-------
* ``cv_roc.png``     — mean ± 1 SD ROC curve across folds
* ``cv_pr.png``      — mean ± 1 SD PR curve across folds
* ``cv_summary.tsv`` — per-fold and mean AUC, threshold at FPR ≤ 0.05,
  confusion matrix at that threshold

Usage::

    python -m orthosynassign.stats.cv \\
        --table sog_gene_edge_long.csv \\
        --output_dir results/ \\
        --cv_folds 5 \\
        --max_fpr 0.05
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_EPSILON = 1e-9


def run_cross_validation(
    table_path: str | Path,
    output_dir: str | Path = ".",
    cv_folds: int = 5,
    max_fpr: float = 0.05,
    seed: int = 42,
) -> dict:
    """Run genome-stratified cross-validation.

    Args:
        table_path: Path to ``sog_gene_edge_long.csv``.
        output_dir: Directory for plots and summary TSV.
        cv_folds: Number of cross-validation folds.
        max_fpr: Maximum allowable FPR when selecting the operating threshold.
        seed: Random seed for logistic regression.

    Returns:
        Dict with keys ``mean_auc_roc``, ``std_auc_roc``, ``mean_auc_pr``,
        ``std_auc_pr``, ``threshold_at_max_fpr``, and ``fold_results``.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import pandas as pd
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import (
            average_precision_score,
            confusion_matrix,
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

    X = np.column_stack([
        internal["flank_score"].values,
        np.log(internal["flank_completeness"].clip(lower=_EPSILON).values + _EPSILON),
        internal["og_size"].values,
        internal["genome_completeness"].fillna(1.0).values,
    ])
    y = internal["is_split"].values
    groups = internal["genome"].values

    n_unique_genomes = len(set(groups))
    actual_folds = min(cv_folds, n_unique_genomes)
    logger.info(
        "Running %d-fold genome-stratified CV on %d genes (%d genomes)",
        actual_folds, len(y), n_unique_genomes,
    )

    cv = GroupKFold(n_splits=actual_folds)

    # For interpolated mean ROC/PR curves
    mean_fpr = np.linspace(0, 1, 200)
    tprs: list[np.ndarray] = []
    mean_recall_grid = np.linspace(0, 1, 200)
    precisions: list[np.ndarray] = []

    fold_results: list[dict] = []

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y, groups=groups), 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        clf = LogisticRegression(solver="lbfgs", max_iter=1000, random_state=seed)
        clf.fit(X_train_s, y_train)
        y_prob = clf.predict_proba(X_test_s)[:, 1]

        if y_test.sum() == 0 or y_test.sum() == len(y_test):
            logger.warning("Fold %d: only one class in test set; skipping.", fold_idx)
            continue

        auc_roc = float(roc_auc_score(y_test, y_prob))
        auc_pr = float(average_precision_score(y_test, y_prob))

        fpr, tpr, thresholds = roc_curve(y_test, y_prob)
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)

        prec, rec, pr_thresh = precision_recall_curve(y_test, y_prob)
        interp_prec = np.interp(mean_recall_grid, rec[::-1], prec[::-1])
        precisions.append(interp_prec)

        # Threshold at max_fpr
        valid = fpr <= max_fpr
        if valid.any():
            best_idx = int(np.argmax(tpr[valid]))
            thresh_at_fpr = float(thresholds[valid][best_idx])
        else:
            thresh_at_fpr = float(thresholds[0])

        y_pred = (y_prob >= thresh_at_fpr).astype(int)
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = (cm.ravel() if cm.size == 4 else (0, 0, 0, 0))

        fold_results.append({
            "fold": fold_idx,
            "auc_roc": auc_roc,
            "auc_pr": auc_pr,
            "threshold": thresh_at_fpr,
            "TP": int(tp),
            "FP": int(fp),
            "TN": int(tn),
            "FN": int(fn),
        })
        logger.info("Fold %d: AUC-ROC=%.4f  AUC-PR=%.4f  threshold=%.4f", fold_idx, auc_roc, auc_pr, thresh_at_fpr)

    if not fold_results:
        raise RuntimeError("All folds were skipped due to single-class test sets.")

    aucs_roc = [r["auc_roc"] for r in fold_results]
    aucs_pr = [r["auc_pr"] for r in fold_results]
    mean_auc_roc = float(np.mean(aucs_roc))
    std_auc_roc = float(np.std(aucs_roc))
    mean_auc_pr = float(np.mean(aucs_pr))
    std_auc_pr = float(np.std(aucs_pr))
    threshold_at_fpr = float(np.mean([r["threshold"] for r in fold_results]))

    logger.info(
        "Mean AUC-ROC=%.4f ± %.4f  Mean AUC-PR=%.4f ± %.4f  Mean threshold=%.4f",
        mean_auc_roc, std_auc_roc, mean_auc_pr, std_auc_pr, threshold_at_fpr,
    )

    # Plot ROC
    _plot_mean_roc(mean_fpr, tprs, mean_auc_roc, std_auc_roc, output_dir / "cv_roc.png")
    # Plot PR
    _plot_mean_pr(mean_recall_grid, precisions, mean_auc_pr, std_auc_pr, output_dir / "cv_pr.png")

    # Write summary TSV
    summary_path = output_dir / "cv_summary.tsv"
    fieldnames = ["fold", "auc_roc", "auc_pr", "threshold", "TP", "FP", "TN", "FN"]
    with open(summary_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(fold_results)
        writer.writerow({
            "fold": "MEAN",
            "auc_roc": f"{mean_auc_roc:.4f}±{std_auc_roc:.4f}",
            "auc_pr": f"{mean_auc_pr:.4f}±{std_auc_pr:.4f}",
            "threshold": f"{threshold_at_fpr:.4f}",
            "TP": "",
            "FP": "",
            "TN": "",
            "FN": "",
        })
    logger.info("Wrote CV summary to %s", summary_path)

    return {
        "mean_auc_roc": mean_auc_roc,
        "std_auc_roc": std_auc_roc,
        "mean_auc_pr": mean_auc_pr,
        "std_auc_pr": std_auc_pr,
        "threshold_at_max_fpr": threshold_at_fpr,
        "fold_results": fold_results,
    }


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------


def _plot_mean_roc(mean_fpr, tprs, mean_auc, std_auc, path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    mean_tpr = np.mean(tprs, axis=0)
    std_tpr = np.std(tprs, axis=0)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(mean_fpr, mean_tpr, lw=2, label=f"Mean ROC (AUC={mean_auc:.3f}±{std_auc:.3f})")
    ax.fill_between(mean_fpr, mean_tpr - std_tpr, mean_tpr + std_tpr, alpha=0.2)
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Cross-validation ROC curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_mean_pr(mean_recall, precisions, mean_auc, std_auc, path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    mean_prec = np.mean(precisions, axis=0)
    std_prec = np.std(precisions, axis=0)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(mean_recall, mean_prec, lw=2, color="darkorange", label=f"Mean PR (AUC={mean_auc:.3f}±{std_auc:.3f})")
    ax.fill_between(mean_recall, mean_prec - std_prec, mean_prec + std_prec, alpha=0.2, color="darkorange")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Cross-validation Precision-Recall curve")
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
    parser.add_argument("--output_dir", type=Path, default=Path("."))
    parser.add_argument("--cv_folds", type=int, default=5)
    parser.add_argument("--max_fpr", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s: %(message)s")
    run_cross_validation(args.table, args.output_dir, args.cv_folds, args.max_fpr, args.seed)
