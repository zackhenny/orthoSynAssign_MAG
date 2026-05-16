"""
Permutation test: confirm that HOG neighbourhood similarity is a non-random
signal for distinguishing split from unsplit genes.

Usage (command line)::

    python -m orthosynassign.stats.permutation \\
        --table sog_gene_edge_long.csv \\
        --n_permutations 1000 \\
        --output_dir results/

Algorithm
---------
1. Load ``sog_gene_edge_long.csv``; keep only ``edge_type == "internal"`` rows
   (ground-truth split status known from orthoSynAssign).
2. For each OG × genome combination, permute the HOG assignments of flanking
   genes (shuffle ``flank_left_hogs`` / ``flank_right_hogs`` across genes in
   the same OG × genome group), then recompute ``flank_score``.
3. Repeat *N* times; collect the permuted-score distribution.
4. Mann–Whitney U test: observed scores of unsplit interior genes vs. pooled
   permuted scores.
5. Output: ``permutation_test.png`` (distribution plot) and
   ``permutation_summary.tsv`` (U-statistic, p-value per OG-size bin).

The module exits with code 1 and prints a warning if the global permutation
test is not significant at α = 0.05.
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger(__name__)


def run_permutation_test(
    table_path: str | Path,
    n_permutations: int = 1000,
    alpha: float = 0.05,
    output_dir: str | Path = ".",
    seed: int | None = 42,
) -> dict:
    """Run the permutation test and return a summary dict.

    Args:
        table_path: Path to ``sog_gene_edge_long.csv``.
        n_permutations: Number of permutation iterations.
        alpha: Significance level for the global test.
        output_dir: Directory for output plot and TSV.
        seed: Random seed for reproducibility.

    Returns:
        Dictionary with keys ``statistic``, ``pvalue``, ``significant``, and
        ``bin_summary`` (list of per-OG-size-bin result dicts).

    Raises:
        ImportError: If ``scipy`` or ``matplotlib`` are not installed.
        FileNotFoundError: If *table_path* does not exist.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import pandas as pd
        from scipy.stats import mannwhitneyu
    except ImportError as exc:
        raise ImportError(
            "scipy and matplotlib are required for the permutation test. "
            "Install them with: pip install 'orthosynassign[stats]'"
        ) from exc

    table_path = Path(table_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not table_path.exists():
        raise FileNotFoundError(f"Training table not found: {table_path}")

    logger.info("Loading training table from %s", table_path)
    df = pd.read_csv(table_path)

    internal = df[df["edge_type"] == "internal"].copy()
    if internal.empty:
        raise ValueError("No internal-edge genes found in the training table.")

    logger.info("Running permutation test on %d internal genes", len(internal))

    if seed is not None:
        random.seed(seed)

    observed_scores = internal["flank_score"].values
    permuted_scores_all: list[float] = []

    # Group by og_id × genome for permutation
    groups = internal.groupby(["og_id", "genome"])

    for _ in range(n_permutations):
        perm_df = internal.copy()
        for (og_id, genome), group_idx in groups.groups.items():
            group = internal.loc[group_idx]
            if len(group) < 2:
                continue
            left_hogs = group["flank_left_hogs"].tolist()
            right_hogs = group["flank_right_hogs"].tolist()
            shuffled_left = left_hogs[:]
            shuffled_right = right_hogs[:]
            random.shuffle(shuffled_left)
            random.shuffle(shuffled_right)
            perm_df.loc[group_idx, "flank_left_hogs"] = shuffled_left
            perm_df.loc[group_idx, "flank_right_hogs"] = shuffled_right

        # Recompute flank_score from the permuted HOG sets
        perm_df["perm_score"] = perm_df.apply(
            lambda row: _recompute_flank_score(row, perm_df), axis=1
        )
        permuted_scores_all.extend(perm_df["perm_score"].tolist())

    # Global Mann-Whitney U test
    stat, pvalue = mannwhitneyu(
        observed_scores,
        permuted_scores_all,
        alternative="greater",
    )
    significant = pvalue < alpha
    if not significant:
        logger.warning(
            "Permutation test NOT significant (U=%.2f, p=%.4f ≥ %.2f). "
            "HOG-neighbourhood signal may be too noisy to use as a filter.",
            stat,
            pvalue,
            alpha,
        )
    else:
        logger.info("Permutation test significant: U=%.2f, p=%.4e", stat, pvalue)

    # Per-OG-size bin summary
    internal["og_size_bin"] = pd.cut(
        internal["og_size"],
        bins=[0, 5, 10, 25, 50, float("inf")],
        labels=["1-5", "6-10", "11-25", "26-50", "50+"],
    )
    bin_summary: list[dict] = []
    for bin_label, grp in internal.groupby("og_size_bin", observed=True):
        obs = grp["flank_score"].values
        if len(obs) < 2:
            continue
        b_stat, b_p = mannwhitneyu(obs, permuted_scores_all, alternative="greater")
        bin_summary.append({"og_size_bin": str(bin_label), "U_statistic": b_stat, "pvalue": b_p, "n_genes": len(obs)})

    # Write summary TSV
    summary_path = output_dir / "permutation_summary.tsv"
    import csv
    with open(summary_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["og_size_bin", "U_statistic", "pvalue", "n_genes"], delimiter="\t")
        writer.writeheader()
        writer.writerows(bin_summary)
    logger.info("Wrote permutation summary to %s", summary_path)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(permuted_scores_all, bins=50, alpha=0.6, label="Permuted scores", color="steelblue", density=True)
    ax.hist(observed_scores, bins=50, alpha=0.7, label="Observed scores (unsplit internal)", color="coral", density=True)
    ax.set_xlabel("Flank score")
    ax.set_ylabel("Density")
    ax.set_title(f"Permutation test  |  U={stat:.1f}  p={pvalue:.2e}  (α={alpha})")
    ax.legend()
    plot_path = output_dir / "permutation_test.png"
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    logger.info("Saved permutation test plot to %s", plot_path)

    return {
        "statistic": float(stat),
        "pvalue": float(pvalue),
        "significant": significant,
        "bin_summary": bin_summary,
    }


def _recompute_flank_score(row, df) -> float:
    """Recompute Jaccard-based flank score from permuted string HOG sets."""
    left = set(row["flank_left_hogs"].split(";")) if isinstance(row["flank_left_hogs"], str) and row["flank_left_hogs"] else set()
    right = set(row["flank_right_hogs"].split(";")) if isinstance(row["flank_right_hogs"], str) and row["flank_right_hogs"] else set()
    focal = left | right
    if not focal:
        return 0.0

    og_rows = df[df["og_id"] == row["og_id"]]
    scores = []
    for _, ref_row in og_rows.iterrows():
        if ref_row["gene_id"] == row["gene_id"]:
            continue
        ref_left = set(ref_row["flank_left_hogs"].split(";")) if isinstance(ref_row["flank_left_hogs"], str) and ref_row["flank_left_hogs"] else set()
        ref_right = set(ref_row["flank_right_hogs"].split(";")) if isinstance(ref_row["flank_right_hogs"], str) and ref_row["flank_right_hogs"] else set()
        ref = ref_left | ref_right
        union = focal | ref
        intersection = focal & ref
        scores.append(len(intersection) / len(union) if union else 0.0)
    return sum(scores) / len(scores) if scores else 0.0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--table", required=True, type=Path, help="Path to sog_gene_edge_long.csv")
    parser.add_argument("--n_permutations", type=int, default=1000, help="Number of permutation iterations (default: 1000)")
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance level (default: 0.05)")
    parser.add_argument("--output_dir", type=Path, default=Path("."), help="Output directory (default: current dir)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s: %(message)s")
    result = run_permutation_test(args.table, args.n_permutations, args.alpha, args.output_dir, args.seed)
    if not result["significant"]:
        sys.exit(1)
