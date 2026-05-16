"""
Python wrapper for the mixed-effects logistic regression R script.

Calls ``Rscript mixed_effects.R`` as a subprocess, then parses the output TSV
files and logs the key results:

* Fixed-effect coefficient for ``flank_score`` with 95 % CI
* ICC: fraction of residual variance explained by genome identity
* Per-genome random intercept table

Usage::

    python -m orthosynassign.stats.mixed_effects \\
        --table sog_gene_edge_long.csv \\
        --output_dir results/

Requirements
------------
* ``Rscript`` must be on the ``PATH``.
* R package ``lme4`` must be installed (``install.packages("lme4")``).
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# Path to the R script bundled with this package
_R_SCRIPT = Path(__file__).with_name("mixed_effects.R")


def run_mixed_effects(
    table_path: str | Path,
    output_dir: str | Path = ".",
    *,
    verbose: bool = False,
) -> dict:
    """Fit the mixed-effects logistic regression via Rscript.

    Args:
        table_path: Path to ``sog_gene_edge_long.csv``.
        output_dir: Directory for output TSV files and ICC summary.
        verbose: Pass ``--verbose`` to the R script.

    Returns:
        Dict with keys ``fixed_effects`` (list of row dicts), ``icc`` (float),
        ``tau0_sq`` (float), and ``random_effects`` (list of row dicts).

    Raises:
        RuntimeError: If ``Rscript`` is not found, the R script fails, or the
            expected output files are missing.
    """
    table_path = Path(table_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _check_rscript()

    cmd = [
        "Rscript",
        str(_R_SCRIPT),
        "--table", str(table_path),
        "--output_dir", str(output_dir),
    ]
    if verbose:
        cmd.append("--verbose")

    logger.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.stdout:
        for line in result.stdout.splitlines():
            logger.info("[R] %s", line)
    if result.stderr:
        for line in result.stderr.splitlines():
            logger.debug("[R stderr] %s", line)

    if result.returncode != 0:
        raise RuntimeError(
            f"Rscript exited with code {result.returncode}.\n"
            f"stderr:\n{result.stderr}"
        )

    # Parse outputs
    fixed_effects = _parse_tsv(output_dir / "mixed_effects_results.tsv")
    random_effects = _parse_tsv(output_dir / "genome_random_effects.tsv")
    icc, tau0_sq = _parse_icc(output_dir / "mixed_effects_icc.txt")

    # Log key results
    for row in fixed_effects:
        if row.get("term") == "flank_score":
            logger.info(
                "flank_score coefficient: %.4f (95%% CI [%.4f, %.4f])",
                float(row["Estimate"]),
                float(row["CI_lower"]),
                float(row["CI_upper"]),
            )
    logger.info("ICC (genome random effect): %.4f", icc)
    logger.info("  → %.1f%% of residual variance explained by genome identity", icc * 100)

    # Flag high-contamination genomes (top 10% most negative random intercepts)
    if random_effects:
        intercepts = sorted(random_effects, key=lambda r: float(r["random_intercept"]))
        cutoff_idx = max(0, len(intercepts) // 10)
        flagged = intercepts[:cutoff_idx]
        if flagged:
            logger.info(
                "Genomes with lowest random intercepts (potential high-contamination / low-completeness):\n  %s",
                "\n  ".join(f"{r['genome']}: {float(r['random_intercept']):.3f}" for r in flagged),
            )

    return {
        "fixed_effects": fixed_effects,
        "icc": icc,
        "tau0_sq": tau0_sq,
        "random_effects": random_effects,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _check_rscript() -> None:
    """Raise RuntimeError if Rscript is not on PATH."""
    if shutil.which("Rscript") is None:
        raise RuntimeError(
            "Rscript not found on PATH.  Install R and the lme4 package:\n"
            "  https://cran.r-project.org/\n"
            "  install.packages('lme4')"
        )


def _parse_tsv(path: Path) -> list[dict]:
    """Parse a tab-separated file and return a list of row dicts."""
    if not path.exists():
        raise FileNotFoundError(f"Expected output file not found: {path}")
    rows: list[dict] = []
    with open(path, encoding="utf-8") as fh:
        header = fh.readline().strip().split("\t")
        for line in fh:
            fields = line.strip().split("\t")
            if fields:
                rows.append(dict(zip(header, fields)))
    return rows


def _parse_icc(path: Path) -> tuple[float, float]:
    """Parse the ICC text file and return (icc, tau0_sq)."""
    if not path.exists():
        raise FileNotFoundError(f"Expected ICC file not found: {path}")
    values: dict[str, float] = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            parts = line.strip().split("\t")
            if len(parts) == 2:
                try:
                    values[parts[0]] = float(parts[1])
                except ValueError:
                    pass
    return values.get("ICC", float("nan")), values.get("tau0_sq", float("nan"))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--table", required=True, type=Path, help="Path to sog_gene_edge_long.csv")
    parser.add_argument("--output_dir", type=Path, default=Path("."))
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s: %(message)s")
    try:
        run_mixed_effects(args.table, args.output_dir, verbose=args.verbose)
    except RuntimeError as exc:
        logger.error("%s", exc)
        sys.exit(1)
