#!/usr/bin/env python3
"""
orthosynassign-score: export the flank-score training table.

Reads BED annotation files, an OrthoFinder orthogroups TSV, an OrthoFinder HOG
table (N0.tsv), and the refined SOG output produced by ``orthosynassign``, then
computes per-gene flank scores and writes ``sog_gene_edge_long.csv``.

The resulting CSV is the primary input for all statistical modelling steps
(permutation test, logistic regression calibration, cross-validation, and the
mixed-effects model).
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import textwrap
import time
from pathlib import Path
from typing import cast

from . import AUTHOR, VERSION
from ._utils import CustomHelpFormatter, RefineArgs, setup_logging, validate_annotations, validate_orthogroup
from .lib import BedParser, FlankRecord, build_flank_window, build_training_table, read_hog_table, read_og_table

_EPILOG = textwrap.dedent(f"""\
Examples:

# Export flank table using default window size:
orthosynassign-score --og_file orthogroups.tsv --hog_file N0.tsv \\
    --sog_file Refined_SOGs.tsv --bed *.bed -o sog_gene_edge_long.csv

# Specify window size and genome completeness:
orthosynassign-score --og_file orthogroups.tsv --hog_file N0.tsv \\
    --sog_file Refined_SOGs.tsv --bed *.bed -w 4 \\
    --genome_completeness completeness.tsv

Written by {AUTHOR}
""")

_AVAIL_CPUS = 1


class ScoreArgs(RefineArgs):
    """Arguments for orthosynassign-score."""

    hog_file: Path
    sog_file: Path
    genome_completeness: Path | None


def run_cli() -> None:
    """Runs the orthosynassign-score CLI entry point."""
    parsed: ScoreArgs = _parse_arguments(sys.argv[1:])
    sys.exit(main(parsed))


def main(args: ScoreArgs) -> int:
    """Main entry point for orthosynassign-score.

    Args:
        args: Parsed command line arguments.

    Returns:
        Exit code.
    """
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    logger.info("Starting orthosynassign-score")

    try:
        annotations = validate_annotations(args)
        og_file = validate_orthogroup(args.og_file)

        # Parse BED files
        genomes = [ann.parse() for ann in annotations]

        # Read orthogroups
        logger.info("Reading orthogroup data from: %s", og_file)
        orthogroups = read_og_table(og_file, genomes)

        # Read HOG table
        logger.info("Reading HOG table from: %s", args.hog_file)
        hog_map = read_hog_table(args.hog_file, genomes)

        # Read SOG assignments
        logger.info("Reading SOG assignments from: %s", args.sog_file)
        sog_assignments = _read_sog_assignments(args.sog_file)

        # Read optional genome completeness
        genome_completeness: dict[str, float] = {}
        if args.genome_completeness:
            genome_completeness = _read_genome_completeness(args.genome_completeness)

        # Build OG sizes (number of distinct genomes per OG)
        og_sizes: dict[str, int] = {}
        for og in orthogroups:
            og_genomes = {g.genome.name for g in og if g.genome}
            og_sizes[og.id] = len(og_genomes)

        # Build flank records for all genes across all genomes
        logger.info("Building flank windows (window_n=%d)…", args.window)
        flank_records: list[FlankRecord] = []
        for genome in genomes:
            for gene_idx in range(len(genome._genes)):
                gene = genome._genes[gene_idx]
                if gene.og is None:
                    continue
                rec = build_flank_window(genome, gene_idx, args.window, hog_map, strand_aware=True)
                flank_records.append(rec)

        logger.info("Built %d flank records", len(flank_records))

        # Assemble training table
        rows = build_training_table(flank_records, sog_assignments, og_sizes, genome_completeness)

        # Write CSV
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        _write_csv(rows, output_path)
        logger.info("Wrote training table to %s (%d rows)", output_path, len(rows))

    except KeyboardInterrupt:
        logging.getLogger(__name__).warning("Terminated by user.")
        return 130
    except FileNotFoundError as exc:
        logging.getLogger(__name__).error("File not found: %s", exc)
        return 2
    except Exception as exc:
        logging.getLogger(__name__).error("An error occurred: %s", exc)
        logging.getLogger(__name__).debug("Traceback:", exc_info=True)
        return 1

    return 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_sog_assignments(sog_file: Path) -> dict[str, str]:
    """Read Refined_SOGs.tsv and return a gene_id → sog_id mapping."""
    assignments: dict[str, str] = {}
    with open(sog_file, encoding="utf-8") as fh:
        header = fh.readline().strip().split("\t")
        # Columns: SOG_ID, sample1, sample2, ...
        for line in fh:
            fields = line.strip("\n").split("\t")
            if not fields[0]:
                continue
            sog_id = fields[0]
            for cell in fields[1:]:
                for gene_id in cell.replace(",", " ").split():
                    gene_id = gene_id.strip()
                    if gene_id:
                        assignments[gene_id] = sog_id
    return assignments


def _read_genome_completeness(path: Path) -> dict[str, float]:
    """Read a two-column TSV (genome_name, completeness) and return a dict."""
    completeness: dict[str, float] = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                try:
                    completeness[parts[0]] = float(parts[1])
                except ValueError:
                    pass
    return completeness


def _write_csv(rows: list[dict], path: Path) -> None:
    """Write a list of row dicts to a CSV file."""
    if not rows:
        logging.getLogger(__name__).warning("No rows to write.")
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _parse_arguments(argv=None) -> ScoreArgs:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=CustomHelpFormatter,
        epilog=_EPILOG,
        add_help=False,
    )
    req = parser.add_argument_group("Required arguments")
    req.add_argument("--og_file", type=Path, required=True, help="Path to OrthoFinder Orthogroups.tsv file")
    req.add_argument("--hog_file", type=Path, required=True, help="Path to OrthoFinder N0.tsv (HOG table)")
    req.add_argument("--sog_file", type=Path, required=True, help="Path to Refined_SOGs.tsv from orthosynassign")
    req.add_argument("--bed", type=Path, required=True, nargs="+", metavar=("file", "files"), help="BED genome annotation files")

    opt = parser.add_argument_group("Options")
    opt.add_argument("-w", "--window", type=int, default=4, help="Half-window size for flank HOG extraction (default: 4)")
    opt.add_argument(
        "--genome_completeness",
        type=Path,
        default=None,
        help="Optional two-column TSV (genome_name, completeness) e.g. from CheckM2",
    )
    opt.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("sog_gene_edge_long.csv"),
        help="Output CSV file path (default: sog_gene_edge_long.csv)",
    )
    opt.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")
    opt.add_argument("-V", "--version", action="version", version=VERSION)
    opt.add_argument("-h", "--help", action="help", help="show this help message and exit")
    # stub threshold/threads so ScoreArgs satisfies RefineArgs Protocol
    parser.set_defaults(threshold=0.5, threads=1)

    return cast(ScoreArgs, parser.parse_args(argv))


if __name__ == "__main__":
    run_cli()
