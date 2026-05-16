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
from collections import Counter
from pathlib import Path
from typing import cast

from . import AUTHOR, VERSION
from ._utils import CustomHelpFormatter, RefineArgs, setup_logging, validate_annotations, validate_orthogroup
from .lib import FlankEngine, read_hog_table, read_og_table
from .lib.engine import vectorize_genomes

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

# Edge-type integer → string mapping (matches flank.rs encoding).
_EDGE_TYPE_MAP = {0: "internal", 1: "left_edge", 2: "right_edge", 3: "both_edge"}


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

        # ------------------------------------------------------------------
        # Vectorise genomic data for the Rust FlankEngine.
        # ------------------------------------------------------------------
        logger.info(
            "Building flank windows and scores with Rust FlankEngine (window_n=%d)…",
            args.window,
        )

        # Reuse the OG/seqid vectorisation already computed by the engine module.
        og_idxs_all, seqid_idxs_all, is_circular_all = vectorize_genomes(genomes, orthogroups)

        # Map HOG ID strings to compact integers so Rust can use integer arrays.
        all_hog_ids: list[str] = sorted(set(hog_map.values()))
        hog_id_to_int: dict[str, int] = {hog: i for i, hog in enumerate(all_hog_ids)}

        hog_idxs_all: list[list[int]] = []
        strand_all: list[list[int]] = []
        for genome in genomes:
            hog_idxs_all.append(
                [hog_id_to_int.get(hog_map.get(g.id, ""), -1) for g in genome._genes]
            )
            strand_all.append(
                [1 if g.strand == "+" else (-1 if g.strand == "-" else 0) for g in genome._genes]
            )

        # One parallel Rust call replaces the O(n²) Python loop over build_flank_window.
        flank_engine = FlankEngine(
            seqid_idxs_all,
            hog_idxs_all,
            og_idxs_all,
            strand_all,
            is_circular_all,
            len(orthogroups),
        )
        raw_results = flank_engine.compute_all(window_n=args.window, strand_aware=True)
        logger.info("Computed flank data for %d genes", len(raw_results))

        # ------------------------------------------------------------------
        # Build training table rows from FlankEngine output.
        # ------------------------------------------------------------------
        # Pre-compute global SOG counts once (O(n)) so each row's _is_split
        # check is O(1) instead of the previous O(n) per gene (overall O(n²)).
        sog_global_counts: Counter[str] = Counter(sog_assignments.values())
        has_multiple_assignments = len(sog_assignments) > 1

        rows = _build_rows(
            raw_results,
            genomes,
            all_hog_ids,
            og_sizes,
            genome_completeness,
            sog_assignments,
            sog_global_counts,
            has_multiple_assignments,
        )

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
# Internal helpers
# ---------------------------------------------------------------------------


def _build_rows(
    raw_results: list[tuple],
    genomes,
    all_hog_ids: list[str],
    og_sizes: dict[str, int],
    genome_completeness: dict[str, float],
    sog_assignments: dict[str, str],
    sog_global_counts: Counter,
    has_multiple_assignments: bool,
) -> list[dict]:
    """Convert FlankEngine.compute_all output into training-table row dicts."""
    rows: list[dict] = []
    for genome_idx, gene_idx, flank_sc, _completeness, edge_type_int, left_hog_idxs, right_hog_idxs in raw_results:
        gene = genomes[genome_idx][gene_idx]
        genome_name = genomes[genome_idx].name
        og_id = gene.og.id if gene.og else ""

        # Map integer HOG indices back to HOG ID strings for diagnostic output.
        left_hog_strs = ";".join(sorted(all_hog_ids[h] for h in left_hog_idxs))
        right_hog_strs = ";".join(sorted(all_hog_ids[h] for h in right_hog_idxs))

        sog_id = sog_assignments.get(gene.id, "")
        is_split = int(_is_split_fast(gene.id, sog_assignments, sog_global_counts, has_multiple_assignments))

        rows.append(
            {
                "gene_id": gene.id,
                "genome": genome_name,
                "og_id": og_id,
                "og_size": og_sizes.get(og_id, 0),
                "genome_completeness": genome_completeness.get(genome_name, 1.0),
                "flank_score": flank_sc,
                "flank_left_hogs": left_hog_strs,
                "flank_right_hogs": right_hog_strs,
                "edge_type": _EDGE_TYPE_MAP.get(edge_type_int, "internal"),
                "sog_id": sog_id,
                "is_split": is_split,
            }
        )
    return rows


def _is_split_fast(
    gene_id: str,
    sog_assignments: dict[str, str],
    sog_global_counts: Counter,
    has_multiple_assignments: bool,
) -> bool:
    """O(1) split detection using a pre-computed global SOG Counter.

    A gene is flagged as split when no other gene shares its SOG assignment
    (i.e. the SOG count is exactly 1 — only this gene carries it).
    """
    if not has_multiple_assignments or gene_id not in sog_assignments:
        return False
    gene_sog = sog_assignments[gene_id]
    return sog_global_counts.get(gene_sog, 0) == 1


def _read_sog_assignments(sog_file: Path) -> dict[str, str]:
    """Read Refined_SOGs.tsv and return a gene_id → sog_id mapping."""
    assignments: dict[str, str] = {}
    with open(sog_file, encoding="utf-8") as fh:
        fh.readline()  # skip header
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
