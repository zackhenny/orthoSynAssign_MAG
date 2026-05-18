#!/usr/bin/env python3
"""
The orthosynassign CLI entry point.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import textwrap
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import TYPE_CHECKING, Iterator, cast

import numpy as np

from . import AUTHOR, VERSION
from . import __doc__ as _module_doc
from ._utils import CustomHelpFormatter, RefineArgs, setup_logging, validate_annotations, validate_orthogroup
from .lib import get_synteny_engine, read_og_table, write_og_table

if TYPE_CHECKING:
    from .lib import Gene, Genome, Orthogroup

_EPSILON = 1e-9

# ---------------------------------------------------------------------------
# Calibration model
# ---------------------------------------------------------------------------

@dataclass
class CalibrationModel:
    """Logistic regression model loaded from a ``calibration.json`` file.

    Feature order: [sim_flank_score, log_flank_completeness, og_size, genome_completeness]
    All of these must be available when :meth:`score` is called.
    """

    feature_names: list[str]
    scaler_mean: np.ndarray
    scaler_scale: np.ndarray
    coef: np.ndarray
    intercept: float
    threshold: float

    # Internal cache: (genome_idx, gene_idx) → (flank_score, completeness, edge_type_int)
    _flank_cache: dict[tuple[int, int], tuple[float, float, int]] = field(default_factory=dict, repr=False)

    @classmethod
    def from_json(cls, path: Path, threshold_type: str = "f1") -> "CalibrationModel":
        """Load model from *calibration.json* produced by
        :func:`orthosynassign.stats.calibrate.run_calibration`.

        Args:
            path: Path to ``calibration.json``.
            threshold_type: ``"f1"`` (default) or ``"recall"``.

        Returns:
            Loaded :class:`CalibrationModel`.
        """
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        threshold_key = "threshold_f1" if threshold_type == "f1" else "threshold_recall"
        return cls(
            feature_names=data["feature_names"],
            scaler_mean=np.array(data["scaler_mean"]),
            scaler_scale=np.array(data["scaler_scale"]),
            coef=np.array(data["coefficients"]),
            intercept=float(data["intercept"]),
            threshold=float(data[threshold_key]),
        )

    @classmethod
    def from_dict(cls, data: dict, threshold_type: str = "f1") -> "CalibrationModel":
        """Build model from a result dict returned by
        :func:`orthosynassign.stats.calibrate.run_calibration`.

        Args:
            data: Calibration result dictionary.
            threshold_type: ``"f1"`` or ``"recall"``.

        Returns:
            Loaded :class:`CalibrationModel`.
        """
        threshold_key = "threshold_f1" if threshold_type == "f1" else "threshold_recall"
        return cls(
            feature_names=data["feature_names"],
            scaler_mean=np.array(data["scaler_mean"]),
            scaler_scale=np.array(data["scaler_scale"]),
            coef=np.array(data["coefficients"]),
            intercept=float(data["intercept"]),
            threshold=float(data[threshold_key]),
        )

    def split_probability(self, flank_score: float, flank_completeness: float, og_size: int, genome_completeness: float) -> float:
        """Predict the probability that a gene is a split artifact.

        Args:
            flank_score: HOG-based Jaccard flank score.
            flank_completeness: Fraction of window slots that are on-contig.
            og_size: Number of distinct genomes in the OG.
            genome_completeness: Assembly completeness estimate (0–1).

        Returns:
            Probability in [0, 1].
        """
        features = np.array([
            flank_score,
            np.log(max(flank_completeness, _EPSILON) + _EPSILON),
            float(og_size),
            float(genome_completeness),
        ])
        x_scaled = (features - self.scaler_mean) / (self.scaler_scale + _EPSILON)
        log_odds = float(x_scaled @ self.coef) + self.intercept
        return float(1.0 / (1.0 + np.exp(-log_odds)))

    def is_split(self, flank_score: float, flank_completeness: float, og_size: int, genome_completeness: float) -> bool:
        """Return True when the gene is classified as a split artifact.

        A gene is considered a split artifact when its predicted
        ``split_probability`` is below the calibrated threshold (meaning
        the gene is *not* confidently assigned — i.e. the SOG split looks
        artifactual).
        """
        return self.split_probability(flank_score, flank_completeness, og_size, genome_completeness) < self.threshold


_EPILOG = textwrap.dedent(f"""\
Examples:

# Specify bed files separately:
orthosynassign --og_file orthogroup.tsv --bed file1.bed file2.bed file3.bed

# Specify all bed files in a directory and processed in parallel with 6 CPUs:
orthosynassign --og_file orthogroup.tsv --bed *.bed -t 6

# Specify output file name for results:
orthosynassign --og_file orthogroup.tsv --bed *.bed -o Refined_SOGs.tsv

# Specify window size and ratio threshold:
orthosynassign --og_file orthogroup.tsv --bed *.bed -w 10 -r 0.8

# Apply a pre-built calibration model to filter edge-gene assignments:
orthosynassign --og_file orthogroup.tsv --bed *.bed --calibration calibration.json \\
    --hog_file N0.tsv

# Self-calibrate from interior genes and then filter edge-gene assignments:
orthosynassign --og_file orthogroup.tsv --bed *.bed --auto_calibrate \\
    --hog_file N0.tsv

# With verbose output:
orthosynassign --og_file orthogroup.tsv --bed *.bed -v

Written by {AUTHOR}
""")


_AVAIL_CPUS = int(os.environ.get("SLURM_CPUS_ON_NODE", os.cpu_count()))


def run_cli() -> None:
    """Runs the orthoSynAssign CLI entry point."""
    parsed: RefineArgs = _parse_arguments(sys.argv[1:])
    sys.exit(main(parsed))


def main(args: RefineArgs) -> int:
    """Main entry point for orthoSynAssign.

    Args:
        args (RefineArgs): Parsed command line arguments.  May also carry the
            optional calibration attributes ``calibration``, ``auto_calibrate``,
            ``hog_file``, and ``genome_completeness`` added by the extended CLI.

    Returns:
        int: Exit code.
    """
    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    logger.info("Starting orthoSynAssign")
    logger.debug("Command: %s", " ".join(sys.argv))

    output_path = Path(args.output)
    tmp_output = output_path.with_suffix(output_path.suffix + ".tmp")

    # Optional calibration attributes (absent in basic RefineArgs).
    calibration_path: Path | None = getattr(args, "calibration", None)
    auto_calibrate: bool = getattr(args, "auto_calibrate", False)
    hog_file_path: Path | None = getattr(args, "hog_file", None)
    genome_completeness_path: Path | None = getattr(args, "genome_completeness", None)

    try:
        # Validate inputs
        annotations = validate_annotations(args)
        og_file = validate_orthogroup(args.og_file)

        # Create output directory
        output_dir = output_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Creating output directory: %s", output_dir)

        # Read annotations
        genomes = []
        for annotation in annotations:
            genome = annotation.parse()
            genomes.append(genome)

        # Read orthogroup
        logger.info("Reading orthogroup data from: %s", og_file)
        orthogroups = read_og_table(og_file, genomes)

        # ------------------------------------------------------------------
        # Optional calibration model setup
        # ------------------------------------------------------------------
        calibration_model: CalibrationModel | None = None

        if calibration_path is not None or auto_calibrate:
            if hog_file_path is None:
                raise ValueError("--hog_file is required when --calibration or --auto_calibrate is used.")

            flank_cache, og_sizes_map, genome_completeness = _build_flank_cache(
                genomes,
                orthogroups,
                hog_file_path,
                args.window,
                genome_completeness_path,
                logger,
            )

            if calibration_path is not None:
                logger.info("Loading calibration model from %s", calibration_path)
                calibration_model = CalibrationModel.from_json(calibration_path)
            else:
                # Auto-calibrate: build the training table from interior genes and
                # fit the logistic model inline.
                logger.info("Auto-calibrating from interior genes…")
                calibration_model = _run_auto_calibration(
                    flank_cache,
                    genomes,
                    orthogroups,
                    og_sizes_map,
                    genome_completeness,
                    logger,
                )
        else:
            flank_cache = {}
            og_sizes_map = {}
            genome_completeness = {}

        # ------------------------------------------------------------------
        # Synteny refinement
        # ------------------------------------------------------------------
        logger.info("Refining orthogroups by pairwise synteny analysis.")
        results_stream = _generate_sog_results(
            orthogroups,
            genomes,
            args,
            cpus=args.threads,
            calibration_model=calibration_model,
            flank_cache=flank_cache,
            og_sizes_map=og_sizes_map,
            genome_completeness=genome_completeness,
        )

        write_og_table(results_stream, [genome.name for genome in genomes], tmp_output)
        tmp_output.replace(output_path)
        logger.info("Refinement complete. Results saved to %s", args.output)

        logger.info("orthoSynAssign completed successfully")

    except KeyboardInterrupt:
        logger.warning("Terminated by user.")
        return 130

    except FileNotFoundError as e:
        logger.error("An error occurred: %s", e)
        logger.debug("Traceback details:", exc_info=True)
        return 2

    except Exception as e:
        logger.error("An error occurred: %s", e)
        logger.debug("Traceback details:", exc_info=True)
        return 1

    finally:
        if tmp_output.exists():
            tmp_output.unlink()

    return 0


def _parse_arguments(argv=None) -> RefineArgs:
    """Parse command line arguments.

    Args:
        argv (list of str, optional): The list of arguments to parse. Defaults to sys.argv[1:].

    Returns:
        RefineArgs: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(
        description=_module_doc,
        formatter_class=CustomHelpFormatter,
        epilog=_EPILOG,
        add_help=False,
    )
    req_args = parser.add_argument_group("Required arguments")
    # OrthoFinder input
    req_args.add_argument(
        "--og_file",
        type=Path,
        required=True,
        help="Path to OrthoFinder Orthogroups.tsv file",
    )

    # Input format group (mutually exclusive)
    req_args.add_argument(
        "--bed",
        type=Path,
        required=True,
        metavar=("file", "files"),
        nargs="+",
        help="Path of BED formatted genome annotation files",
    )

    opt_args = parser.add_argument_group("Options")

    opt_args.add_argument(
        "-w",
        "--window",
        type=int,
        default=8,
        help="Controls how many total genes are considered when determining synteny for a single gene",
    )

    opt_args.add_argument(
        "-r",
        "--ratio_threshold",
        dest="threshold",
        type=float,
        default=0.5,
        help=textwrap.dedent("""
            Controls how many genes within a window must provide synteny support
            to classify the genes being compared as syntenous
        """),
    )

    opt_args.add_argument(
        "-o",
        "--output",
        type=Path,
        default="Refined_SOGs-%s.tsv" % time.strftime("%Y%m%d-%H%M%S", time.gmtime()),
        help="Output of results (default: Refined_SOGs-[YYYYMMDD-HHMMSS].tsv (UTC timestamp))",
    )
    opt_args.add_argument("-t", "--threads", type=int, default=min(_AVAIL_CPUS, 4), help="Number of cpus to use")
    opt_args.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")
    opt_args.add_argument("-V", "--version", action="version", version=VERSION)
    opt_args.add_argument("-h", "--help", action="help", help="show this help message and exit")

    # ------------------------------------------------------------------
    # Optional calibration arguments (new in orthoSynAssign_MAG)
    # ------------------------------------------------------------------
    calib_args = parser.add_argument_group("Calibration (optional)")
    calib_args.add_argument(
        "--calibration",
        type=Path,
        default=None,
        metavar="calibration.json",
        help=(
            "Path to a calibration.json produced by orthosynassign.stats.calibrate. "
            "When provided, edge genes whose split_probability falls below the "
            "calibrated threshold are removed from their synteny cluster. "
            "Requires --hog_file."
        ),
    )
    calib_args.add_argument(
        "--auto_calibrate",
        action="store_true",
        default=False,
        help=(
            "Fit a logistic regression model on interior genes from this run "
            "and use it to filter edge-gene assignments (requires [stats] extras "
            "and --hog_file).  Ignored when --calibration is also supplied."
        ),
    )
    calib_args.add_argument(
        "--hog_file",
        type=Path,
        default=None,
        metavar="N0.tsv",
        help="Path to OrthoFinder N0.tsv (HOG table). Required when --calibration or --auto_calibrate is used.",
    )
    calib_args.add_argument(
        "--genome_completeness",
        type=Path,
        default=None,
        metavar="completeness.tsv",
        help="Optional two-column TSV (genome_name, completeness) for calibration features.",
    )

    return cast(RefineArgs, parser.parse_args(argv))


def _generate_sog_results(
    orthogroups: list[Orthogroup],
    genomes: list[Genome],
    args: RefineArgs,
    *,
    cpus: int = 1,
    calibration_model: CalibrationModel | None = None,
    flank_cache: dict[tuple[int, int], tuple[float, float, int]] | None = None,
    og_sizes_map: dict[str, int] | None = None,
    genome_completeness: dict[str, float] | None = None,
) -> Iterator[tuple[str, list[Gene]]]:
    """
    Processes orthogroups and yields results one by one.

    When *calibration_model* is provided together with *flank_cache*, each
    cluster returned by the Rust engine is post-filtered: edge genes whose
    ``split_probability`` falls below the calibrated threshold are removed
    before the cluster is yielded.  This integrates the statistical model
    directly into the main refinement pass without requiring a separate run.

    Args:
        orthogroups: The list of orthogroups to process.
        genomes: Genome objects.
        args: Command-line arguments.
        cpus: Number of CPUs to use for parallel processing.
        calibration_model: Optional pre-loaded or self-trained calibration model.
        flank_cache: Per-gene flank data keyed by ``(genome_idx, gene_idx)``.
        og_sizes_map: Mapping from OG ID to genome count.
        genome_completeness: Mapping from genome name to completeness estimate.

    Yields:
        ``(sog_id, genes)`` tuples.
    """
    total_ogs = len(orthogroups)
    indices = list(range(total_ogs))
    engine = get_synteny_engine(genomes, orthogroups)

    _flank_cache = flank_cache or {}
    _og_sizes = og_sizes_map or {}
    _gc = genome_completeness or {}

    def worker_func(og_idx: int) -> list[list[tuple[int, int]]]:
        return engine.refine(og_idx=og_idx, window_size=args.window, ratio_threshold=args.threshold)

    def process_results(results_iterable: Iterator[list[list[tuple[int, int]]]]):
        global_sog_counter = 1
        step_size = min(max(1000, total_ogs // 100 * 10), 10000)

        for idx, list_of_clusters in enumerate(results_iterable, 1):
            if idx % step_size == 0 or idx == total_ogs:
                logging.info("Progress: %d / %d orthogroups processed...", idx, total_ogs)

            cur_orthogroup = orthogroups[idx - 1]
            og_size = _og_sizes.get(cur_orthogroup.id, len(set(g.genome for g in cur_orthogroup if g.genome)))

            isoform_mapper: defaultdict[Gene, list[Gene]] = defaultdict(list)
            for gene in cur_orthogroup:
                if gene.representative != gene:
                    isoform_mapper[gene.representative].append(gene)

            for cluster in list_of_clusters:
                # Apply calibration filter when a model is available.
                if calibration_model is not None and _flank_cache:
                    cluster = _filter_cluster_with_model(
                        cluster,
                        calibration_model,
                        _flank_cache,
                        og_size,
                        genomes,
                        _gc,
                    )
                    if not cluster:
                        continue

                genes: list[Gene] = []
                for genome_idx, gene_id in cluster:
                    gene = genomes[genome_idx][gene_id]
                    if gene in isoform_mapper:
                        genes.extend(isoform_mapper[gene])
                    else:
                        genes.append(gene)

                sog_id = f"SOG{global_sog_counter:06d}.{cur_orthogroup.id}"
                yield (sog_id, genes)
                global_sog_counter += 1

    logging.info(f"Refining with {cpus} cpu{'s' if cpus > 1 else ''}")
    if cpus == 1:
        results = map(worker_func, indices)
        yield from process_results(results)
        return

    # Parallel path
    opt_chunksize = _calculate_optimal_chunksize(total_ogs, cpus)
    with ThreadPool(processes=cpus) as pool:
        results = pool.imap(worker_func, indices, chunksize=opt_chunksize)
        yield from process_results(results)


def _calculate_optimal_chunksize(iterable_size: int, pool_size: int) -> int:
    """
    Calculate the optimal chunk size for dividing work among workers.

    Args:
        iterable_size (int): The total number of items to be processed.
        pool_size (int): The number of worker processes available.

    Returns:
        int: The calculated optimal chunk size. This is intended to balance the overhead of task distribution with the load
            balancing across workers. A standard heuristic used by many libraries suggests dividing the work into 4 chunks per
            worker.
    """
    if iterable_size == 0:
        return 1

    chunksize, extra = divmod(iterable_size, pool_size * 10)
    if extra:
        chunksize += 1
    return max(1, min(chunksize, 100))


# ---------------------------------------------------------------------------
# Calibration helpers
# ---------------------------------------------------------------------------

# Edge-type integer → string mapping (mirrors score.py / flank.rs encoding).
_EDGE_TYPE_MAP = {0: "internal", 1: "left_edge", 2: "right_edge", 3: "both_edge"}


def _build_flank_cache(
    genomes: list[Genome],
    orthogroups: list[Orthogroup],
    hog_file_path: Path,
    window: int,
    genome_completeness_path: Path | None,
    logger: logging.Logger,
) -> tuple[dict[tuple[int, int], tuple[float, float, int]], dict[str, int], dict[str, float]]:
    """Build the flank data cache required for calibration-based filtering.

    Uses :class:`~orthosynassign.lib.rs.FlankEngine` to compute per-gene flank
    scores, flank completeness, and edge-type encoding in a single parallel pass.

    Args:
        genomes: Genome objects (already parsed).
        orthogroups: List of all orthogroups.
        hog_file_path: Path to OrthoFinder N0.tsv.
        window: Half-window size (same as ``--window``).
        genome_completeness_path: Optional path to genome-completeness TSV.
        logger: Logger instance.

    Returns:
        Tuple of:
        * ``flank_cache`` — mapping ``(genome_idx, gene_idx)`` →
          ``(flank_score, flank_completeness, edge_type_int)``.
        * ``og_sizes_map`` — mapping OG ID → number of distinct genomes.
        * ``genome_completeness`` — mapping genome name → completeness.
    """
    from .lib import FlankEngine, read_hog_table
    from .lib.engine import vectorize_genomes
    from .score import _read_genome_completeness

    logger.info("Reading HOG table from %s for calibration…", hog_file_path)
    hog_map = read_hog_table(hog_file_path, genomes)

    genome_completeness: dict[str, float] = {}
    if genome_completeness_path is not None:
        genome_completeness = _read_genome_completeness(genome_completeness_path)

    og_sizes_map: dict[str, int] = {}
    for og in orthogroups:
        og_genomes = {g.genome.name for g in og if g.genome}
        og_sizes_map[og.id] = len(og_genomes)

    og_idxs_all, seqid_idxs_all, is_circular_all = vectorize_genomes(genomes, orthogroups)

    all_hog_ids: list[str] = sorted(set(hog_map.values()))
    hog_id_to_int: dict[str, int] = {hog: i for i, hog in enumerate(all_hog_ids)}

    hog_idxs_all: list[list[int]] = []
    strand_all: list[list[int]] = []
    for genome in genomes:
        hog_idxs_all.append([hog_id_to_int.get(hog_map.get(g.id, ""), -1) for g in genome._genes])
        strand_all.append([1 if g.strand == "+" else (-1 if g.strand == "-" else 0) for g in genome._genes])

    logger.info("Computing flank data for edge-gene calibration filter…")
    flank_engine = FlankEngine(
        seqid_idxs_all,
        hog_idxs_all,
        og_idxs_all,
        strand_all,
        is_circular_all,
        len(orthogroups),
    )
    raw_results = flank_engine.compute_all(window_n=window, strand_aware=True)

    flank_cache: dict[tuple[int, int], tuple[float, float, int]] = {}
    for genome_idx, gene_idx, flank_sc, completeness, edge_type_int, _left, _right in raw_results:
        flank_cache[(genome_idx, gene_idx)] = (flank_sc, completeness, edge_type_int)

    logger.info("Flank cache built: %d genes", len(flank_cache))
    return flank_cache, og_sizes_map, genome_completeness


def _run_auto_calibration(
    flank_cache: dict[tuple[int, int], tuple[float, float, int]],
    genomes: list[Genome],
    orthogroups: list[Orthogroup],
    og_sizes_map: dict[str, int],
    genome_completeness: dict[str, float],
    logger: logging.Logger,
) -> CalibrationModel | None:
    """Self-calibrate a logistic regression model from interior genes.

    Builds an in-memory training table (same schema as ``sog_gene_edge_long.csv``)
    from interior genes and calls
    :func:`orthosynassign.stats.calibrate.run_calibration`.

    Args:
        flank_cache: Per-gene flank data from :func:`_build_flank_cache`.
        genomes: Genome objects.
        orthogroups: List of all orthogroups.
        og_sizes_map: OG ID → genome count.
        genome_completeness: Genome name → completeness.
        logger: Logger instance.

    Returns:
        Fitted :class:`CalibrationModel`, or ``None`` if calibration failed
        (e.g. too few interior genes or no positive examples).
    """
    try:
        import pandas as pd

        from .stats.calibrate import run_calibration
    except ImportError as exc:
        raise ImportError(
            "pandas and scikit-learn are required for --auto_calibrate. "
            "Install with: pip install 'orthosynassign[stats]'"
        ) from exc

    # Build a minimal training table (interior genes only) in memory.
    og_idx_map = {og.id: i for i, og in enumerate(orthogroups)}
    rows: list[dict] = []
    for (genome_idx, gene_idx), (flank_sc, completeness, edge_type_int) in flank_cache.items():
        if edge_type_int != 0:
            continue  # skip edge genes — use only interior for training
        gene = genomes[genome_idx][gene_idx]
        og_id = gene.og.id if gene.og else ""
        genome_name = genomes[genome_idx].name
        rows.append({
            "gene_id": gene.id,
            "genome": genome_name,
            "og_id": og_id,
            "og_size": og_sizes_map.get(og_id, 0),
            "genome_completeness": genome_completeness.get(genome_name, 1.0),
            "flank_score": flank_sc,
            "flank_completeness": max(completeness, _EPSILON),
            "flank_left_hogs": "",
            "flank_right_hogs": "",
            "edge_type": "internal",
            "sog_id": "",
            "is_split": 0,
        })

    if not rows:
        logger.warning("No interior genes found — skipping auto-calibration.")
        return None

    df = pd.DataFrame(rows)
    if df["is_split"].sum() == 0:
        logger.warning(
            "Auto-calibration: no positive (is_split=1) examples in interior genes — "
            "skipping model fit.  The calibration model requires prior SOG assignments."
        )
        return None

    logger.info("Auto-calibrating on %d interior genes…", len(df))
    try:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp_dir:
            result = run_calibration(dataframe=df, output_dir=tmp_dir)
        return CalibrationModel.from_dict(result)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Auto-calibration failed (%s) — proceeding without model.", exc)
        return None


def _filter_cluster_with_model(
    cluster: list[tuple[int, int]],
    model: CalibrationModel,
    flank_cache: dict[tuple[int, int], tuple[float, float, int]],
    og_size: int,
    genomes: list[Genome],
    genome_completeness: dict[str, float],
) -> list[tuple[int, int]]:
    """Remove edge genes that the calibration model classifies as split artifacts.

    Only edge genes (``edge_type_int != 0``) are evaluated; interior genes pass
    through unconditionally.

    Args:
        cluster: List of ``(genome_idx, gene_idx)`` pairs from the Rust engine.
        model: Loaded :class:`CalibrationModel`.
        flank_cache: Flank data keyed by ``(genome_idx, gene_idx)``.
        og_size: Number of distinct genomes in the orthogroup.
        genomes: Genome objects.
        genome_completeness: Genome name → completeness estimate.

    Returns:
        Filtered list (may be shorter than *cluster*).  Empty list means the
        cluster should be discarded entirely.
    """
    filtered: list[tuple[int, int]] = []
    for genome_idx, gene_idx in cluster:
        key = (genome_idx, gene_idx)
        if key not in flank_cache:
            filtered.append((genome_idx, gene_idx))
            continue
        flank_sc, completeness, edge_type_int = flank_cache[key]
        if edge_type_int == 0:
            # Interior gene — always keep.
            filtered.append((genome_idx, gene_idx))
            continue
        genome_name = genomes[genome_idx].name
        gc = genome_completeness.get(genome_name, 1.0)
        if not model.is_split(flank_sc, completeness, og_size, gc):
            filtered.append((genome_idx, gene_idx))
        # else: drop this edge gene (classified as a split artifact)
    return filtered if len(filtered) > 1 else []


if __name__ == "__main__":
    run_cli()
