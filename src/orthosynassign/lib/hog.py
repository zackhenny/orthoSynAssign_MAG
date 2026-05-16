"""
HOG (Hierarchical Orthogroup) table reader and flank-window builder.

Reads OrthoFinder N0.tsv / Phylogenetic_Hierarchical_Orthogroups TSV files and
provides utilities to extract the set of HOG IDs flanking a given gene in its
genome.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .gene import Genome

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class FlankRecord:
    """Stores the flanking HOG context for a single gene.

    Attributes:
        gene_id: Identifier of the focal gene.
        genome: Name of the genome the gene belongs to.
        og_id: Parent orthogroup identifier.
        hog_ids_left: Frozen set of HOG IDs found on the *left* (upstream) flank.
        hog_ids_right: Frozen set of HOG IDs found on the *right* (downstream) flank.
        edge_type: ``"internal"`` when both flanks exist, ``"left_edge"`` when the
            gene is at the start of its contig, ``"right_edge"`` when at the end,
            and ``"both_edge"`` for single-gene contigs.
    """

    gene_id: str
    genome: str
    og_id: str
    hog_ids_left: frozenset[str] = field(default_factory=frozenset)
    hog_ids_right: frozenset[str] = field(default_factory=frozenset)
    edge_type: str = "internal"


# ---------------------------------------------------------------------------
# HOG table reader
# ---------------------------------------------------------------------------


def read_hog_table(file: str | Path, genomes: list[Genome]) -> dict[str, str]:
    """Parse an OrthoFinder N0.tsv (or equivalent HOG table) into a gene→HOG map.

    The expected file format is tab-separated with:

    * Column 0: HOG ID (e.g. ``N0.HOG0000001``)
    * Column 1: OG (unused)
    * Column 2: Gene tree node (unused)
    * Columns 3+: comma-separated gene IDs for each species whose sample name
      matches a ``Genome.name``.

    Args:
        file: Path to the HOG TSV file.
        genomes: List of :class:`~orthosynassign.lib.gene.Genome` objects; used
            to identify which columns correspond to which sample.

    Returns:
        Dictionary mapping each ``gene_id`` to its ``hog_id``.  Genes that
        appear in multiple HOG rows (should not happen) are overwritten by the
        last occurrence; a warning is emitted.
    """
    file = Path(file)
    if not file.exists():
        raise FileNotFoundError(f"HOG table not found: {file}")

    genome_names = {g.name for g in genomes}
    gene_to_hog: dict[str, str] = {}

    with open(file, encoding="utf-8") as fh:
        header = fh.readline().strip().split("\t")
        # Identify which columns belong to our genomes (skip HOG/OG/GeneTreeNode)
        sample_cols: list[tuple[int, str]] = []
        for col_idx, col_name in enumerate(header):
            if col_name in genome_names:
                sample_cols.append((col_idx, col_name))

        if not sample_cols:
            raise ValueError(
                f"No sample columns matching loaded genomes found in HOG table header: {header}"
            )

        logger.info("Found %d matching sample columns in HOG table", len(sample_cols))

        for line_num, line in enumerate(fh, 2):
            fields = line.strip().split("\t")
            if not fields[0]:
                continue
            hog_id = fields[0]

            for col_idx, _sample in sample_cols:
                if col_idx >= len(fields):
                    continue
                cell = fields[col_idx].strip()
                if not cell:
                    continue
                for raw_gene in cell.replace(",", " ").split():
                    gene_id = raw_gene.strip()
                    if not gene_id:
                        continue
                    if gene_id in gene_to_hog:
                        logger.warning(
                            "Line %d: gene %s already assigned to HOG %s; overwriting with %s",
                            line_num,
                            gene_id,
                            gene_to_hog[gene_id],
                            hog_id,
                        )
                    gene_to_hog[gene_id] = hog_id

    logger.info("Loaded HOG assignments for %d genes from %s", len(gene_to_hog), file)
    return gene_to_hog


# ---------------------------------------------------------------------------
# Flank-window builder
# ---------------------------------------------------------------------------


def build_flank_window(
    genome: Genome,
    gene_idx: int,
    window_n: int,
    hog_map: dict[str, str],
    *,
    strand_aware: bool = True,
) -> FlankRecord:
    """Extract the set of HOG IDs in the ±N neighbourhood of a gene.

    Neighbours are restricted to the same contig (``seqid``).  If
    ``strand_aware`` is ``True`` and the focal gene has a known strand
    (``"+"`` or ``"-"``), *left* refers to genomic-upstream and *right* to
    genomic-downstream based on that strand.

    Args:
        genome: The :class:`~orthosynassign.lib.gene.Genome` containing the gene.
        gene_idx: Index of the focal gene in ``genome._genes``.
        window_n: Half-window size.  Up to *window_n* genes are collected on
            each side.
        hog_map: Mapping of ``gene_id → hog_id`` as returned by
            :func:`read_hog_table`.
        strand_aware: When ``True`` the left/right labelling respects the focal
            gene's strand so that *left* always points upstream.

    Returns:
        A :class:`FlankRecord` with populated ``hog_ids_left``, ``hog_ids_right``,
        and ``edge_type``.
    """
    genes = genome._genes
    focal = genes[gene_idx]
    seqid = focal.seqid
    og_id = focal.og.id if focal.og else ""

    # Collect same-contig gene indices
    contig_indices = [i for i, g in enumerate(genes) if g.seqid == seqid]
    pos_in_contig = contig_indices.index(gene_idx)

    left_indices = contig_indices[max(0, pos_in_contig - window_n) : pos_in_contig]
    right_indices = contig_indices[pos_in_contig + 1 : pos_in_contig + 1 + window_n]

    left_hogs = frozenset(hog_map[genes[i].id] for i in left_indices if genes[i].id in hog_map)
    right_hogs = frozenset(hog_map[genes[i].id] for i in right_indices if genes[i].id in hog_map)

    # Determine edge type
    has_left = bool(left_indices)
    has_right = bool(right_indices)
    if has_left and has_right:
        edge_type = "internal"
    elif not has_left and not has_right:
        edge_type = "both_edge"
    elif not has_left:
        edge_type = "left_edge"
    else:
        edge_type = "right_edge"

    # Flip left/right for reverse-strand genes so "left" = upstream
    if strand_aware and focal.strand == "-":
        left_hogs, right_hogs = right_hogs, left_hogs
        if edge_type == "left_edge":
            edge_type = "right_edge"
        elif edge_type == "right_edge":
            edge_type = "left_edge"

    return FlankRecord(
        gene_id=focal.id,
        genome=genome.name,
        og_id=og_id,
        hog_ids_left=left_hogs,
        hog_ids_right=right_hogs,
        edge_type=edge_type,
    )
