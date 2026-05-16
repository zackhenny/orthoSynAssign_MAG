"""
Flank-score computation and training-table export utilities.

The *flank score* measures how similar the HOG neighbourhood of a focal gene
is to the HOG neighbourhoods observed across all other members of its parent
orthogroup.  A high score means the local gene order around this gene is
consistent with its orthogroup peers → the SOG assignment is likely reliable.
A low score suggests the gene may be a split artifact or misassigned.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .gene import Gene, Genome
    from .hog import FlankRecord

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core scoring functions
# ---------------------------------------------------------------------------


def jaccard(set_a: frozenset[str], set_b: frozenset[str]) -> float:
    """Compute the Jaccard similarity between two sets of HOG IDs.

    Returns 0.0 when both sets are empty.

    Args:
        set_a: First set of HOG IDs.
        set_b: Second set of HOG IDs.

    Returns:
        Jaccard index in [0, 1].
    """
    if not set_a and not set_b:
        return 0.0
    union = set_a | set_b
    intersection = set_a & set_b
    return len(intersection) / len(union)


def flank_score(
    left_hogs: frozenset[str],
    right_hogs: frozenset[str],
    reference_hog_sets: list[frozenset[str]],
) -> float:
    """Compute the mean Jaccard similarity of a gene's flank HOGs vs. reference sets.

    The *reference_hog_sets* are the combined (left ∪ right) HOG sets of all
    other members of the parent orthogroup, collected across all genomes.  Each
    reference set represents the neighbourhood context for one orthogroup member.

    The focal gene's neighbourhood is also expressed as left ∪ right.  The
    score is the mean Jaccard of the focal union against every reference union.

    Args:
        left_hogs: HOG IDs on the upstream flank of the focal gene.
        right_hogs: HOG IDs on the downstream flank of the focal gene.
        reference_hog_sets: List of combined-flank HOG sets from reference genes.

    Returns:
        Mean Jaccard similarity ∈ [0, 1].  Returns 0.0 if *reference_hog_sets*
        is empty.
    """
    if not reference_hog_sets:
        return 0.0
    focal_union = left_hogs | right_hogs
    scores = [jaccard(focal_union, ref) for ref in reference_hog_sets]
    return sum(scores) / len(scores)


def flank_completeness(gene: Gene, genome: Genome, window_n: int) -> float:
    """Fraction of the ±N window slots that are on-contig (non-edge).

    A gene at the very start of a contig with window_n=4 can only draw 0
    upstream neighbours; completeness = 4/8 = 0.5.  An internal gene has
    completeness 1.0.

    Args:
        gene: The focal gene (must have ``gene.index`` set and ``gene.seqid``).
        genome: The genome containing the gene.
        window_n: Half-window size (same value used in
            :func:`~orthosynassign.lib.hog.build_flank_window`).

    Returns:
        Completeness ∈ (0, 1].
    """
    if window_n == 0:
        return 1.0

    genes = genome._genes
    seqid = gene.seqid

    contig_indices = [i for i, g in enumerate(genes) if g.seqid == seqid]
    pos_in_contig = contig_indices.index(gene.index)

    available_left = min(pos_in_contig, window_n)
    available_right = min(len(contig_indices) - pos_in_contig - 1, window_n)

    total_slots = 2 * window_n
    available_slots = available_left + available_right

    return max(available_slots / total_slots, 1 / total_slots)


# ---------------------------------------------------------------------------
# Training-table builder
# ---------------------------------------------------------------------------


def build_training_table(
    flank_records: list[FlankRecord],
    sog_assignments: dict[str, str],
    og_sizes: dict[str, int],
    genome_completeness: dict[str, float],
) -> list[dict]:
    """Assemble the training table rows used by the statistical models.

    Args:
        flank_records: One :class:`~orthosynassign.lib.hog.FlankRecord` per gene.
        sog_assignments: Mapping of ``gene_id → sog_id`` from the orthoSynAssign
            output.  A gene is labelled ``is_split = 1`` when its SOG contains
            only one genome (singleton) or when the SOG ID differs from the
            majority SOG within its orthogroup.
        og_sizes: Mapping of ``og_id → number of unique genomes represented``.
        genome_completeness: Mapping of ``genome_name → completeness fraction``
            (e.g. from CheckM2 output; pass a dict of 1.0 values if unavailable).

    Returns:
        List of row dicts with keys: ``gene_id``, ``genome``, ``og_id``,
        ``og_size``, ``genome_completeness``, ``flank_score``, ``flank_left_hogs``,
        ``flank_right_hogs``, ``flank_completeness``, ``edge_type``, ``is_split``.
    """
    # Build reference HOG sets per OG (combine all flank records for same OG)
    from collections import Counter, defaultdict

    og_to_ref_sets: dict[str, list[frozenset[str]]] = defaultdict(list)
    for rec in flank_records:
        if rec.og_id:
            og_to_ref_sets[rec.og_id].append(rec.hog_ids_left | rec.hog_ids_right)

    # Pre-compute the global SOG counts once to avoid an O(n²) scan per record.
    sog_global_counts: Counter[str] = Counter(sog_assignments.values())
    has_multiple_assignments = len(sog_assignments) > 1

    rows: list[dict] = []
    for rec in flank_records:
        ref_sets = [s for s in og_to_ref_sets.get(rec.og_id, []) if s]

        # Exclude self from reference
        focal_union = rec.hog_ids_left | rec.hog_ids_right
        ref_sets_no_self = [s for s in ref_sets if s != focal_union] if ref_sets else []

        score = flank_score(rec.hog_ids_left, rec.hog_ids_right, ref_sets_no_self)

        sog_id = sog_assignments.get(rec.gene_id, "")
        is_split = int(_is_split_fast(rec.gene_id, sog_assignments, sog_global_counts, has_multiple_assignments))

        rows.append(
            {
                "gene_id": rec.gene_id,
                "genome": rec.genome,
                "og_id": rec.og_id,
                "og_size": og_sizes.get(rec.og_id, 0),
                "genome_completeness": genome_completeness.get(rec.genome, 1.0),
                "flank_score": score,
                "flank_left_hogs": ";".join(sorted(rec.hog_ids_left)),
                "flank_right_hogs": ";".join(sorted(rec.hog_ids_right)),
                "edge_type": rec.edge_type,
                "sog_id": sog_id,
                "is_split": is_split,
            }
        )
    return rows


def _is_split_fast(
    gene_id: str,
    sog_assignments: dict[str, str],
    sog_global_counts: Counter[str],
    has_multiple_assignments: bool,
) -> bool:
    """Return True if this gene ended up in a SOG that doesn't appear in any other assignment.

    Uses a pre-computed Counter to achieve O(1) per-gene lookups instead of
    iterating over the entire sog_assignments dict for every gene (which was O(n²)).

    A gene is considered "split" when:
    - There is more than one gene with a SOG assignment, AND
    - No other gene shares this gene's SOG (i.e. the SOG count is exactly 1).
    """
    if not has_multiple_assignments or gene_id not in sog_assignments:
        return False
    gene_sog = sog_assignments[gene_id]
    return sog_global_counts.get(gene_sog, 0) == 1


def _is_split(gene_id: str, og_id: str, sog_assignments: dict[str, str]) -> bool:
    """Return True if this gene ended up in a SOG that doesn't represent the majority of its OG.

    .. deprecated::
        Use :func:`_is_split_fast` with a pre-computed Counter instead.
        This implementation has O(n) cost per call, making overall complexity O(n²).
    """
    if not og_id or gene_id not in sog_assignments:
        return False

    from collections import Counter

    gene_sog = sog_assignments[gene_id]
    og_sogs = [v for k, v in sog_assignments.items() if k != gene_id]
    counts = Counter(og_sogs)
    total = sum(counts.values())
    return total > 0 and counts.get(gene_sog, 0) == 0
