import numpy as np
import pytest

from orthosynassign.lib import calculate_synteny_ratio, get_synteny_engine


class TestCalculateSyntenyRatio:
    @pytest.mark.parametrize(
        "win_a, win_b, expected_ratio, description",
        [
            # 1. Perfect Identity
            ([1, 2], [1, 2], 1.0, "Identical windows"),
            # 2. Partial Overlap
            ([1, 2, 3], [1, 2, 4], 2 / 3, "Partial overlap (2/3)"),
            # 3. Tandem Duplication (1-to-1 matching)
            ([1, 1], [1], 0.5, "A has extra; match is 1/2"),
            ([1, 1], [1, 1], 1.0, "Both have two copies; match is 2/2"),
            # 4. Order Independence
            ([1, 2], [2, 1], 1.0, "Different order, same content"),
            # 5. Length Penalization
            ([1], [1, 2, 3, 4], 0.25, "B is much longer; ratio drops"),
            # 6. No Overlap
            ([1, 2], [3, 4], 0.0, "Zero shared orthogroups"),
            # 7. Empty Inputs
            ([], [1], 0.0, "Window A is empty"),
            ([1], [], 0.0, "Window B is empty"),
            ([], [], 0.0, "Both windows are empty"),
        ],
    )
    def test_calculate_synteny_ratio(self, win_a, win_b, expected_ratio, description):
        """
        Tests the synteny ratio calculation across multiple genomic scenarios.
        Using pytest.approx for floating point comparisons.
        """
        arr_a = np.array(win_a, dtype=np.int32)
        arr_b = np.array(win_b, dtype=np.int32)

        result = calculate_synteny_ratio(arr_a, arr_b)
        assert result == pytest.approx(expected_ratio), f"Failed: {description}"


class TestSyntenyEngineRefinement:
    @pytest.fixture
    def og(self, og_factory):
        """Provides a fresh Orthogroup instance."""
        return og_factory("OG00001")

    def test_refine_integration(self, gene_factory, genome_factory, og_factory, og) -> None:
        """
        Tests the full flow of refine using real functions.
        This ensures Orthogroup, compare_gene_sets, and consolidate_into_sogs
        all talk to each other correctly.
        """
        # 1. Setup: Create two genomes with one perfectly syntenic pair
        genome_a = genome_factory("Genome_A")
        genome_b = genome_factory("Genome_B")

        # We need at least one neighbor to satisfy window_size=2
        # Anchor genes (the focal ones)
        g_a_focal = gene_factory("A_focal", "chr1", 1000, 2000)
        g_b_focal = gene_factory("B_focal", "chr1", 1000, 2000)

        # Syntenic neighbors (to ensure the ratio is 1.0)
        g_a_neighbor = gene_factory("A_neighbor", "chr1", 2100, 3100)
        g_b_neighbor = gene_factory("B_neighbor", "chr1", 2100, 3100)

        # Setup genomic context
        for g in [g_a_focal, g_a_neighbor]:
            genome_a.add_gene(g)
        for g in [g_b_focal, g_b_neighbor]:
            genome_b.add_gene(g)

        # Add focal genes to the test OG
        og.add_gene(g_a_focal)
        og.add_gene(g_b_focal)

        # Assign neighbors to a different OG so they act as anchors
        neighbor_og = og_factory("OG_NEIGHBOR")
        neighbor_og.add_gene(g_a_neighbor)
        neighbor_og.add_gene(g_b_neighbor)

        # Initialize the Engine
        # The engine needs the list of all relevant genomes and OGs
        genomes = [genome_a, genome_b]
        orthogroups = [og, neighbor_og]
        engine = get_synteny_engine(genomes, orthogroups)
        result = engine.refine(0, window_size=2, ratio_threshold=1.0)

        # 3. Assertions
        assert len(result) == 1
        sog = result[0]
        sog_genes = [genomes[genome_idx][gene_idx] for genome_idx, gene_idx in sog]
        assert g_a_focal in sog_genes
        assert g_b_focal in sog_genes

    def test_refine_no_synteny_found(self, gene_factory, genome_factory, og):
        """Test that an OG with no syntenic support returns an empty list."""
        genome_a = genome_factory("Genome_A")
        genome_b = genome_factory("Genome_B")

        # Genes in different scaffolds/locations with no neighbors
        g_a = gene_factory("A1", "chr1", 1000, 2000)
        g_b = gene_factory("B1", "chr2", 5000, 6000)

        genome_a.add_gene(g_a)
        genome_b.add_gene(g_b)
        og.add_gene(g_a)
        og.add_gene(g_b)

        # Since there are no shared neighbors, this should return []
        genomes = [genome_a, genome_b]
        orthogroups = [og]
        engine = get_synteny_engine(genomes, orthogroups)
        result = engine.refine(0, window_size=2, ratio_threshold=1.0)
        assert result == []

    def test_refine_edge_gene_not_oversplit(self, gene_factory, genome_factory, og_factory) -> None:
        """Edge gene (start of contig) should still be grouped with its syntenic
        partner even when the internal gene has more neighbours.

        Genome A layout (contig 'chr1'):
            [A_focal] [A_n1] [A_n2]    -- A_focal is at the LEFT edge (no left neighbour)

        Genome B layout (contig 'chr1'):
            [B_n_left] [B_focal] [B_n1] [B_n2]   -- B_focal is internal

        Both A_n1/A_n2 and B_n1/B_n2 (and B_n_left) belong to the same pair of
        shared anchor OGs so the comparison is:
            p_win (A_focal) = [OG_anchor1, OG_anchor2]   (only right side)
            s_win (B_focal) = [OG_anchor1, OG_anchor2]   (from both sides combined)

        Without edge adjustment: ratio = 2 / max(2, 2) = 1.0 -- passes anyway in this
        minimal case.  We therefore use a stricter scenario where the internal genome
        has an *extra* anchor that the edge genome cannot see:

        Genome A (chr1): [A_focal] [A_n1] [A_n2]
            A_focal window: OG_anchor1, OG_anchor2  (right only, 2 genes)
        Genome B (chr1): [B_n_left] [B_n_extra] [B_focal] [B_n1] [B_n2]
            B_focal window (half_win=2 each side): OG_extra, OG_anchor1 | OG_anchor2 (3 anchors total,
            but only 2 shared with A, window = 4 genes across 2+2)

        Without edge adjustment: ratio = 2 / max(2, 4) = 0.5 -- may fail threshold 0.6.
        With edge adjustment:    ratio = 2 / min(2, 4) = 1.0 -- passes.
        """
        genome_a = genome_factory("Genome_A")
        genome_b = genome_factory("Genome_B")

        # Focal genes (the OG we're refining)
        og = og_factory("OG_FOCAL")
        g_a_focal = gene_factory("A_focal", "chr1", 1000, 2000)
        g_b_focal = gene_factory("B_focal", "chr1", 3000, 4000)

        # Shared anchor OGs (appear in both genomes)
        og_anchor1 = og_factory("OG_ANCHOR1")
        og_anchor2 = og_factory("OG_ANCHOR2")

        # Extra anchor present only in Genome B (adds to B's window, not A's)
        og_extra = og_factory("OG_EXTRA")

        # --- Genome A: A_focal at contig start, no left neighbour ---
        a_n1 = gene_factory("A_n1", "chr1", 2100, 3100)  # OG_ANCHOR1
        a_n2 = gene_factory("A_n2", "chr1", 3200, 4200)  # OG_ANCHOR2
        for g in [g_a_focal, a_n1, a_n2]:
            genome_a.add_gene(g)

        # --- Genome B: B_focal is internal with 2 left + 2 right neighbours ---
        b_n_left1 = gene_factory("B_n_extra", "chr1", 1000, 2000)  # OG_EXTRA
        b_n_left2 = gene_factory("B_n_left", "chr1", 2100, 3100)   # OG_ANCHOR1 (left of focal)
        b_n1 = gene_factory("B_n1", "chr1", 4100, 5100)            # OG_ANCHOR2 (right of focal)
        b_n2 = gene_factory("B_n2", "chr1", 5200, 6200)            # OG_ANCHOR2 duplicate (filler)
        for g in [b_n_left1, b_n_left2, g_b_focal, b_n1, b_n2]:
            genome_b.add_gene(g)

        og.add_gene(g_a_focal)
        og.add_gene(g_b_focal)

        og_anchor1.add_gene(a_n1)
        og_anchor1.add_gene(b_n_left2)

        og_anchor2.add_gene(a_n2)
        og_anchor2.add_gene(b_n1)

        og_extra.add_gene(b_n_left1)
        # b_n2 gets no OG → acts as genomic filler with no shared synteny

        genomes = [genome_a, genome_b]
        orthogroups = [og, og_anchor1, og_anchor2, og_extra]
        engine = get_synteny_engine(genomes, orthogroups)

        # threshold=0.6: without edge adjustment 2/4=0.5 fails; with it 2/2=1.0 passes
        result = engine.refine(0, window_size=4, ratio_threshold=0.6)
        assert len(result) == 1, (
            "Edge gene should be grouped with its syntenic partner; got %r" % result
        )
        sog = result[0]
        sog_genes = [genomes[gi][gene_i] for gi, gene_i in sog]
        assert g_a_focal in sog_genes
        assert g_b_focal in sog_genes

# class TestVisualizeEngine:
#     def test_integration_get_aligned_og(self, gene_factory, genome_factory, og_factory):
#         """
#         Test that two neighborhoods with different focal gene offsets
#         are shifted to match a common pivot.
#         """
#         g1, g2, g3 = gene_factory("G1"), gene_factory("G2"), gene_factory("G3")
#         focal_a, focal_b = gene_factory("focal_a"), gene_factory("focal_B")

#         genome_a = genome_factory("Genome_A")
#         genome_b = genome_factory("Genome_B")

#         for g in [g_a_focal, g_a_neighbor]:
#             genome_a.add_gene(g)
#         for g in [g_b_focal, g_b_neighbor]:
#             genome_b.add_gene(g)

#         # We need at least one neighbor to satisfy window_size=2
#         # Anchor genes (the focal ones)
#         g_a_focal = gene_factory("A_focal", "chr1", 1000, 2000)
#         g_b_focal = gene_factory("B_focal", "chr1", 1000, 2000)

#         # Syntenic neighbors (to ensure the ratio is 1.0)
#         g_a_neighbor = gene_factory("A_neighbor", "chr1", 2100, 3100)
#         g_b_neighbor = gene_factory("B_neighbor", "chr1", 2100, 3100)


#         # Scenario:
#         # Dict 1: [G1, G2, focal_A] -> focal is at index 2
#         # Dict 2: [focal_B, G3]     -> focal is at index 0
#         engine = get_visualize_engine()
#         engine.get_aligned_og
#         sog_dict = {focal_a: [g1, g2, focal_a], focal_b: [focal_b, g3]}

#         aligned = align_sog_dict(sog_dict)

#         # The pivot should be 2 (the max index of a focal gene)
#         # List 1: [G1, G2, focal_A] (no change to front, needs 0 backpad)
#         # List 2: [None, None, focal_B, G3] (needs 2 frontpads to move focal_B to index 2)

#         assert aligned[focal_a] == [g1, g2, focal_a, None]
#         assert aligned[focal_b] == [None, None, focal_b, g3]
#         assert len(aligned[focal_a]) == len(aligned[focal_b])
