import numpy as np
import pytest

from orthosynassign.lib import calculate_directional_synteny_ratio, calculate_synteny_ratio, get_synteny_engine, get_window_split


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


class TestGetWindowSplit:
    """Tests for the directional split window function."""

    def test_split_internal_gene(self):
        """Internal gene returns equal left and right windows."""
        # seqid = all 0 (same contig), 7 genes total, focal is index 3
        seqid_vec = [0, 0, 0, 0, 0, 0, 0]
        og_mask = [True] * 7
        left, right = get_window_split(og_mask, seqid_vec, gene_idx=3, window_size=4, is_circular=False)
        assert left == [2, 1] or left == sorted(left, reverse=False)
        # Half window = 2, so 2 left and 2 right
        assert len(left) == 2
        assert len(right) == 2
        assert all(idx < 3 for idx in left)
        assert all(idx > 3 for idx in right)

    def test_split_left_edge_gene(self):
        """Gene at left edge of contig has empty left window."""
        seqid_vec = [0, 0, 0, 0, 0]
        og_mask = [True] * 5
        left, right = get_window_split(og_mask, seqid_vec, gene_idx=0, window_size=4, is_circular=False)
        assert left == []
        assert len(right) == 2
        assert right[0] == 1

    def test_split_right_edge_gene(self):
        """Gene at right edge of contig has empty right window."""
        seqid_vec = [0, 0, 0, 0, 0]
        og_mask = [True] * 5
        left, right = get_window_split(og_mask, seqid_vec, gene_idx=4, window_size=4, is_circular=False)
        assert right == []
        assert len(left) == 2
        assert left[-1] == 3

    def test_split_near_left_edge(self):
        """Gene one step from left edge has 1 left and up to 2 right neighbors."""
        seqid_vec = [0, 0, 0, 0, 0]
        og_mask = [True] * 5
        left, right = get_window_split(og_mask, seqid_vec, gene_idx=1, window_size=4, is_circular=False)
        assert left == [0]
        assert right == [2, 3]

    def test_split_contig_boundary_respected(self):
        """Genes on different contigs do not bleed into each other."""
        # Two contigs: [0,0,0] and [1,1,1]
        seqid_vec = [0, 0, 0, 1, 1, 1]
        og_mask = [True] * 6
        # Gene at index 2 (last gene of contig 0)
        left, right = get_window_split(og_mask, seqid_vec, gene_idx=2, window_size=4, is_circular=False)
        assert 3 not in right and 4 not in right and 5 not in right  # no cross-contig
        assert right == []  # right edge of its contig

    def test_split_og_mask_filters_neighbors(self):
        """Only genes passing the OG mask are included."""
        seqid_vec = [0, 0, 0, 0, 0]
        og_mask = [False, True, False, True, False]  # only indices 1 and 3 pass
        left, right = get_window_split(og_mask, seqid_vec, gene_idx=2, window_size=4, is_circular=False)
        assert left == [1]
        assert right == [3]

    def test_split_ascending_order_preserved(self):
        """Left indices are in ascending order (smallest first)."""
        seqid_vec = [0] * 10
        og_mask = [True] * 10
        left, right = get_window_split(og_mask, seqid_vec, gene_idx=5, window_size=4, is_circular=False)
        assert left == sorted(left)
        assert right == sorted(right)


class TestCalculateDirectionalSyntenyRatio:
    """Tests for the directional synteny ratio function."""

    def test_both_internal_same_as_calculate_synteny_ratio(self):
        """Both internal: behaviour matches calculate_synteny_ratio (merged, max denom)."""
        left_a = [1, 2]
        right_a = [3, 4]
        left_b = [1, 2]
        right_b = [3, 4]
        expected = calculate_synteny_ratio(
            np.array([1, 2, 3, 4], dtype=np.int32),
            np.array([1, 2, 3, 4], dtype=np.int32),
        )
        result = calculate_directional_synteny_ratio(
            left_a, right_a, left_b, right_b,
            False, False, False, False,
        )
        assert result == pytest.approx(expected)

    def test_left_edge_a_compares_right_sides_only(self):
        """A is left-edge only: only right sides are compared."""
        # A has no left context; right=[3,4] for A, right=[3,4] for B
        # B has left=[1,2] which should be ignored
        result = calculate_directional_synteny_ratio(
            [], [3, 4],          # A: no left, right=[3,4]
            [1, 2], [3, 4],      # B: left=[1,2] (ignored), right=[3,4]
            True, False,         # A: left_edge=True, right_edge=False
            False, False,        # B: internal
        )
        # Should match right sides: 2/min(2,2) = 1.0
        assert result == pytest.approx(1.0)

    def test_right_edge_a_compares_left_sides_only(self):
        """A is right-edge only: only left sides are compared."""
        result = calculate_directional_synteny_ratio(
            [1, 2], [],          # A: left=[1,2], no right
            [1, 2], [5, 6],      # B: left=[1,2], right=[5,6] (ignored)
            False, True,         # A: right_edge=True
            False, False,        # B: internal
        )
        # Should match left sides: 2/min(2,2) = 1.0
        assert result == pytest.approx(1.0)

    def test_edge_gene_not_penalised_for_missing_context(self):
        """Edge gene with full right match should not be penalised for absent left."""
        # Reproduce the plan's example: A is at left edge, B is internal with extra left neighbor
        # A right:  [OG_ANCHOR1=1, OG_ANCHOR2=2]  (only right context)
        # B left:   [OG_EXTRA=0, OG_ANCHOR1=1]
        # B right:  [OG_ANCHOR2=2]
        # Non-directional (merged, max denom): 2 matches out of max(2, 3) = 0.67
        # Directional (right vs right):  matches([2], [2]) / min(2, 1) = 1/1 = 1.0
        # BUT B has only 1 right neighbor, so min(2,1)=1; matches=1 (OG_ANCHOR2) → 1.0
        result = calculate_directional_synteny_ratio(
            [], [1, 2],          # A: left_edge, right=[anchor1, anchor2]
            [0, 1], [2],         # B: internal, left=[extra, anchor1], right=[anchor2]
            True, False,         # A: left_edge
            False, False,        # B: internal
        )
        # Compare right only: A_right=[1,2], B_right=[2] → match=1, denom=min(2,1)=1 → 1.0
        assert result == pytest.approx(1.0)

    def test_both_edge_same_side_uses_that_side(self):
        """Both genes are left-edge: compare right sides."""
        result = calculate_directional_synteny_ratio(
            [], [1, 2],          # A: left_edge, right=[1,2]
            [], [1, 2],          # B: left_edge, right=[1,2]
            True, False,
            True, False,
        )
        # use_left = !True && !True = False
        # use_right = !False && !False = True → compare right sides
        # 2/min(2,2) = 1.0
        assert result == pytest.approx(1.0)

    def test_both_edges_no_comparable_side_uses_fallback(self):
        """A is left-edge, B is right-edge: no comparable side → fallback to merged min-denom."""
        result = calculate_directional_synteny_ratio(
            [], [1, 2],          # A: left_edge, right=[1,2]
            [1, 2], [],          # B: right_edge, left=[1,2]
            True, False,         # A: left_edge_a
            False, True,         # B: right_edge_b
        )
        # use_left = !True && !False = False
        # use_right = !False && !True = False
        # Fallback: merge A=[1,2], B=[1,2], min(2,2)=2, matches=2 → 1.0
        assert result == pytest.approx(1.0)

    def test_empty_windows_return_zero(self):
        """All-empty windows return 0."""
        result = calculate_directional_synteny_ratio(
            [], [], [], [], True, True, True, True,
        )
        assert result == pytest.approx(0.0)

    def test_near_edge_gene_partial_right(self):
        """Gene 1 step from right edge: right side has 1 gene, left side has up to half_win."""
        # A: 1 step from right edge, so A_right=[anchor], A_left=[a1, a2]
        # B: internal, B_right=[anchor, x], B_left=[a1, a2]
        result = calculate_directional_synteny_ratio(
            [5, 6], [7],         # A: left=[a1=5,a2=6], right=[anchor=7]
            [5, 6], [7, 8],      # B: left=[a1,a2], right=[anchor,x=8]
            False, False,        # A: fully internal (right is just shorter, not right_edge)
            False, False,        # B: internal
        )
        # Both internal → merged windows: A=[5,6,7], B=[5,6,7,8]
        # matches=3, denom=max(3,4)=4 → 0.75
        expected = calculate_synteny_ratio(
            np.array([5, 6, 7], dtype=np.int32),
            np.array([5, 6, 7, 8], dtype=np.int32),
        )
        assert result == pytest.approx(expected)


class TestDirectionalWindowIntegrationWithEngine:
    """Integration tests verifying that the engine uses directional windows correctly."""

    def test_right_edge_gene_matches_left_only(self, gene_factory, genome_factory, og_factory):
        """Gene at right contig edge is correctly matched using only left context.

        Layout:
            Genome A: [A_n2] [A_n1] [A_focal]   (A_focal at right edge)
            Genome B: [B_n2] [B_n1] [B_focal] [B_extra]  (B_focal internal)

        A_n1 and B_n1 share OG_ANCHOR1; A_n2 and B_n2 share OG_ANCHOR2.
        B_extra is in OG_EXTRA (not in A). Without directional logic, B_focal's
        right neighbor inflates its window size and can reduce the ratio.
        """
        genome_a = genome_factory("Genome_A")
        genome_b = genome_factory("Genome_B")

        og = og_factory("OG_FOCAL")
        og_a1 = og_factory("OG_ANCHOR1")
        og_a2 = og_factory("OG_ANCHOR2")
        og_extra = og_factory("OG_EXTRA")

        # Genome A: right edge gene
        a_n2 = gene_factory("A_n2", "chr1", 100, 200)
        a_n1 = gene_factory("A_n1", "chr1", 300, 400)
        g_a_focal = gene_factory("A_focal", "chr1", 500, 600)

        # Genome B: internal gene
        b_n2 = gene_factory("B_n2", "chr1", 100, 200)
        b_n1 = gene_factory("B_n1", "chr1", 300, 400)
        g_b_focal = gene_factory("B_focal", "chr1", 500, 600)
        b_extra = gene_factory("B_extra", "chr1", 700, 800)

        for g in [a_n2, a_n1, g_a_focal]:
            genome_a.add_gene(g)
        for g in [b_n2, b_n1, g_b_focal, b_extra]:
            genome_b.add_gene(g)

        og.add_gene(g_a_focal)
        og.add_gene(g_b_focal)
        og_a1.add_gene(a_n1)
        og_a1.add_gene(b_n1)
        og_a2.add_gene(a_n2)
        og_a2.add_gene(b_n2)
        og_extra.add_gene(b_extra)

        genomes = [genome_a, genome_b]
        orthogroups = [og, og_a1, og_a2, og_extra]
        engine = get_synteny_engine(genomes, orthogroups)

        # With window_size=4 (half_win=2): A has 2 left, B has 2 left + 1 right.
        # Directional: compare left sides only (A is right-edge) → ratio = 1.0 → passes.
        result = engine.refine(0, window_size=4, ratio_threshold=0.6)
        assert len(result) == 1, "Right-edge gene should match its internal partner"
        sog_genes = [genomes[gi][gene_i] for gi, gene_i in result[0]]
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
