use pyo3::prelude::*;
use rayon::prelude::*;
use std::collections::HashMap;

/// Intermediate per-gene flank data computed in Phase 2.
struct GeneFlankData {
    genome_idx: usize,
    gene_idx: usize,
    og_idx: i32,
    left_hogs: Vec<i32>,  // sorted, deduped HOG indices for left flank
    right_hogs: Vec<i32>, // sorted, deduped HOG indices for right flank
    edge_type: u8,        // 0=internal, 1=left_edge, 2=right_edge, 3=both_edge
    completeness: f64,
}

/// Pre-built contig-position data for a single genome.
struct ContigData {
    /// Maps seqid → sorted list of gene indices in that contig.
    contig_to_genes: HashMap<i16, Vec<usize>>,
    /// gene_idx → position within its contig.
    gene_contig_pos: Vec<usize>,
}

impl ContigData {
    fn build(seqids: &[i16]) -> Self {
        let n = seqids.len();
        let mut contig_to_genes: HashMap<i16, Vec<usize>> = HashMap::new();
        for (idx, &seqid) in seqids.iter().enumerate() {
            contig_to_genes.entry(seqid).or_default().push(idx);
        }
        // Gene indices were pushed in ascending order, so each Vec is already sorted.

        let mut gene_contig_pos = vec![0usize; n];
        for genes_in_contig in contig_to_genes.values() {
            for (pos, &gene_idx) in genes_in_contig.iter().enumerate() {
                gene_contig_pos[gene_idx] = pos;
            }
        }

        ContigData {
            contig_to_genes,
            gene_contig_pos,
        }
    }
}

/// Compute the sorted, deduplicated union of two already-sorted, deduplicated slices.
fn sorted_union(a: &[i32], b: &[i32]) -> Vec<i32> {
    let mut result = Vec::with_capacity(a.len() + b.len());
    let (mut i, mut j) = (0, 0);
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Less => {
                result.push(a[i]);
                i += 1;
            }
            std::cmp::Ordering::Greater => {
                result.push(b[j]);
                j += 1;
            }
            std::cmp::Ordering::Equal => {
                result.push(a[i]);
                i += 1;
                j += 1;
            }
        }
    }
    result.extend_from_slice(&a[i..]);
    result.extend_from_slice(&b[j..]);
    result
}

/// Jaccard similarity between two sorted, deduplicated integer slices.
/// Returns 0.0 when both slices are empty.
fn jaccard_sorted_sets(a: &[i32], b: &[i32]) -> f64 {
    if a.is_empty() && b.is_empty() {
        return 0.0;
    }
    let mut intersection = 0usize;
    let (mut i, mut j) = (0, 0);
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Equal => {
                intersection += 1;
                i += 1;
                j += 1;
            }
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
        }
    }
    let union_size = a.len() + b.len() - intersection;
    if union_size == 0 {
        return 0.0;
    }
    intersection as f64 / union_size as f64
}

/// High-performance Rust backend for flank-score computation.
///
/// Accepts pre-vectorised genome data (seqid indices, HOG indices, OG indices, strand)
/// and computes per-gene flank scores and related metrics in a single parallel pass.
#[pyclass]
#[pyo3(name = "FlankEngine")]
pub struct FlankEngine {
    /// seqid index per gene, per genome.
    seqids_vec: Vec<Vec<i16>>,
    /// HOG index per gene, per genome.  -1 when no HOG is assigned.
    hog_idxs_vec: Vec<Vec<i32>>,
    /// OG index per gene, per genome.  -1 when no OG is assigned.
    og_idxs_vec: Vec<Vec<i32>>,
    /// Strand per gene, per genome.  1 = '+', -1 = '-', 0 = '.' / unknown.
    strand_vec: Vec<Vec<i8>>,
    /// Whether each genome is circular (unused for flank windows — kept for API symmetry).
    #[allow(dead_code)]
    is_circular_vec: Vec<bool>,
    /// Total number of orthogroups (used for bounds checking only).
    #[allow(dead_code)]
    num_ogs: usize,
}

#[pymethods]
impl FlankEngine {
    /// Create a new FlankEngine.
    ///
    /// Args:
    ///     seqids_all:      Per-genome list of seqid indices (i16) per gene.
    ///     hog_idxs_all:    Per-genome list of HOG indices (i32) per gene; -1 if unassigned.
    ///     og_idxs_all:     Per-genome list of OG indices (i32) per gene; -1 if unassigned.
    ///     strand_all:      Per-genome list of strand values (i8): 1='+', -1='-', 0='.'.
    ///     is_circular_all: Per-genome circularity flags.
    ///     num_ogs:         Total number of orthogroups.
    #[new]
    pub fn new(
        seqids_all: Vec<Vec<i16>>,
        hog_idxs_all: Vec<Vec<i32>>,
        og_idxs_all: Vec<Vec<i32>>,
        strand_all: Vec<Vec<i8>>,
        is_circular_all: Vec<bool>,
        num_ogs: usize,
    ) -> PyResult<Self> {
        Ok(FlankEngine {
            seqids_vec: seqids_all,
            hog_idxs_vec: hog_idxs_all,
            og_idxs_vec: og_idxs_all,
            strand_vec: strand_all,
            is_circular_vec: is_circular_all,
            num_ogs,
        })
    }

    /// Compute flank scores and related metrics for every gene that has an OG assignment.
    ///
    /// The function releases the Python GIL and uses Rayon for parallelism.
    ///
    /// Args:
    ///     window_n:     Half-window size — up to *window_n* genes are examined on each side.
    ///     strand_aware: When True, left/right labels are flipped for minus-strand genes so
    ///                   that "left" always means genomic-upstream.
    ///
    /// Returns:
    ///     A list of 7-tuples, one per qualifying gene:
    ///         (genome_idx, gene_idx, flank_score, flank_completeness, edge_type,
    ///          left_hog_idxs, right_hog_idxs)
    ///
    ///     edge_type encoding:  0=internal, 1=left_edge, 2=right_edge, 3=both_edge
    pub fn compute_all(
        &self,
        py: Python<'_>,
        window_n: usize,
        strand_aware: bool,
    ) -> Vec<(usize, usize, f64, f64, u8, Vec<i32>, Vec<i32>)> {
        py.detach(|| self.compute_all_logic(window_n, strand_aware))
    }
}

impl FlankEngine {
    fn compute_all_logic(
        &self,
        window_n: usize,
        strand_aware: bool,
    ) -> Vec<(usize, usize, f64, f64, u8, Vec<i32>, Vec<i32>)> {
        // ----------------------------------------------------------------
        // Phase 1: Build per-genome contig-position maps (sequential, O(N)).
        // ----------------------------------------------------------------
        let contig_data: Vec<ContigData> = self
            .seqids_vec
            .iter()
            .map(|seqids| ContigData::build(seqids))
            .collect();

        // ----------------------------------------------------------------
        // Phase 2: Collect qualifying (genome_idx, gene_idx) pairs.
        // ----------------------------------------------------------------
        let gene_pairs: Vec<(usize, usize)> = self
            .og_idxs_vec
            .iter()
            .enumerate()
            .flat_map(|(g_idx, og_idxs)| {
                og_idxs
                    .iter()
                    .enumerate()
                    .filter(|(_, &og)| og >= 0)
                    .map(move |(gene_i, _)| (g_idx, gene_i))
            })
            .collect();

        // ----------------------------------------------------------------
        // Phase 3: Compute flank data for every qualifying gene in parallel.
        // ----------------------------------------------------------------
        let flank_data: Vec<GeneFlankData> = gene_pairs
            .into_par_iter()
            .map(|(genome_idx, gene_idx)| {
                self.compute_gene_flank(genome_idx, gene_idx, window_n, strand_aware, &contig_data)
            })
            .collect();

        // ----------------------------------------------------------------
        // Phase 4: Group HOG unions by OG (sequential, O(N)).
        // Each entry is a sorted, deduplicated union of left ∪ right HOGs.
        // ----------------------------------------------------------------
        let mut og_to_unions: HashMap<i32, Vec<Vec<i32>>> = HashMap::new();
        for data in &flank_data {
            let union = sorted_union(&data.left_hogs, &data.right_hogs);
            og_to_unions.entry(data.og_idx).or_default().push(union);
        }

        // ----------------------------------------------------------------
        // Phase 5: Compute per-gene mean Jaccard scores in parallel.
        // ----------------------------------------------------------------
        flank_data
            .into_par_iter()
            .map(|data| {
                let my_union = sorted_union(&data.left_hogs, &data.right_hogs);

                let score = match og_to_unions.get(&data.og_idx) {
                    None => 0.0,
                    Some(peer_unions) => {
                        // Replicate Python's self-exclusion logic: remove peers whose
                        // union is equal to the focal gene's union (set-value equality).
                        let refs: Vec<&Vec<i32>> = peer_unions
                            .iter()
                            .filter(|u| !u.is_empty() && u.as_slice() != my_union.as_slice())
                            .collect();

                        if refs.is_empty() {
                            0.0
                        } else {
                            let total: f64 = refs
                                .iter()
                                .map(|u| jaccard_sorted_sets(&my_union, u))
                                .sum();
                            total / refs.len() as f64
                        }
                    }
                };

                (
                    data.genome_idx,
                    data.gene_idx,
                    score,
                    data.completeness,
                    data.edge_type,
                    data.left_hogs,
                    data.right_hogs,
                )
            })
            .collect()
    }

    /// Compute flank window data for a single gene.
    fn compute_gene_flank(
        &self,
        genome_idx: usize,
        gene_idx: usize,
        window_n: usize,
        strand_aware: bool,
        contig_data: &[ContigData],
    ) -> GeneFlankData {
        let seqid = self.seqids_vec[genome_idx][gene_idx];
        let cd = &contig_data[genome_idx];
        let contig_genes = &cd.contig_to_genes[&seqid];
        let pos = cd.gene_contig_pos[gene_idx];
        let og_idx = self.og_idxs_vec[genome_idx][gene_idx];
        let strand = self.strand_vec[genome_idx][gene_idx];

        // Determine the left and right slices of gene indices within this contig.
        // This matches build_flank_window: always linear (no circular wrapping).
        let left_start = pos.saturating_sub(window_n);
        let left_genes = &contig_genes[left_start..pos];
        let right_end = (pos + 1 + window_n).min(contig_genes.len());
        let right_genes = &contig_genes[pos + 1..right_end];

        // Edge type is based on the *existence* of any contig neighbours, not
        // whether they carry a HOG assignment — matching build_flank_window behaviour.
        let has_left = pos > 0;
        let has_right = pos + 1 < contig_genes.len();
        let edge_type: u8 = match (has_left, has_right) {
            (true, true) => 0,   // internal
            (false, true) => 1,  // left_edge
            (true, false) => 2,  // right_edge
            (false, false) => 3, // both_edge
        };

        // Collect HOG indices from each side, filtering for assigned genes only.
        let mut left_hogs: Vec<i32> = left_genes
            .iter()
            .filter_map(|&g| {
                let h = self.hog_idxs_vec[genome_idx][g];
                if h >= 0 {
                    Some(h)
                } else {
                    None
                }
            })
            .collect();
        left_hogs.sort_unstable();
        left_hogs.dedup();

        let mut right_hogs: Vec<i32> = right_genes
            .iter()
            .filter_map(|&g| {
                let h = self.hog_idxs_vec[genome_idx][g];
                if h >= 0 {
                    Some(h)
                } else {
                    None
                }
            })
            .collect();
        right_hogs.sort_unstable();
        right_hogs.dedup();

        // Strand-aware swap: flip left/right (and adjust edge_type) for '-' genes.
        let (final_left, final_right, final_edge_type) = if strand_aware && strand == -1 {
            let flipped = match edge_type {
                1 => 2, // left_edge  → right_edge
                2 => 1, // right_edge → left_edge
                other => other,
            };
            (right_hogs, left_hogs, flipped)
        } else {
            (left_hogs, right_hogs, edge_type)
        };

        // Flank completeness: fraction of window slots that are on-contig.
        let n_contig = contig_genes.len();
        let completeness = if window_n == 0 {
            1.0
        } else {
            let available_left = pos.min(window_n);
            let available_right = (n_contig - pos - 1).min(window_n);
            let total_slots = 2 * window_n;
            ((available_left + available_right) as f64 / total_slots as f64)
                .max(1.0 / total_slots as f64)
        };

        GeneFlankData {
            genome_idx,
            gene_idx,
            og_idx,
            left_hogs: final_left,
            right_hogs: final_right,
            edge_type: final_edge_type,
            completeness,
        }
    }
}
