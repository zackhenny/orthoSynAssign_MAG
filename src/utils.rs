use rayon::prelude::*;
use std::collections::{HashMap, HashSet};

pub fn get_orthogroups_vec(num_orthogroups: usize, ogs: &[Vec<i32>]) -> Vec<Vec<(usize, usize)>> {
    let mut orthogroups: Vec<Vec<(usize, usize)>> = vec![Vec::new(); num_orthogroups];
    for (genome_idx, og_vec) in ogs.iter().enumerate() {
        for (gene_idx, &og_idx) in og_vec.iter().enumerate() {
            if og_idx >= 0 {
                let og_idx_usize = og_idx as usize;
                if og_idx_usize < num_orthogroups {
                    orthogroups[og_idx_usize].push((genome_idx, gene_idx));
                }
            }
        }
    }
    orthogroups
}

pub fn build_shared_matrix(ogs: &[Vec<i32>]) -> Vec<Vec<Vec<i32>>> {
    let num_genomes = ogs.len();
    let genome_sets: Vec<HashSet<i32>> = ogs
        .iter()
        .map(|arr| arr.iter().filter(|&&id| id != -1).cloned().collect())
        .collect();

    // Compute the upper triangle (including diagonal) in parallel.
    // Each entry is (i, j, sorted_intersection).
    let pairs: Vec<(usize, usize, Vec<i32>)> = (0..num_genomes)
        .into_par_iter()
        .flat_map_iter(|i| {
            let genome_sets = &genome_sets;
            (i..num_genomes).map(move |j| {
                let mut intersection: Vec<i32> = if i == j {
                    genome_sets[i].iter().cloned().collect()
                } else {
                    genome_sets[i]
                        .intersection(&genome_sets[j])
                        .cloned()
                        .collect()
                };
                intersection.sort_unstable();
                (i, j, intersection)
            })
        })
        .collect();

    // Fill the matrix from the computed pairs.
    let mut matrix = vec![vec![Vec::new(); num_genomes]; num_genomes];
    for (i, j, intersection) in pairs {
        if i == j {
            matrix[i][j] = intersection;
        } else {
            matrix[j][i] = intersection.clone();
            matrix[i][j] = intersection;
        }
    }
    matrix
}

pub fn align_windows(
    og_windows: Vec<((usize, usize), Vec<usize>)>,
) -> Vec<((usize, usize), Vec<Option<usize>>)> {
    if og_windows.is_empty() {
        return Vec::new();
    }

    let mut max_prefix_len = 0;
    let mut focal_gene_offsets = Vec::with_capacity(og_windows.len());

    // Get the indices of each focal gene in the Vec
    for (focal_gene, genes) in og_windows.iter() {
        let pos = genes.iter().position(|&x| x == focal_gene.1).unwrap_or(0);
        focal_gene_offsets.push(pos);
        if pos > max_prefix_len {
            max_prefix_len = pos;
        }
    }

    // Get the window with max total len after aligning the focal genes
    let mut max_total_len = 0;
    for (i, (_, genes)) in og_windows.iter().enumerate() {
        let total_len = (max_prefix_len - focal_gene_offsets[i]) + genes.len();
        if total_len > max_total_len {
            max_total_len = total_len;
        }
    }

    // Padding
    let mut aligned = Vec::with_capacity(og_windows.len());
    for ((focal_gene, genes), offset) in og_windows.into_iter().zip(focal_gene_offsets) {
        let front_pad_size = max_prefix_len - offset;
        let mut aligned_row = Vec::with_capacity(max_total_len);
        for _ in 0..front_pad_size {
            aligned_row.push(None);
        }
        aligned_row.extend(genes.into_iter().map(Some));

        while aligned_row.len() < max_total_len {
            aligned_row.push(None);
        }
        aligned.push((focal_gene, aligned_row));
    }
    aligned
}

pub fn get_window<F>(
    seqid_vec: &[i16],
    gene_idx: usize,
    window_size: usize,
    is_circular: bool,
    og_is_valid: F,
    buffer: &mut Vec<usize>,
) where
    F: Fn(usize) -> bool,
{
    if is_circular {
        get_window_circular(seqid_vec, gene_idx, window_size, og_is_valid, buffer);
    } else {
        get_window_linear(seqid_vec, gene_idx, window_size, og_is_valid, buffer);
    }
}

pub fn calculate_synteny_ratio(win_a: &[i32], win_b: &[i32], edge_adjusted: bool) -> f64 {
    let len_a = win_a.len();
    let len_b = win_b.len();

    if len_a == 0 || len_b == 0 {
        return 0.0;
    }

    let mut matches = 0;
    let (mut i, mut j) = (0, 0);

    // Two-pointer walk on pre-sorted slices
    while i < len_a && j < len_b {
        if win_a[i] == win_b[j] {
            matches += 1;
            i += 1;
            j += 1;
        } else if win_a[i] < win_b[j] {
            i += 1;
        } else {
            j += 1;
        }
    }

    // When at least one gene is at a contig edge its window is truncated through
    // missing data, not absent synteny.  Normalise by the smaller window so the
    // question becomes "of what IS visible, how much matches?" rather than
    // penalising the edge gene for the unavailable context.
    let denom = if edge_adjusted {
        std::cmp::min(len_a, len_b)
    } else {
        std::cmp::max(len_a, len_b)
    };

    matches as f64 / denom as f64
}

/// Count the number of genes on the same contig as `gene_idx` within `half_win`
/// steps on each side, **without** any OG-validity filter.
///
/// This mirrors the scan pattern of `get_window_linear` / `get_window_circular`
/// but counts all same-seqid neighbours regardless of OG assignment.  The
/// result is compared against `2 * half_win` to determine whether a gene sits
/// at a contig edge (result < 2 * half_win) and therefore has a structurally
/// truncated synteny window.
pub fn count_raw_contig_neighbors(
    seqid_vec: &[i16],
    gene_idx: usize,
    half_win: usize,
    is_circular: bool,
) -> usize {
    if half_win == 0 {
        return 0;
    }
    if is_circular {
        count_raw_contig_neighbors_circular(seqid_vec, gene_idx, half_win)
    } else {
        count_raw_contig_neighbors_linear(seqid_vec, gene_idx, half_win)
    }
}

fn count_raw_contig_neighbors_linear(
    seqid_vec: &[i16],
    gene_idx: usize,
    half_win: usize,
) -> usize {
    let focal_seqid = seqid_vec[gene_idx];
    let n = seqid_vec.len();

    let mut left = 0usize;
    let mut i = gene_idx;
    while i > 0 && left < half_win {
        i -= 1;
        if seqid_vec[i] == focal_seqid {
            left += 1;
        }
    }

    let mut right = 0usize;
    let mut j = gene_idx;
    while j + 1 < n && right < half_win {
        j += 1;
        if seqid_vec[j] == focal_seqid {
            right += 1;
        }
    }

    left + right
}

fn count_raw_contig_neighbors_circular(
    seqid_vec: &[i16],
    gene_idx: usize,
    half_win: usize,
) -> usize {
    let focal_seqid = seqid_vec[gene_idx];
    let n = seqid_vec.len();
    if n == 0 {
        return 0;
    }
    let max_neighbors = n - 1;

    let mut left = 0usize;
    let mut i = gene_idx;
    for _ in 0..n {
        if left >= half_win {
            break;
        }
        i = (i + n - 1) % n;
        if i == gene_idx {
            break;
        }
        if seqid_vec[i] == focal_seqid {
            left += 1;
        }
    }

    let r_limit = half_win.min(max_neighbors - left);
    let mut right = 0usize;
    let mut j = gene_idx;
    for _ in 0..n {
        if right >= r_limit {
            break;
        }
        j = (j + 1) % n;
        if j == gene_idx {
            break;
        }
        if seqid_vec[j] == focal_seqid {
            right += 1;
        }
    }

    left + right
}

pub fn cluster_genes(
    pairs: Vec<((usize, usize), (usize, usize))>,
    all_genes: &[(usize, usize)],
) -> Vec<Vec<(usize, usize)>> {
    let n = all_genes.len();
    let gene_to_id: HashMap<(usize, usize), usize> =
        all_genes.iter().enumerate().map(|(i, &c)| (c, i)).collect();

    // data[i] < 0 => Root, value is -(rank + 1)
    let mut dsu = vec![-1; n];

    for (u, v) in pairs {
        if let (Some(&u_id), Some(&v_id)) = (gene_to_id.get(&u), gene_to_id.get(&v)) {
            let root_u = find_dsu(&mut dsu, u_id);
            let root_v = find_dsu(&mut dsu, v_id);

            if root_u != root_v {
                // dsu[root] is negative.
                // If dsu[root_u] is -1 and dsu[root_v] is -2:
                // -1 > -2 is true, but -2 is the deeper tree.
                if dsu[root_u] > dsu[root_v] {
                    // root_v is deeper, attach u to v
                    dsu[root_u] = root_v as i32;
                } else if dsu[root_u] < dsu[root_v] {
                    // root_u is deeper, attach v to u
                    dsu[root_v] = root_u as i32;
                } else {
                    // Ranks are equal, attach u to v and increment v's rank
                    dsu[root_u] = root_v as i32;
                    dsu[root_v] -= 1; // Rank becomes more negative
                }
            }
        }
    }

    // Grouping remains the same, but now the root IDs will be consistent
    let mut clusters: HashMap<usize, Vec<(usize, usize)>> = HashMap::with_capacity(n);
    for i in 0..n {
        let r = find_dsu(&mut dsu, i);
        clusters.entry(r).or_default().push(all_genes[i]);
    }

    let mut result: Vec<Vec<(usize, usize)>> = clusters.into_values().collect();

    // Crucial for matching Python output exactly
    for c in &mut result {
        c.sort_unstable();
    }
    result.sort_unstable_by(|a, b| a[0].cmp(&b[0]));

    result
}

fn get_window_linear<F>(
    seqid_vec: &[i16],
    gene_idx: usize,
    window_size: usize,
    og_is_valid: F,
    buffer: &mut Vec<usize>,
) where
    F: Fn(usize) -> bool,
{
    buffer.clear();
    let half_win = window_size / 2;
    let focal_seqid = seqid_vec[gene_idx];

    // Look Left: Find up to half_win valid indices before gene_idx

    let mut left_indices = Vec::with_capacity(half_win);
    let (mut i, mut left_count) = (gene_idx, 0);
    while i > 0 && left_count < half_win {
        i -= 1;
        if seqid_vec[i] == focal_seqid && og_is_valid(i) {
            left_indices.push(i);
            left_count += 1;
        }
    }
    // Since we scanned backwards, reverse to keep ascending order
    buffer.extend(left_indices.into_iter().rev());

    // Look Right: Find up to half_win valid indices after gene_idx
    let (mut j, mut right_count) = (gene_idx, 0);
    while j < seqid_vec.len() - 1 && right_count < half_win {
        j += 1;
        if seqid_vec[j] == focal_seqid && og_is_valid(j) {
            buffer.push(j);
            right_count += 1;
        }
    }
}

fn get_window_circular<F>(
    seqid_vec: &[i16],
    gene_idx: usize,
    window_size: usize,
    og_is_valid: F,
    buffer: &mut Vec<usize>,
) where
    F: Fn(usize) -> bool,
{
    buffer.clear();
    let half_win = window_size / 2;
    let focal_seqid = seqid_vec[gene_idx];

    let n_genes = seqid_vec.len();
    if n_genes == 0 {
        return;
    }
    let max_neighbors = std::cmp::min(window_size, n_genes - 1);

    // Look Left (Counter-clockwise)
    let mut left_indices = Vec::with_capacity(half_win);
    let (mut i, mut left_count) = (gene_idx, 0);

    for _ in 0..n_genes {
        if left_count >= half_win {
            break;
        }
        i = (i + n_genes - 1) % n_genes; // Circular decrement
        if i == gene_idx {
            break;
        }

        if seqid_vec[i] == focal_seqid && og_is_valid(i) {
            left_indices.push(i);
            left_count += 1;
        }
    }
    buffer.extend(left_indices.into_iter().rev());

    // Look Right (Clockwise)
    let (mut j, mut right_count) = (gene_idx, 0);
    let r_limit = std::cmp::min(half_win, max_neighbors - left_count);

    for _ in 0..n_genes {
        if right_count >= r_limit {
            break;
        }
        j = (j + 1) % n_genes; // Circular increment
        if j == gene_idx {
            break;
        }

        if seqid_vec[j] == focal_seqid && og_is_valid(j) {
            buffer.push(j);
            right_count += 1;
        }
    }
}

fn find_dsu(dsu: &mut [i32], mut i: usize) -> usize {
    let mut root = i;
    while dsu[root] >= 0 {
        root = dsu[root] as usize;
    }
    while dsu[i] >= 0 {
        let n = dsu[i] as usize;
        dsu[i] = root as i32;
        i = n;
    }
    root
}
