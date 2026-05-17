mod engine;
mod flank;
mod utils;

use crate::engine::*;
use crate::flank::*;
use crate::utils::*;
use pyo3::prelude::*;

#[pyfunction]
#[pyo3(name = "get_window")]
#[pyo3(text_signature = "(og_masked_array, seq_array, gene_idx, window_size)")]
pub fn get_window_py(
    og_mask_vec: Vec<bool>,
    seqid_vec: Vec<i16>,
    gene_idx: usize,
    window_size: usize,
    is_circular: bool,
) -> PyResult<Vec<usize>> {
    let mut result_vec = Vec::with_capacity(window_size);
    get_window(
        &seqid_vec,
        gene_idx,
        window_size,
        is_circular,
        |i| og_mask_vec[i],
        &mut result_vec,
    );
    Ok(result_vec)
}

#[pyfunction]
#[pyo3(name = "get_window_split")]
#[pyo3(text_signature = "(og_masked_array, seq_array, gene_idx, window_size, is_circular)")]
pub fn get_window_split_py(
    og_mask_vec: Vec<bool>,
    seqid_vec: Vec<i16>,
    gene_idx: usize,
    window_size: usize,
    is_circular: bool,
) -> PyResult<(Vec<usize>, Vec<usize>)> {
    let result = get_window_split(
        &seqid_vec,
        gene_idx,
        window_size,
        is_circular,
        |i| og_mask_vec[i],
    );
    Ok(result)
}

#[pyfunction]
#[pyo3(name = "calculate_synteny_ratio")]
#[pyo3(text_signature = "(win_a, win_b)")]
pub fn calculate_synteny_ratio_py(mut win_a: Vec<i32>, mut win_b: Vec<i32>) -> PyResult<f64> {
    win_a.sort_unstable();
    win_b.sort_unstable();
    Ok(calculate_synteny_ratio(&win_a, &win_b, false))
}

#[pyfunction]
#[pyo3(name = "calculate_directional_synteny_ratio")]
#[pyo3(text_signature = "(left_a, right_a, left_b, right_b, left_edge_a, right_edge_a, left_edge_b, right_edge_b)")]
pub fn calculate_directional_synteny_ratio_py(
    mut left_a: Vec<i32>,
    mut right_a: Vec<i32>,
    mut left_b: Vec<i32>,
    mut right_b: Vec<i32>,
    left_edge_a: bool,
    right_edge_a: bool,
    left_edge_b: bool,
    right_edge_b: bool,
) -> PyResult<f64> {
    left_a.sort_unstable();
    right_a.sort_unstable();
    left_b.sort_unstable();
    right_b.sort_unstable();
    Ok(calculate_directional_synteny_ratio(
        &left_a,
        &right_a,
        &left_b,
        &right_b,
        left_edge_a,
        right_edge_a,
        left_edge_b,
        right_edge_b,
    ))
}

#[pymodule]
fn rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<SyntenyEngine>()?;
    m.add_class::<VisualizeEngine>()?;
    m.add_class::<FlankEngine>()?;
    m.add_function(wrap_pyfunction!(get_window_py, m)?)?;
    m.add_function(wrap_pyfunction!(get_window_split_py, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_synteny_ratio_py, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_directional_synteny_ratio_py, m)?)?;
    Ok(())
}
