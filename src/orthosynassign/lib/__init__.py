"""
Library module for orthoSynAssign containing file parsing utilities.
"""

from __future__ import annotations

from .engine import get_synteny_engine, get_visualize_engine, vectorize_genomes
from .flank_score import build_training_table, flank_completeness, flank_score, jaccard
from .gene import Gene, Genome
from .hog import FlankRecord, build_flank_window, read_hog_table
from .orthogroup import Orthogroup
from .parsers import BedParser, read_og_table, write_og_table
from .rs import FlankEngine, calculate_directional_synteny_ratio, calculate_synteny_ratio, get_window_split

__all__ = [
    get_synteny_engine,
    get_visualize_engine,
    vectorize_genomes,
    Gene,
    Genome,
    Orthogroup,
    BedParser,
    read_og_table,
    write_og_table,
    calculate_synteny_ratio,
    calculate_directional_synteny_ratio,
    get_window_split,
    FlankEngine,
    FlankRecord,
    build_flank_window,
    read_hog_table,
    jaccard,
    flank_score,
    flank_completeness,
    build_training_table,
]
