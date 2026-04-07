"""
Library module for orthoSynAssign containing file parsing utilities.
"""

from __future__ import annotations

from .engine import get_synteny_engine, get_visualize_engine
from .gene import Gene, Genome
from .orthogroup import Orthogroup
from .parsers import BedParser, read_og_table, write_og_table
from .rs import calculate_synteny_ratio

__all__ = [
    get_synteny_engine,
    get_visualize_engine,
    Gene,
    Genome,
    Orthogroup,
    BedParser,
    read_og_table,
    write_og_table,
    calculate_synteny_ratio,
]
