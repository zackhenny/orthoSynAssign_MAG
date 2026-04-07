from __future__ import annotations

from typing import TYPE_CHECKING

from .rs import SyntenyEngine, VisualizeEngine

if TYPE_CHECKING:
    from .gene import Genome
    from .orthogroup import Orthogroup


def get_synteny_engine(genomes: list[Genome], orthogroups: list[Orthogroup]) -> SyntenyEngine:
    """
    Converts biological objects into integer vectors and initializes the Rust SyntenyEngine.
    """

    og_list_all, seqid_list_all, is_circular_all = _data_vectorization(genomes, orthogroups)

    # Initialize the Rust Engine
    engine = SyntenyEngine(len(orthogroups), og_list_all, seqid_list_all, is_circular_all)

    return engine


def get_visualize_engine(genomes: list[Genome], ogs: list[Orthogroup], sogs: list[Orthogroup]) -> VisualizeEngine:
    """
    Converts biological objects into integer vectors and initializes the Rust VisualizeEngine.
    """
    og_list_all, seqid_list_all, is_circular_all = _data_vectorization(genomes, ogs)

    # Process Synteny Orthogroup (SOG) data
    sogs_data: list[list[tuple[int, int]]] = []
    genome_to_int = {genome: idx for idx, genome in enumerate(genomes)}

    for sog in sogs:
        unique_genes = sorted(
            list(set(gene.representative for gene in sog._genes)), key=lambda x: (genome_to_int[x.genome], x.index)
        )

        # Create a list of coordinates for this specific SOG
        current_sog_coords = [(genome_to_int[gene.genome], gene.index) for gene in unique_genes]
        sogs_data.append(current_sog_coords)

    # Initialize the Rust Engine
    engine = VisualizeEngine(sogs_data, og_list_all, seqid_list_all, is_circular_all)

    return engine


def _data_vectorization(
    genomes: list[Genome], orthogroups: list[Orthogroup]
) -> tuple[list[list[int]], list[list[int]], list[bool]]:
    # Map OG IDs to integers for the entire project
    og_str_to_int = {og.id: i for i, og in enumerate(orthogroups)}

    og_list_all: list[list[int]] = []
    seqid_list_all: list[list[int]] = []
    is_circular_all: list[bool] = []

    for genome in genomes:
        # Local map for scaffold IDs per genome (e.g., 'Chr1' -> 0)
        distinct_seqids = dict.fromkeys(gene.seqid for gene in genome._genes)
        seqid_map = {seqid: i for i, seqid in enumerate(distinct_seqids)}

        og_list_genome: list[int] = [og_str_to_int.get(gene.og.id, -1) if gene.og else -1 for gene in genome._genes]
        seqid_list_genome: list[int] = [seqid_map[gene.seqid] for gene in genome._genes]

        # Pass lists to the Rust constructor; Rust converts them to Vec internally.
        og_list_all.append(og_list_genome)
        seqid_list_all.append(seqid_list_genome)
        is_circular_all.append(genome.is_circular)

    return og_list_all, seqid_list_all, is_circular_all
