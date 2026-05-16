[![CI/build and test](https://github.com/stajichlab/orthoSynAssign/actions/workflows/build_and_test.yml/badge.svg?branch=main)](https://github.com/stajichlab/orthoSynAssign/actions/workflows/build_and_test.yml)
[![Python](https://img.shields.io/badge/python-3.9_%7C_3.10_%7C_3.11_%7C_3.12_%7C_3.13-blue?logo=python)](https://github.com/stajichlab/Phyling/actions/workflows/build_and_test.yml)
[![codecov](https://codecov.io/gh/stajichlab/orthoSynAssign/graph/badge.svg?token=mNxGMxekfv)](https://codecov.io/gh/stajichlab/orthoSynAssign)
[![License](https://img.shields.io/github/license/stajichlab/orthoSynAssign?label=license)](https://github.com/stajichlab/orthoSynAssign/blob/main/LICENSE)
[![DOI](https://zenodo.org/badge/1140709253.svg)](https://doi.org/10.5281/zenodo.18762979)

# orthoSynAssign_MAG

> **⚠️ This is a fork.** This repository (`zackhenny/orthoSynAssign_MAG`) is a fork of the original
> [stajichlab/orthoSynAssign](https://github.com/stajichlab/orthoSynAssign) project created by
> **Cheng-Hung Tsai** and **Jason Stajich** at the University of California, Riverside.
> All credit for the original tool, its design, and its core algorithms belongs to them.
> Please cite their work (see [Citation](#citation)) if you use this software.
>
> The changes made in this fork are described in the [Fork Changes](#fork-changes) section below.
> For the canonical version of this tool, visit the original repository at
> <https://github.com/stajichlab/orthoSynAssign>.

## Fork Changes

This fork (`orthoSynAssign_MAG`) extends the original `orthoSynAssign` tool with two sets of
additions aimed at large-scale and MAG (Metagenome-Assembled Genome) analyses, where fragmented
contigs and assembly incompleteness make the fixed `-r` heuristic less reliable and raw Python
throughput a bottleneck.

### 1. Statistical modelling pipeline (`orthosynassign.stats`)

**What was added:** A complete data-driven calibration pipeline that replaces the fixed `-r`
synteny-ratio heuristic with a logistic regression model trained on interior genes with known
split status. The pipeline consists of five sequential steps:

1. `orthosynassign-score` – exports a per-gene flank-score training table.
2. `orthosynassign.stats.permutation` – permutation test confirming that the HOG-neighbourhood
   signal is non-random.
3. `orthosynassign.stats.calibrate` – logistic regression calibration; outputs model coefficients
   and operating thresholds.
4. `orthosynassign.stats.cv` – stratified k-fold cross-validation with ROC/PR curve diagnostics.
5. `orthosynassign.stats.mixed_effects` – mixed-effects logistic regression (via R/lme4) that
   accounts for per-genome random intercepts.
6. `orthosynassign.stats.apply_model` – scores edge genes and adds `split_probability`,
   `split_confidence`, and `rescue_flag` columns to the output table.

**Why:** MAG datasets often contain many fragmented, edge-located genes for which the simple
ratio threshold is too coarse. Replacing it with a calibrated probability model improves
precision on fragmentary assemblies and provides a principled FPR-controlled operating point.

### 2. Rust `FlankEngine` with Rayon parallelism

**What was added:** The flank-score computation (`build_shared_matrix` and `refine_logic`) was
migrated from Python to a new Rust module (`src/flank.rs`) that uses
[Rayon](https://github.com/rayon-rs/rayon) for data-parallel processing. The Python layer now
calls the compiled Rust extension via PyO3.

**Why:** MAG studies routinely involve hundreds of genomes, making the all-vs-all flank matrix
computationally expensive in pure Python. The Rust/Rayon implementation reduces wall-clock time
significantly for large genome sets, making the full calibration pipeline practical at scale.

Ortholog Synteny Assignment Tool - A Python tool to refine orthologous groups using synteny information inferred from genome
annotation files (BED converted from GFF3/GTF). This is a Python re-implementation of the [OrthoRefine], which is written in C++
but had some issues with sample matching and memory usage. This tool is designed to be more efficient, easier to use, and more
flexible for custom analyses. In addition, to ensure optimal processing speed and memory usage, the core synteny analysis is
implemented in [Rust]. We also provide a companion visualization tool, `orthosynassign-vis`, for users to verify the results. This
refined version includes improved memory management and a streamlined workflow for assigning syntenic regions, addressing
limitations identified in the original OrthoRefine implementation. Specifically, it incorporates optimized data structures for
handling genomic ranges and utilizes more efficient algorithms for finding overlaps between ortholog groups and genomic segments.

## Usage

First, install the package following the [instruction](#install) below.

### Refine orthogroups by synteny analysis

`orthosynassign` is the main program for running the analysis. It takes the OrthoFinder-style `orthogroup.tsv` or the `N0.tsv`
under phylogenetic hierarchical orthogroups directory and output the refined orthogroups with synteny information determined using
the genome annotation files.

Most genome annotation files are distributed in GFF3 or GTF formats. However, the high degree of flexibility in the 9th attribute
column often makes it challenging to parse specific protein IDs and match them to entries in an orthogroup file.

To simplify this process, we provide a utility script, misc/gff2bed.bash, which converts GFF3 files into a standardized BED
format. This script ensures that genomic coordinates are correctly linked to the protein IDs used by OrthoFinder. If a gene contains multiple isoforms, the script collapses them into a single entry. In this case, the 4th column of the BED file will contain all associated protein IDs, concatenated using a semicolon (;) as a delimiter.

[!IMPORTANT]
If you choose to prepare your own BED files manually, you must use a semicolon (;) to separate protein IDs for multiple isoforms. orthoSynAssign is specifically programmed to use this delimiter to resolve isoform-related mapping issues automatically.

The `orthogroup.tsv` or `N0.tsv` file from OrthoFinder should be tab-separated with:

- First column: Orthogroup ID (e.g., OG0000001)
- Subsequent columns: Protein IDs for each species (column headers are species names)

Please use `orthosynassign --help` to see all available options and arguments:

```
Required arguments:
  --og_file OG_FILE     Path to OrthoFinder Orthogroups.tsv file
  --bed file [files ...]
                        Path of BED formatted genome annotation files

Options:
  -w, --window WINDOW   Controls how many total genes are considered when determining synteny for a single gene (default: 8)
  -r, --ratio_threshold THRESHOLD
                        Controls how many genes within a window must provide synteny support to classify the genes being compared as syntenous (default: 0.5)
  -o, --output OUTPUT   Output of results (default: Refined_SOGs-[YYYYMMDD-HHMMSS].tsv (UTC timestamp))
  -t, --threads THREADS
                        Number of cpus to use (default: 4)
  -v, --verbose         Enable verbose logging
  -V, --version         show program's version number and exit
  -h, --help            show this help message and exit
```

We provided some example files in directory `example`, which contains three BED annotations and a orthogroup file:

```
FungiDB-68_AfumigatusA1163.bed
FungiDB-68_AfumigatusAf293.bed
FungiDB-68_AnovofumigatusIBT16806.bed
orthogroups.tsv
```

Use the following command to run the refinement process:

```bash
orthosynassign --og_file orthogroups.tsv --bed *.bed -o Refined_SOGs.tsv
```

The refined result will output to `Refined_SOGs.tsv`.

### Visualize the refined orthogroups

`orthosynassign-vis` is a companion visualization script to verify the refined results of `orthosynassign`. It utilizes the
[pyGenomeViz] to plot the orthogroups and their synteny relationships. It takes the original, unrefined `orthogroup.tsv` file along
with the refined orthogroup file to plot a certain set of refined orthogroups using their previous orthogroup IDs as the labels
for each gene in the plot. Please use `orthosynassign-vis --help` to see all available options and arguments:

```
Required arguments:
  --og_file OG_FILE     Path to the original orthogroups.tsv file
  --sog_file SOG_FILE   Path to the refined orthogroups.tsv file
  --bed file [files ...]
                        Path of BED formatted genome annotation files
  --sog SOG [SOG ...]   Plot the SOG of the previous orthosynassign analysis

Options:
  -w, --window WINDOW   The window size applied to the previous orthosynassign analysis (default: 8)
  -o, --output OUTPUT   Output directory (default: visualize_[sog_file])
  -f, --fmt {png,jpg,svg,pdf}
                        Output image format. (default: png)
  -k, --keep_all_genes  Keep genes that are not assigned to any orthogroup
  -v, --verbose         Enable verbose logging
  -V, --version         show program's version number and exit
  -h, --help            show this help message and exit
```

The `example` directory contains another refined orthogroup file - `Refined_SOGs.tsv`, say if we want to verify one of the refined
orthogroup `SOG000039.OG0000040`:

```bash
orthosynassign-vis --og_file orthogroups.tsv --sog_file Refined_SOGs.tsv --bed *.bed --sog SOG000039.OG0000040 -f svg
```

The figure will output to `visualize_Refined_SOGs/SOG000039.OG0000040.svg`. In this figure, the genes of the observed refined
orthogroup are labelled in yellow; genes assigned to the same orthogroup within this given window are labelled in other chromatic
colors; genes with orthologs in other genomes located outside the given window are labelled in gray.

<img src= "misc/SOG000039.OG0000040.svg" alt="A refined orthogroup SOG000039" width="800">

## Statistical Modelling

The `orthosynassign-score` command and the `orthosynassign.stats` sub-package
provide a data-driven calibration pipeline that replaces the fixed `-r` heuristic
with a logistic regression model trained on interior genes with known split status.

### Recommended Execution Sequence

#### 1. Export the flank-score training table

```bash
orthosynassign-score \
    --og_file orthogroups.tsv \
    --hog_file N0.tsv \
    --sog_file Refined_SOGs.tsv \
    --bed *.bed \
    -w 4 \
    -o sog_gene_edge_long.csv
```

This produces `sog_gene_edge_long.csv` with per-gene flank scores and
ground-truth split labels for interior genes.

#### 2. Permutation test (confirm HOG signal is non-random)

```bash
python -m orthosynassign.stats.permutation \
    --table sog_gene_edge_long.csv \
    --n_permutations 1000 \
    --output_dir results/
```

Outputs `permutation_test.png` and `permutation_summary.tsv`.
Exits with code 1 if the HOG-neighbourhood signal is not significant at α = 0.05.

#### 3. Logistic regression calibration

```bash
python -m orthosynassign.stats.calibrate \
    --table sog_gene_edge_long.csv \
    --output_dir results/
```

Outputs `calibration.json` (model coefficients and operating thresholds),
`calibration_roc.png`, and `calibration_pr.png`.

#### 4. Cross-validation / ROC analysis

```bash
python -m orthosynassign.stats.cv \
    --table sog_gene_edge_long.csv \
    --output_dir results/ \
    --cv_folds 5 \
    --max_fpr 0.05
```

Outputs `cv_roc.png`, `cv_pr.png`, and `cv_summary.tsv` with per-fold AUC and
optimal threshold at the specified FPR.

#### 5. Mixed-effects logistic regression (R / lme4)

```bash
python -m orthosynassign.stats.mixed_effects \
    --table sog_gene_edge_long.csv \
    --output_dir results/
```

Calls `Rscript` internally.  Outputs `mixed_effects_results.tsv` (fixed-effect
coefficients), `genome_random_effects.tsv` (per-genome intercepts), and
`mixed_effects_icc.txt` (intra-class correlation).

#### 6. Score edge genes and flag rescues

```bash
python -m orthosynassign.stats.apply_model \
    --table sog_gene_edge_long.csv \
    --calibration results/calibration.json \
    --output sog_gene_edge_scored.csv \
    --threshold_type f1
```

Adds `split_probability`, `split_confidence`, and `rescue_flag` columns to the
edge-gene table.

## Requirements

- [Python] >= 3.9, < 3.14
- [numpy] >= 2.0.0
- [pyGenomeViz] >= 1.6.0

### Statistical Modelling (optional)

The `[stats]` extra installs the Python dependencies for the calibration and
evaluation pipeline:

```bash
pip install 'orthosynassign[stats]'
```

This adds: `scipy`, `scikit-learn`, `numpy`, `matplotlib`, `pandas`.

The mixed-effects logistic regression (Step 4) additionally requires **R** (≥ 4.0)
and the `lme4` package:

```r
install.packages("lme4")
```

`Rscript` must be available on your `PATH`.  See [CRAN](https://cran.r-project.org/)
for R installation instructions.

## Install

### From source

To install **this fork** (with the statistical modelling and Rust FlankEngine additions), clone
from this repository:

```bash
git clone https://github.com/zackhenny/orthoSynAssign_MAG.git
```

To install the **original upstream tool** instead, clone from:

```bash
git clone https://github.com/stajichlab/orthoSynAssign.git
```

Navigate to the project directory and install the package.

```bash
cd orthoSynAssign_MAG   # or orthoSynAssign for the upstream version
pip install .
```

### For developing

Developers should clone the project directly and install the package with dev flag. Please also set up the pre-commit first before
making commit.

```bash
pip install -e ".[dev]"
pre-commit install
```

## Citation

If you use orthoSynAssign in your research, please cite:

[Cheng-Hung Tsai, & Jason Stajich. (2026). orthoSynAssign - a Python tool to refine orthogroups using synteny information [Computer software]](https://doi.org/10.5281/zenodo.18762979)

[Ludwig, J., Mrázek, J. OrthoRefine: automated enhancement of prior ortholog identification via synteny. BMC Bioinformatics 25, 163 (2024)](https://doi.org/10.1186/s12859-024-05786-7)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

[Python]: https://www.python.org/
[OrthoRefine]: https://github.com/jl02142/OrthoRefine
[numpy]: https://numpy.org/
[pyGenomeViz]: https://github.com/moshi4/pyGenomeViz
[Rust]: https://rust-lang.org/
