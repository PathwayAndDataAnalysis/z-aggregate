# z-aggregate

[![Python](https://img.shields.io/badge/Python-3.12%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-LGPL_v2.1-blue.svg)](LICENSE)
[![Dependency Manager](https://img.shields.io/badge/packaging-uv-purple)](https://github.com/astral-sh/uv)

## Abstract

**Motivation:** Resource-efficient algorithms for assessing transcriptional
factor activities in single-cell transcriptomics are a pressing need. Such
algorithms can help us understand the underlying cellular mechanisms behind the
observed RNA differences.

**Results:** We present a new statistical method for predicting transcription
factor activities from transcriptomic profiles using prior knowledge of target
genes. It aggregates the standardized expression of a transcription factor’s
known target genes into a cell-level activity score using the direction of
regulation and the strength of the target-gene signals. Compared to
alternatives, the method has high predictive power, is faster to compute, and
is memory efficient, making it suitable for analyzing large single-cell RNA
profiles.

**Availability:** A Python implementation of the method is available at
[https://github.com/PathwayAndDataAnalysis/z-aggregate](https://github.com/PathwayAndDataAnalysis/z-aggregate).

## Installation

Clone the repository:

```bash
git clone https://github.com/PathwayAndDataAnalysis/z-aggregate
cd z-aggregate
```

Install `uv` if it is not already available:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

On Windows PowerShell:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Install the project dependencies:

```bash
uv sync
```

Build the project:

```bash
uv build
```


## Quick Start

Run `z-aggregate` with an expression dataset, a prior network, and an output
directory:

```bash
uv run z-aggregate \
  --dataset ./dataset/example.h5ad \
  --prior-type collectri \
  --output ./results \
  --default-preprocess
```

A fuller example is:

```bash
uv run z-aggregate \
  --dataset ./dataset/TianKampmann2021_CRISPRi.h5ad \
  --prior-type collectri \
  --output ./results \
  --default-preprocess \
  --weight-type UNIFORM \
  --output-format both \
  --verbose
```

The dataset above can be downloaded from scPerturb:

```bash
mkdir -p dataset
wget "https://zenodo.org/records/13350497/files/TianKampmann2021_CRISPRi.h5ad?download=1" \
  -O dataset/TianKampmann2021_CRISPRi.h5ad
```

## Command-Line Options

| Option | Required | Default | Description |
| --- | --- | --- | --- |
| `-ds`, `--dataset` | Yes | - | Path to the expression dataset. Supported formats are `.h5ad`, `.csv`, `.tsv`, and `.txt`. |
| `-p`, `--prior-type` | Yes | - | Prior network to use. Use a named prior such as `causalpath`, `collectri`, `dorothea`, or `ensemble`, or provide a custom file path. |
| `-o`, `--output` | Yes | - | Directory where output files will be written. |
| `--min-targets` | No | `5` | Minimum number of target genes required for a transcription factor to be included. |
| `--default-preprocess` | No | Enabled | Explicitly select adaptive preprocessing. This mode is already used when no preprocessing flag is supplied. |
| `--no-preprocess` | No | Disabled | Skip preprocessing when input data are already quality controlled, normalized, and transformed. |
| `--custom-preprocess` | No | Disabled | Apply preprocessing with explicitly supplied QC thresholds. Requires `--min-genes`, `--min-cells`, and `--max-mt-pct`. |
| `--min-genes` | With `--custom-preprocess` | - | Minimum number of genes required per cell during fixed-threshold preprocessing. |
| `--min-cells` | With `--custom-preprocess` | - | Minimum number of cells required per gene during fixed-threshold preprocessing. |
| `--max-mt-pct` | With `--custom-preprocess` | - | Maximum mitochondrial read percentage allowed during fixed-threshold preprocessing. |
| `--weight-type` | No | `UNIFORM` | Edge-weighting strategy: `UNIFORM`, `CORRELATION`, `SPECIFICITY`, `NONZERORATE`, or `EXISTING`. Values are case-sensitive. |
| `--output-format` | No | `both` | Output format: `tsv`, `csv`, `parquet`, `h5ad`, `both`, or `all`. `both` writes TSV and H5AD; `all` writes every format. |
| `-v`, `--verbose` | No | Disabled | Print more detailed log messages. |

## Input Data

### Expression Dataset

The expression dataset is passed with `--dataset`.

Supported formats:

- `.h5ad`: AnnData object.
- `.csv`: comma-separated matrix.
- `.tsv` or `.txt`: tab-separated matrix.

For text files, the first column should contain cell or sample identifiers, and
the remaining columns should be genes. The matrix should be organized as
observations by genes.

### Prior Network

The prior network is passed with `--prior-type`.

You may use a named prior network:

- `causalpath`
- `collectri`
- `dorothea`
- `ensemble`

You may also provide a path to a custom `.csv`, `.tsv`, or `.txt` file.

A prior network must contain transcription factor-target relationships. The
standard columns are:

| Column | Meaning |
| --- | --- |
| `source` | Transcription factor or regulator. |
| `interaction` | Direction of regulation. Positive values indicate activation; negative values indicate inhibition. |
| `target` | Target gene. |
| `weight` | Optional edge-weight magnitude. Used when `--weight-type EXISTING` is selected. |

Common alternative column names such as `tf`, `regulator`, `gene`,
`target_gene`, `mor`, `mode`, `direction`, `effect`, and `sign` are also
accepted.

Interaction values may be numeric, or may use `upregulates-expression`,
`downregulates-expression`, `upregulates`, and `downregulates`. Values are
reduced to their sign, and zero or unrecognized values are discarded.

## Preprocessing

Adaptive preprocessing is the default, including when none of the three
preprocessing mode flags is supplied. `--default-preprocess` simply makes that
choice explicit. It runs the following pipeline on a copy of the loaded data:

1. Convert cell and gene names to strings, strip surrounding whitespace, and
   make duplicate gene names unique. Duplicate cell and gene names are also
   made unique when the input file is loaded.
2. Compute `min_genes = floor(0.01 × n_genes)` from the original shape and
   remove cells expressing fewer genes than this threshold.
3. Compute `min_cells = floor(0.001 × n_cells)` from the original shape and
   remove genes detected in fewer cells than this threshold.
4. Mark mitochondrial genes using a case-insensitive `MT-` prefix and compute
   each cell's mitochondrial count percentage.
5. Set the mitochondrial cutoff to `median + 3 × MAD`, where MAD uses the
   normal-consistency scale factor, then clamp the cutoff to the range 10–25%.
   Keep cells whose mitochondrial percentage is strictly below the cutoff.
6. Normalize each cell to a total count of 10,000 and apply `log1p`.

The CLI does not scale genes to unit variance after this pipeline.

To choose fixed QC thresholds instead, use `--custom-preprocess` and provide all
three required values:

```bash
uv run z-aggregate \
  --dataset ./dataset/example.h5ad \
  --prior-type collectri \
  --output ./results \
  --custom-preprocess \
  --min-genes 1000 \
  --min-cells 10 \
  --max-mt-pct 20
```

Custom preprocessing uses the supplied filtering and mitochondrial thresholds,
then performs the same total-count normalization and `log1p` transformation.
All three threshold options must be provided together and cannot be combined
with `--default-preprocess` or `--no-preprocess`.

Use `--no-preprocess` when the dataset has already been quality controlled,
normalized, and transformed. This avoids filtering, mitochondrial QC,
renormalization, and another log transformation.

## Weighting Strategies

Choose the edge-weighting method with `--weight-type`. CLI values are uppercase
and case-sensitive. Regulatory direction is stored in `interaction` as `-1` or
`+1`, while `weight` is a non-negative magnitude; z-aggregate multiplies the
two when constructing its signed network.

| Value | Description |
| --- | --- |
| `UNIFORM` | Assigns magnitude 1 to every edge and preserves the prior interaction sign. |
| `CORRELATION` | Uses absolute Spearman TF–target correlation as the magnitude and replaces the interaction sign with the sign of a nonzero correlation. Missing, undefined, or zero correlations receive magnitude 0. |
| `SPECIFICITY` | Uses `1 / number of TFs regulating the target` as the magnitude and preserves the prior interaction sign. |
| `NONZERORATE` | Uses the fraction of cells in which the target's processed expression is greater than zero and preserves the prior interaction sign. |
| `EXISTING` | Uses the absolute value of the prior's `weight` column and preserves the interaction sign. If the column is absent, it falls back to uniform magnitudes; missing entries use magnitude 1. |

Edges with magnitude 0 are removed before scoring.

## Output Files

Output files are written to the directory given by `--output`.

For table output, `z-aggregate` writes:

- `<dataset>_z-aggregate_<prior>_<WEIGHT>.<format>` for activity scores
- `<dataset>_z-aggregate_<prior>_<WEIGHT>_pvalues.<format>` for p-values

For AnnData output, it writes:

- `<dataset>_z-aggregate_<prior>_<WEIGHT>_results.h5ad`

The AnnData output contains the activity scores in `.obsm["z-aggregate_scores"]`
and p-values in `.obsm["z-aggregate_pvalues"]`.

## Reproducing Paper Results

Instructions for reproducing the paper results are provided in
[reproduce/README.md](reproduce/README.md).

The main reproduction notebooks are:

- [scRNA-seq reproduction guide](<reproduce/Reproduce scRNASeq Results/README.md>)
- [simulated reproduction guide](<reproduce/Reproduce Simulated Results/README.md>)
