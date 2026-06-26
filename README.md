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
  --weight-type Uniform \
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
| `--default-preprocess` | No | Enabled | Apply adaptive default preprocessing. This is the default behavior. |
| `--no-preprocess` | No | Disabled | Skip preprocessing when input data are already quality controlled, normalized, and transformed. |
| `--custom-preprocess` | No | Disabled | Apply preprocessing with explicitly supplied QC thresholds. Requires `--min-genes`, `--min-cells`, and `--max-mt-pct`. |
| `--min-genes` | With `--custom-preprocess` | - | Minimum number of genes required per cell during fixed-threshold preprocessing. |
| `--min-cells` | With `--custom-preprocess` | - | Minimum number of cells required per gene during fixed-threshold preprocessing. |
| `--max-mt-pct` | With `--custom-preprocess` | - | Maximum mitochondrial read percentage allowed during fixed-threshold preprocessing. |
| `--weight-type` | No | `Uniform` | Weighting strategy for prior-network edges. |
| `--output-format` | No | `both` | Output format: `tsv`, `csv`, `h5ad`, or `both`. With `both`, table files and an AnnData file are written. |
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
| `weight` | Optional edge weight. Used when `--weight-type Existing` is selected. |

Common alternative column names such as `tf`, `regulator`, `gene`,
`target_gene`, `mor`, `mode`, `direction`, `effect`, and `sign` are also
accepted.

Interaction values may be numeric, or may use terms such as
`upregulates-expression`, `downregulates-expression`, `activation`, and
`inhibition`.

## Preprocessing

Adaptive default preprocessing runs unless `--custom-preprocess` or
`--no-preprocess` is supplied. `--default-preprocess` may be used to state the
default mode explicitly. The adaptive workflow:

1. Cell names and gene names are stripped of surrounding whitespace, and
   duplicate gene names are made unique.
2. Cells are filtered when they express fewer than 1% of the dataset's genes;
   genes are filtered when they are expressed in fewer than 0.1% of cells.
3. Mitochondrial-content filtering uses `median + 3 × MAD`, with the cutoff
   bounded between 10% and 25%.
4. Counts are normalized to a target sum of 10,000 and log transformed.

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

Use `--no-preprocess` when the dataset has already been quality controlled,
normalized, and transformed.

## Weighting Strategies

Choose the edge-weighting method with `--weight-type`.

| Value | Description |
| --- | --- |
| `Uniform` | Assigns a positive weight of 1 to all prior edges. |
| `Correlation` | Uses absolute Spearman correlation between transcription factor expression and target-gene expression. Replaces the existing interaction by the sign of the correlation. |
| `Specificity` | Weights each target by `1 / number of TFs regulating that target`, giving lower weight to broadly regulated targets. |
| `NonzeroRate` | Weights each target by its detection rate in the dataset. |
| `Existing` | Uses the `weight` column from the prior network, if present. |

## Output Files

Output files are written to the directory given by `--output`.

For table output, `z-aggregate` writes:

- `<dataset>_<prior>_z-aggregate_scores.<format>`
- `<dataset>_<prior>_z-aggregate_pvalues.<format>`

For AnnData output, it writes:

- `<dataset>_z-aggregate_results.h5ad`

The AnnData output contains the activity scores in `.obsm["z-aggregate_scores"]`
and p-values in `.obsm["z-aggregate_pvalues"]`.

## Reproducing Paper Results

Instructions for reproducing the paper results are provided in
[reproduce/README.md](reproduce/README.md).

The main reproduction notebooks are:

- [scRNA-seq reproduction guide](<reproduce/Reproduce scRNASeq Results/README.md>)
- [simulated reproduction guide](<reproduce/Reproduce Simulated Results/README.md>)
