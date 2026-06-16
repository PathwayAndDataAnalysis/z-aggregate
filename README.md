# z-aggregate

[![Python](https://img.shields.io/badge/Python-3.12%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-LGPL_v2.1-blue.svg)](LICENSE)
[![Dependency Manager](https://img.shields.io/badge/packaging-uv-purple)](https://github.com/astral-sh/uv)

`z-aggregate` is a command-line tool for estimating transcription factor
activities from transcriptomic profiles. It combines an expression matrix with a
prior regulatory network and reports transcription factor activity scores and
associated p-values for each cell or sample.

The method is intended for single-cell or bulk transcriptomic data where rows are
observations and columns are genes.

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

## Quick Start

Run `z-aggregate` with an expression dataset, a prior network, and an output
directory:

```bash
uv run z-aggregate \
  --dataset ./dataset/example.h5ad \
  --prior-type collectri \
  --output ./results
```

A fuller example is:

```bash
uv run z-aggregate \
  --dataset ./dataset/TianKampmann2021_CRISPRi.h5ad \
  --prior-type collectri \
  --output ./results \
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
| `-p`, `--prior-type` | Yes | - | Prior network to use. Use a named prior such as `causalpath-priors`, `collectri`, `dorothea`, or `ensemble-priors`, or provide a custom file path. |
| `-o`, `--output` | Yes | - | Directory where output files will be written. |
| `--min-targets` | No | `5` | Minimum number of target genes required for a transcription factor to be included. |
| `--preprocess` | No | Enabled | Apply standard preprocessing before running the method. |
| `--no-preprocess` | No | - | Skip preprocessing. Use this when the input data are already filtered and normalized. |
| `--min-genes` | No | `1000` | Minimum number of genes required per cell during preprocessing. |
| `--min-cells` | No | `10` | Minimum number of cells required per gene during preprocessing. |
| `--max-mt-pct` | No | `20.0` | Maximum mitochondrial read percentage allowed during preprocessing. |
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

- `causalpath-priors`
- `collectri`
- `dorothea`
- `ensemble-priors`

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

Preprocessing is enabled by default. It performs:

1. Cell filtering using `--min-genes`.
2. Gene filtering using `--min-cells`.
3. Mitochondrial-content filtering using `--max-mt-pct`.
4. Library-size normalization to a target sum of 10,000.
5. Log transformation.

Use `--no-preprocess` when the dataset has already been quality controlled,
normalized, and transformed.

## Weighting Strategies

Choose the edge-weighting method with `--weight-type`.

| Value | Description |
| --- | --- |
| `Uniform` | Uses the sign of the prior interaction as the edge weight. |
| `Correlation` | Uses Spearman correlation between transcription factor expression and target-gene expression. |
| `Specificity` | Weights each target by `1 / number of TFs regulating that target`. |
| `NonzeroRate` | Weights each target by its detection rate in the dataset. |
| `Existing` | Uses the `weight` column from the prior network, if present. |

## Output Files

Output files are written to the directory given by `--output`.

For table output, `z-aggregate` writes:

- `<dataset>_<prior>_z_agg_scores.<format>`
- `<dataset>_<prior>_z_agg_pvalues.<format>`

For AnnData output, it writes:

- `<dataset>_z_aggregate_results.h5ad`

The AnnData output contains the activity scores in `.obsm["z_aggregate_scores"]`
and p-values in `.obsm["z_aggregate_pvalues"]`.

## Reproducing Paper Results

Instructions for reproducing the paper results are provided in
[reproduce/README.md](reproduce/README.md).

The main reproduction notebooks are:

- [scRNA-seq reproduction guide](<reproduce/Reproduce scRNASeq Results/README.md>)
- [simulated reproduction guide](<reproduce/Reproduce Simulated Results/README.md>)
