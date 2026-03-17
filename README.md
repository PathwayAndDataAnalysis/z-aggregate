<div align="center">

# 🧬 z-aggregate


[![Python](https://img.shields.io/badge/Python-3.12%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-LGPL_v2.1-blue.svg)](LICENSE)
[![Code Style](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Dependency Manager](https://img.shields.io/badge/packaging-uv-purple)](https://github.com/astral-sh/uv)

</div>

---


***

# z-aggregate

A fast and efficient statistical method for predicting transcription factor activities from transcriptomic profiles using prior knowledge of target genes.
## Getting Started

Follow these instructions to set up the environment and run the application on your local machine.

### 1. Clone the Repository
Open your terminal and clone the project repository:

```bash
git clone https://github.com/PathwayAndDataAnalysis/z-aggregate
cd z-aggregate
```

### 2. Install `uv` (if not installed)
This project uses **[uv](https://github.com/astral-sh/uv)** for extremely fast package management and execution.

**macOS / Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows (PowerShell):**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

> *Note: After installation, you may need to restart your terminal.*

### 3. Install Dependencies & Build
You do not need to manually create virtual environments. Run the following command to sync the project and install all required packages (Scanpy, NumPy, Pandas, etc.) into a managed environment:

```bash
uv sync
```

---

## How to Run?

To run the application, use `uv run`.

### Basic Syntax
```bash
uv run z-aggregate -ds <path_to_data> -p <path_to_network> -o <output_folder>
```

### Example Usage
```bash
uv run z-aggregate \
  --dataset ./data/sc_counts.h5ad \
  --prior-type causalpath-priors \
  --output ./results \
  --weight-type Uniform \
  --verbose
```

### Example Run
To run z-aggregate we need a single-cell expression dataset and a prior network file (this is already provided in the repository in `data/causal-priors.tsv`).

To test the application, let's download a sample dataset from scPerturb [here](https://zenodo.org/records/7041849/files/TianKampmann2021_CRISPRi.h5ad?download=1) (e.g., `TianKampmann2021_CRISPRi.h5ad`), and then run the following command:

```bash
!wget "https://zenodo.org/records/7041849/files/TianKampmann2021_CRISPRi.h5ad?download=1" -O data/TianKampmann2021_CRISPRi.h5ad
```

```bash
uv run z-aggregate \
  --dataset ./data/TianKampmann2021_CRISPRi.h5ad \
  --prior-type causalpath-priors \
  --output ./results \
  --weight-type Uniform \
  --verbose
```

## Parameter Reference

| Flag | Long Flag | Type | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| **-ds** | `--dataset` | Path | **Required** | Path to expression data. Supports `.h5ad`, `.csv`, `.tsv`, `.txt`. |
| **-p** | `--prior-type` | Str | **Required/Provided** | Type of prior network to use. Options: `causalpath-priors`, `collectri`, `dorothea`, `ensemble-priors`. |
| **-o** | `--output` | Path | **Required** | Directory where results will be saved. |
| **-v** | `--verbose` | Flag | `False` | Enable detailed logging output. |
| | `--min-targets` | Int | `5` | Minimum number of target genes required per TF to be included. |
| | `--weight-type` | Enum | `Uniform` | Weighting strategy. See **Weight Types** below. |
| | `--output-format` | Str | `both` | Format of output. Options: `tsv`, `h5ad`, `both`. |
| **Preprocessing Options** | | | | |
| | `--preprocess` | Flag | `True` | Enable standard QC and LogNormal preprocessing. |
| | `--no-preprocess` | Flag | - | Disable preprocessing (use if input is already normalized). |
| | `--min-genes` | Int | `1000` | Minimum genes per cell (QC). |
| | `--min-cells` | Int | `10` | Minimum cells per gene (QC). |
| | `--max-mt-pct` | Float | `20.0` | Maximum mitochondrial percentage allowed (QC). |

### Weight Types
You can adjust how the algorithm weights the edges between TFs and Target Genes using `--weight-type`:

*   `Uniform`: No weights.
*   `Correlation`: Weights are scaled by the Spearman Correlation between TF and Target Genes. The interaction/direction (i.e. `upregulates-expression` or `downregulates-expression`) in priors is replaced by the sign of the correlation.
*   `Specificity`: Weights are scaled by `1 / (Number of TFs regulating that gene)`.
*   `NonzeroRate`: Weights are scaled by the detection rate of the target gene.
*   `Existing`: Uses the weight column provided in the input prior file.

---

## Input File Formats

### 1. Expression Data (`--dataset`)
*   **Formats:** `.h5ad` (Anndata), `.csv` (comma-separated), `.tsv` (tab-separated). While using `csv` or `tsv`, ensure the that it is in the Cells x Genes format, which is rows as Cells and columns as Genes.
*   **Structure:** If text-based, rows should be **Cells** and columns **Genes**, or standard Anndata structure.

### 2. Prior Network (`--prior-type`)
*  **Options:** `causalpath-priors`, `collectri`, `dorothea`, `ensemble-priors`, or a custom file path.
A CSV or TSV file containing TF-Target interactions.
<!-- *   **Required Columns:** `source` (TF), `interaction` (mode), `target` (Gene).
*   **Optional:** `weight`.
*   **Example:**
    ```csv
    source  interaction  target
    TF_A  upregulates-expression Gene_X 
    TF_B  downregulates-expression  Gene_Y
    ```
    | Note: This is a tab-separated file.  -->

---

## Output Files

The tool generates the following files in the specified output directory:

1.  **`<dataset-file-name>_pathway-commons_z_agg_scores.tsv`**: Matrix of inferred TF activities (Cells x TFs).
2.  **`<dataset-file-name>_pathway-commons_z_agg_pvalues.tsv`**: Significance values for the activities.
3.  **`<dataset-file-name>_pathway-commons_z_agg_results.h5ad`** (Optional): A copy of the input Anndata object containing the results in `obsm`.


---

## Reproduce the results from the paper
1. Please refer to the [reproduce/README.md](reproduce/README.md) file for detailed instructions on how to replicate the results presented in the original publication. Or directly go to the notebook [reproduce/scRNASeq_results_reproduce/notebook.ipynb](reproduce/scRNASeq_results_reproduce/notebook.ipynb) to reproduce the scRNA-Seq results.

2. To reproduce the simulated results, please to the notebook [reproduce/simulated_results_reproduce/notebook.ipynb](reproduce/simulated_results_reproduce/notebook.ipynb).