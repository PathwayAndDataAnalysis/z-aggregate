# Reproduce scRNASeq Results

This folder contains the workflow for reproducing the scPerturb single-cell
benchmark results from the paper. It compares z-aggregate with VIPER, ULM, and
zscore, and also evaluates prior-network and edge-weighting choices.

## Folder Contents

```text
Reproduce scRNASeq Results/
  scRNASeq/        input .h5ad datasets
  scores/         generated activity-score parquet files
  common_tfs/     common TF lists for prior-network comparisons
  results/        generated statistics, plots, and supplementary tables
  scripts/        helper scripts called by the notebooks
```

The notebooks discover input datasets from:

```text
scRNASeq/*.h5ad
```

## Recommended Workflow

### 1. Generate Activity Scores

Open and run:

```text
run_methods.ipynb
```

This notebook reads datasets from `scRNASeq/` and writes activity scores to
`scores/<dataset>/`.

It generates:

- z-aggregate scores with multiple prior networks.
- z-aggregate scores with multiple weighting strategies.
- VIPER, ULM, and zscore activity scores through Decoupler.


### 2. Run Statistical Tests

Open and run:

```text
mwu_delongs.ipynb
```

This notebook runs the Mann-Whitney U and top-two DeLong analyses. Outputs are
written to:

- `results/Methods_MWU-Delongs/`
- `results/Priors_MWU-Delongs/`
- `results/Weights_MWU-Delongs/`

### 3. Generate ROC and Precision-Recall Plots

Open and run:

```text
roc_pr_plots.ipynb
```

This notebook generates ROC and precision-recall plots. Outputs are written to:

- `results/Methods_ROC_plots/`
- `results/Priors_ROC_plots/`
- `results/Weights_ROC_plots/`

## Optional Outputs

After running the main analysis notebooks, you may also run:

- `generate_venn_diagrams.ipynb`: creates Venn diagrams from significant MWU
  result files.
- `create_supplementary_file.ipynb`: creates supplementary comparison tables
  from the DeLong results.

## Notes

- The workflow can take substantial time because it processes large AnnData
  files and computes several score matrices.
- The `common_tfs/` files are used to make prior-network comparisons on matched
  transcription factors.
