# Reproduce Simulated Results

This folder contains the workflow for reproducing the simulated benchmark
results from the paper. The workflow generates simulated expression data,
simulated regulatory priors, and ground-truth transcription factor states, then
compares activity-inference methods with ROC curves.

## Folder Contents

```text
Reproduce Simulated Results/
  simulated_data_notebook.ipynb   main simulation notebook
  simulated_data_generator.py    simulation generator
  simulated_data/                generated expression, prior, and ground truth files
  roc_plots/                     generated simulation figures and metrics
```

## Recommended Workflow

Open and run:

```text
simulated_data_notebook.ipynb
```

Use this folder as the notebook working directory:

```text
reproduce/Reproduce Simulated Results/
```

## Generated Data

The generator writes the following files to `simulated_data/`:

- `simulated_scRNASeq.tsv`
- `simulated_prior_network.tsv`
- `simulated_noisy_prior_network.tsv`
- `simulated_ground_truth.tsv`

These files are regenerated as the notebook runs different simulation
settings.

## Generated Figures

The notebook writes simulation figures to `roc_plots/`:

- `roc_plots/experiment1/`: prior-noise experiment.
- `roc_plots/experiment2/`: missing-value experiment.
- `roc_plots/experiment3/`: gene-propensity variation experiment.

The combined results are saved to `roc_plots/roc_metrics.tsv`, with one row per
method and activation/inhibition task for each simulation setting. Rows include
ROC AUC, evaluated cell and TF counts, positive and negative cell–TF pair counts,
and the simulation parameters. These are pooled cell–TF
metrics, not individual-TF or replicate-level estimates.

Each simulation run updates its plot rows in the TSV while retaining other
runs, so rerunning an experiment does not duplicate its rows. Archived figures
under `roc_plots/old_experiment/` are retained separately from the current results.
