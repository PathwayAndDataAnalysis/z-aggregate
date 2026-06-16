# Reproduce Simulated Results

This folder contains the workflow for reproducing the simulated benchmark
results from the paper. The workflow generates simulated expression data,
simulated regulatory priors, and ground-truth transcription factor states, then
compares activity-inference methods with ROC curves.

## Folder Contents

```text
Reproduce Simulated Results/
  notebook.ipynb                 main simulation notebook
  simulated_data_generator.py    script used to create simulated data
  simulated_data/                generated expression, prior, and ground truth files
  illustrations/                 generated simulation figures
```

## Recommended Workflow

Open and run:

```text
notebook.ipynb
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

The notebook writes simulation figures to `illustrations/`:

- `illustrations/exp1/`: prior-noise experiment.
- `illustrations/exp2/`: missing-value experiment.
- `illustrations/exp3/`: gene-propensity variation experiment.
