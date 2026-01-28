## Reproduce the results from the paper

We have two folders here:
1. `scRNASeq_results_reproduce`: This folder contains all the code and data required to reproduce the scPerturb Datasets (scRNA-Seq) results from the paper.
2. `simulated_results_reproduce`: This folder contains all the code and data required to reproduce the simulation results from the paper.


Each folder contains a `notebook.ipynb` file that provides step-by-step instructions to reproduce the results. Please follow the instructions in the respective notebooks to replicate the findings presented in the paper.

### For scRNA-Seq Results Reproduction:
1. You first have to download the datasets from `scPerturbDatasets` using the code provided in the `notebook.ipynb` file located in the `scRNASeq_results_reproduce` folder.
2. After downloading the datasets, follow the steps in the notebook to run z-aggregate, ulm, and viper on the datasets and generate activity scores.
3. Finally, use the provided code in the notebook to benchmark the methods and reproduce the figures from the paper.

As the datasets are large, we have also provided pre-calculated activity scores for z-aggregate, ulm, and viper. You can download these pre-calculated scores using the code provided in the notebook to save time.