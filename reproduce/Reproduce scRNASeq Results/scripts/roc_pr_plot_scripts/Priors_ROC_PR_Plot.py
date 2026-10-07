"""Compare z-aggregate priors using TFs shared across the prior tables."""

from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd
import roc_pr_common as common
import scanpy as sc
from tqdm.auto import tqdm
from utility_functions import (
    get_single_perturbation,
    load_adata_files_with_params,
    preprocess_adata,
)

warnings.filterwarnings("ignore", category=FutureWarning)

ANALYSIS_DIR = Path(__file__).resolve().parents[2]


# Config
DATASET_DIR = ANALYSIS_DIR / "scRNASeq"
SCORES_DIR = ANALYSIS_DIR / "scores"
OUTPUT_PLOT_DIR = ANALYSIS_DIR / "results" / "Priors_ROC_plots"

PRIOR_TYPES = [
    "causalpath",
    "collectri",
    "dorothea",
    "ensemble",
]

WEIGHT_TYPE = "UNIFORM"
RUN_TAG = f"prior-knowledge_{WEIGHT_TYPE}_matchedTFs"

METHOD_STYLES = {
    f"z-aggregate_causalpath_{WEIGHT_TYPE}": {
        "label": "CausalPath",
        "color": "tab:green",
    },
    f"z-aggregate_collectri_{WEIGHT_TYPE}": {
        "label": "CollecTRI",
        "color": "tab:orange",
    },
    f"z-aggregate_dorothea_{WEIGHT_TYPE}": {
        "label": "DoRothEA",
        "color": "tab:blue",
    },
    f"z-aggregate_ensemble_{WEIGHT_TYPE}": {
        "label": "Ensemble",
        "color": "tab:red",
    },
}

MIN_POS_CELLS = 5
MIN_CONTROL_CELLS = 5
SAVE_FORMAT = "svg"
DPI = 300


# Score loading
def method_to_prior(method: str) -> str:
    return method.replace("z-aggregate_", "").replace(f"_{WEIGHT_TYPE}", "")


def get_common_datasets(params_by_prior: dict) -> list[str]:
    dataset_sets = [set(params_by_prior[prior]) for prior in PRIOR_TYPES]
    return sorted(set.intersection(*dataset_sets))


def get_matched_tfs(dataset_name: str, params_by_prior: dict) -> list[str]:
    tf_sets = [
        set(params_by_prior[prior][dataset_name]["common_perturbed_tfs"])
        for prior in PRIOR_TYPES
    ]
    return sorted(set.intersection(*tf_sets))


def load_scores(dataset_name: str, matched_tfs: list[str]) -> dict[str, pd.DataFrame]:
    score_dir = SCORES_DIR / dataset_name

    if not score_dir.exists():
        raise FileNotFoundError(f"Missing score directory: {score_dir}")

    method_scores: dict[str, pd.DataFrame] = {}

    for prior in PRIOR_TYPES:
        method = f"z-aggregate_{prior}_{WEIGHT_TYPE}"
        score_path = score_dir / f"{dataset_name}_{method}.parquet"

        if not score_path.exists():
            raise FileNotFoundError(f"Missing score file: {score_path}")

        method_scores[method] = common.load_score_table(score_path, matched_tfs)
        print(f"   Loaded {method}: {method_scores[method].shape}")

    missing = [method for method in METHOD_STYLES if method not in method_scores]
    if missing:
        raise ValueError(f"Missing score files for {dataset_name}: {missing}")

    return method_scores


def load_processed_adata(dataset_name: str) -> sc.AnnData:
    dataset_path = DATASET_DIR / f"{dataset_name}.h5ad"

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    adata = sc.read_h5ad(dataset_path)
    adata = preprocess_adata(adata, do_scale=False)
    adata.obs_names = common.clean_index(adata.obs_names)

    if "perturbation" not in adata.obs.columns:
        raise ValueError(f"'perturbation' column missing for {dataset_name}")

    condition = adata.obs["perturbation"].apply(get_single_perturbation)
    condition = condition.astype("string").str.strip()
    condition = condition.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})

    adata = adata[condition.notna()].copy()
    adata.obs["condition_clean"] = (
        condition.loc[adata.obs_names].astype(str).str.strip()
    )

    return adata


# TF evaluation
def evaluate_tf(
    tf: str,
    dataset_name: str,
    is_activation: bool,
    condition: pd.Series,
    control_cells: pd.Index,
    method_scores: dict[str, pd.DataFrame],
    dataset_plot_dir: Path,
) -> list[dict]:
    rows: list[dict] = []

    try:
        perturbed_cells = pd.Index(condition.index[condition == tf])
        if len(perturbed_cells) < MIN_POS_CELLS:
            return rows

        selected_cells = pd.Index(
            perturbed_cells.tolist() + control_cells.tolist()
        ).drop_duplicates()

        shared_cells = common.get_shared_cells(
            tf=tf,
            selected_cells=selected_cells,
            method_scores=method_scores,
            required_methods=METHOD_STYLES,
        )

        if shared_cells.empty:
            return rows

        y_true = shared_cells.isin(perturbed_cells).astype(int)

        n_pos = int((y_true == 1).sum())
        n_control = int((y_true == 0).sum())

        if n_pos < MIN_POS_CELLS or n_control < MIN_CONTROL_CELLS:
            return rows

        curves_by_method = common.compute_tf_curves(
            tf=tf,
            shared_cells=shared_cells,
            y_true=y_true,
            method_scores=method_scores,
            required_methods=METHOD_STYLES,
            is_activation=is_activation,
        )

        if len(curves_by_method) != len(METHOD_STYLES):
            return rows

        common.plot_tf_curves(
            dataset_name=dataset_name,
            tf=tf,
            curves_by_method=curves_by_method,
            dataset_plot_dir=dataset_plot_dir,
            styles=METHOD_STYLES,
            save_format=SAVE_FORMAT,
            dpi=DPI,
        )

        for method, metrics in curves_by_method.items():
            method_style = METHOD_STYLES[method]
            rows.append(
                {
                    "Dataset": dataset_name,
                    "TF": tf,
                    "Prior_Type": method_to_prior(method),
                    "Method": method,
                    "Method_Label": method_style["label"],
                    "N_Pos": metrics["n_pos"],
                    "N_Control": metrics["n_neg"],
                    "ROC_AUC": metrics["roc_auc"],
                    "PR_AUC": metrics["ap"],
                    "AP_Baseline": metrics["baseline_ap"],
                    "AP_Lift": metrics["ap_lift"],
                }
            )

    except Exception as error:
        print(f"[ERROR] {dataset_name} | {tf}: {error}")

    return rows


# Main
def main() -> None:
    OUTPUT_PLOT_DIR.mkdir(parents=True, exist_ok=True)

    params_by_prior = {
        prior: load_adata_files_with_params(prior_type=prior) for prior in PRIOR_TYPES
    }
    dataset_names = get_common_datasets(params_by_prior)

    print(f"Datasets shared across priors: {len(dataset_names)}")

    all_rows: list[dict] = []

    for dataset_name in dataset_names:
        dataset_path = DATASET_DIR / f"{dataset_name}.h5ad"
        if not dataset_path.is_file():
            print(f"Skipping {dataset_name}: missing {dataset_path}")
            continue

        matched_tfs = get_matched_tfs(dataset_name, params_by_prior)

        if not matched_tfs:
            print(f"Skipping {dataset_name}: no matched TFs across priors")
            continue

        print(f"\n{'=' * 60}")
        print(f"Processing {dataset_name}")
        print(f"Matched TFs: {len(matched_tfs)}")
        print(f"{'=' * 60}")

        ref_params = params_by_prior[PRIOR_TYPES[0]][dataset_name]
        is_activation = bool(ref_params["is_activation"])

        adata = load_processed_adata(dataset_name)
        condition = adata.obs["condition_clean"]
        control_cells = pd.Index(condition.index[condition == "control"])

        if len(control_cells) < MIN_CONTROL_CELLS:
            print(
                f"Skipping {dataset_name}: fewer than {MIN_CONTROL_CELLS} control cells"
            )
            continue

        method_scores = load_scores(dataset_name, matched_tfs)

        dataset_plot_dir = OUTPUT_PLOT_DIR / dataset_name
        common.make_plot_directories(dataset_plot_dir)

        for tf in tqdm(matched_tfs, desc=f"Plotting prior curves for {dataset_name}"):
            all_rows.extend(
                evaluate_tf(
                    tf=tf,
                    dataset_name=dataset_name,
                    is_activation=is_activation,
                    condition=condition,
                    control_cells=control_cells,
                    method_scores=method_scores,
                    dataset_plot_dir=dataset_plot_dir,
                )
            )

    if all_rows:
        output_file = OUTPUT_PLOT_DIR / f"tf_curve_metrics_{RUN_TAG}.tsv"
        pd.DataFrame(all_rows).to_csv(output_file, sep="\t", index=False)
        print(f"\nSaved summary metrics: {output_file}")

    print(f"\nPlots saved under: {OUTPUT_PLOT_DIR}")


if __name__ == "__main__":
    main()
