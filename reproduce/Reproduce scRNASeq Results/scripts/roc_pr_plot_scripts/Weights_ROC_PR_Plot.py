"""Compare z-aggregate weight strategies using a fixed prior."""

from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd
import roc_pr_common as common
from tqdm.auto import tqdm
from utility_functions import load_adata_files_with_params

warnings.filterwarnings("ignore", category=FutureWarning)

ANALYSIS_DIR = Path(__file__).resolve().parents[2]

DATASET_DIR = ANALYSIS_DIR / "scRNASeq"
BASE_RESULT_DIR = ANALYSIS_DIR / "scores"
OUTPUT_PLOT_DIR = ANALYSIS_DIR / "results" / "Weights_ROC_plots"

PRIOR_TYPE = "causalpath"
RUN_TAG = f"{PRIOR_TYPE}_weight-strategies"

METHOD_STYLES = {
    "z-aggregate_UNIFORM": {
        "label": "uniform",
        "color": "#0072B2",
    },
    "z-aggregate_CORRELATION": {
        "label": "correlation",
        "color": "#009E73",
    },
    "z-aggregate_SPECIFICITY": {
        "label": "specificity",
        "color": "#E69F00",
    },
    "z-aggregate_NONZERORATE": {
        "label": "nonzero-rate",
        "color": "#CC79A7",
    },
}

MIN_POS_CELLS = 2
MIN_NEG_CELLS = 2
SAVE_FORMAT = "svg"  # png or svg
DPI = 300


# Score loading
def load_method_scores(
    dataset_name: str,
    common_tfs: list[str],
) -> dict[str, pd.DataFrame]:
    output_dir = BASE_RESULT_DIR / dataset_name
    if not output_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {output_dir}")

    score_prefix = f"{dataset_name}_z-aggregate_{PRIOR_TYPE}_"
    score_files = [
        path
        for path in output_dir.iterdir()
        if path.suffix == ".parquet"
        and path.name.startswith(score_prefix)
        and "pvalue" not in path.name.lower()
    ]

    method_scores: dict[str, pd.DataFrame] = {}

    for path in score_files:
        weight_type = path.stem.replace(score_prefix, "", 1)
        method_key = f"z-aggregate_{weight_type}"

        if method_key not in METHOD_STYLES:
            continue

        method_scores[method_key] = common.load_score_table(path, common_tfs)

    missing = [method for method in METHOD_STYLES if method not in method_scores]
    if missing:
        raise ValueError(
            f"Missing required method score files for {dataset_name}: {missing}"
        )

    return method_scores


# TF evaluation
def evaluate_tf(
    tf: str,
    dataset_name: str,
    is_activation: bool,
    condition_clean: pd.Series,
    control_cells: pd.Index,
    method_scores: dict[str, pd.DataFrame],
    dataset_plot_dir: Path,
) -> list[dict]:
    rows: list[dict] = []

    try:
        marked_cells = pd.Index(condition_clean.index[condition_clean == tf])
        if len(marked_cells) < MIN_POS_CELLS:
            return rows

        selected_cells = marked_cells.append(control_cells)

        shared_cells = common.get_shared_cells(
            tf=tf,
            selected_cells=selected_cells,
            method_scores=method_scores,
            required_methods=METHOD_STYLES,
        )

        if shared_cells.empty:
            return rows

        y_true = shared_cells.isin(marked_cells).astype(int)

        if (y_true == 1).sum() < MIN_POS_CELLS or (y_true == 0).sum() < MIN_NEG_CELLS:
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

        for method_name, metrics in curves_by_method.items():
            method_style = METHOD_STYLES[method_name]

            rows.append(
                {
                    "Dataset": dataset_name,
                    "TF": tf,
                    "Method": method_name,
                    "Method_Label": method_style["label"],
                    "N_Pos": metrics["n_pos"],
                    "N_Neg": metrics["n_neg"],
                    "ROC_AUC": metrics["roc_auc"],
                    "PR_AUC": metrics["ap"],
                    "AP_Baseline": metrics["baseline_ap"],
                    "AP_Lift": metrics["ap_lift"],
                }
            )

    except Exception as err:
        print(f"[ERROR] {dataset_name} | {tf}: {err}")

    return rows


# Main
def main() -> None:
    OUTPUT_PLOT_DIR.mkdir(parents=True, exist_ok=True)

    adata_files_with_params = load_adata_files_with_params(PRIOR_TYPE)
    all_rows: list[dict] = []

    for dataset_name, params in adata_files_with_params.items():
        dataset_path = DATASET_DIR / f"{dataset_name}.h5ad"
        if not dataset_path.is_file():
            print(f"Skipping {dataset_name}: missing {dataset_path}")
            continue

        is_activation = params["is_activation"]
        common_tfs = params["common_perturbed_tfs"]

        print(f"\n{'=' * 60}\nProcessing {dataset_name}\n{'=' * 60}")

        adata = common.load_processed_adata(dataset_name, DATASET_DIR)
        condition_clean = adata.obs["condition_clean"]
        control_cells = pd.Index(condition_clean.index[condition_clean == "control"])

        method_scores = load_method_scores(
            dataset_name=dataset_name,
            common_tfs=common_tfs,
        )

        print("Loaded methods:", sorted(method_scores.keys()))

        dataset_plot_dir = OUTPUT_PLOT_DIR / dataset_name
        common.make_plot_directories(dataset_plot_dir)

        for tf in tqdm(common_tfs, desc=f"Plotting TF curves for {dataset_name}"):
            all_rows.extend(
                evaluate_tf(
                    tf=tf,
                    dataset_name=dataset_name,
                    is_activation=is_activation,
                    condition_clean=condition_clean,
                    control_cells=control_cells,
                    method_scores=method_scores,
                    dataset_plot_dir=dataset_plot_dir,
                )
            )

    if all_rows:
        out_file = OUTPUT_PLOT_DIR / f"tf_curve_metrics_{RUN_TAG}.tsv"
        pd.DataFrame(all_rows).to_csv(out_file, sep="\t", index=False)
        print(f"\nSaved summary metrics: {out_file}")

    print(f"\nPlots saved under: {OUTPUT_PLOT_DIR}")


if __name__ == "__main__":
    main()
