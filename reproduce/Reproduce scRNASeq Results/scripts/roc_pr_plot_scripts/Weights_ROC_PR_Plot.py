from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from tqdm.auto import tqdm
from utility_functions import (
    get_single_perturbation,
    load_adata_files_with_params,
    preprocess_adata,
    set_publication_style,
)

warnings.filterwarnings("ignore", category=FutureWarning)

DATASET_DIR = Path("/hpcstor6/scratch01/k/kisan.thapa001/z_agg_data/scRNASeq")
BASE_RESULT_DIR = Path("/hpcstor6/scratch01/k/kisan.thapa001/z_agg_data/scores")
OUTPUT_PLOT_DIR = Path("/hpcstor6/scratch01/k/kisan.thapa001/z_agg_data/Weights_ROC_PR_Plots")

PRIOR_TYPE = "causalpath"
RUN_TAG = f"{PRIOR_TYPE}_weight-strategies"

METHOD_STYLES = {
    "z-aggregate_UNIFORM": {
        "label": "Uniform",
        "color": "#0072B2",
    },
    "z-aggregate_CORRELATION": {
        "label": "Correlation",
        "color": "#009E73",
    },
    "z-aggregate_SPECIFICITY": {
        "label": "Specificity",
        "color": "#E69F00",
    },
    "z-aggregate_NONZERORATE": {
        "label": "Non-zero rate",
        "color": "#CC79A7",
    },
}

REQUIRED_METHODS = list(METHOD_STYLES)

MIN_POS_CELLS = 5
MIN_NEG_CELLS = 5
SAVE_FORMAT = "svg"  # png or svg
DPI = 300

def sanitize_filename(text: str) -> str:
    text = str(text)
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text


def get_method_style(method_name: str) -> dict:
    return METHOD_STYLES.get(
        method_name,
        {
            "label": method_name,
            "color": None,
        },
    )


def load_method_scores(
    dataset_name: str,
    common_tfs: List[str],
    required_methods: List[str],
) -> Dict[str, pd.DataFrame]:
    output_dir = BASE_RESULT_DIR / dataset_name
    if not output_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {output_dir}")

    score_prefix = f"{dataset_name}_z-aggregate_{PRIOR_TYPE}_"
    score_files = [
        path for path in output_dir.iterdir() if path.suffix == ".parquet" and path.name.startswith(score_prefix)
    ]

    method_scores: Dict[str, pd.DataFrame] = {}

    for path in score_files:
        weight_type = path.stem.replace(score_prefix, "", 1)
        method_key = f"z-aggregate_{weight_type}"

        if method_key not in required_methods:
            continue

        df_scores = pd.read_parquet(path)
        df_scores.index = pd.Index(df_scores.index.astype(str)).str.strip()
        df_scores.columns = pd.Index(df_scores.columns.astype(str)).str.strip()
        df_scores = df_scores.reindex(columns=common_tfs, fill_value=np.nan)

        method_scores[method_key] = df_scores

    missing = [method for method in required_methods if method not in method_scores]
    if missing:
        raise ValueError(f"Missing required method score files for {dataset_name}: {missing}")

    return method_scores


def load_processed_adata(dataset_name: str) -> sc.AnnData:
    dataset_path = DATASET_DIR / f"{dataset_name}.h5ad"
    adata = sc.read_h5ad(dataset_path)
    adata = preprocess_adata(adata, do_scale=False)

    if "perturbation" not in adata.obs.columns:
        raise ValueError(f"'perturbation' column not found in adata.obs for {dataset_name}")

    adata.obs["condition_clean"] = adata.obs["perturbation"].apply(get_single_perturbation)
    adata = adata[adata.obs["condition_clean"].notna()].copy()
    adata.obs["condition_clean"] = adata.obs["condition_clean"].astype(str).str.strip()
    adata.obs_names = pd.Index(adata.obs_names.astype(str)).str.strip()

    return adata


def compute_curves(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    roc_auc = roc_auc_score(y_true, y_score)
    fpr, tpr, _ = roc_curve(y_true, y_score)

    ap = average_precision_score(y_true, y_score)
    precision, recall, _ = precision_recall_curve(y_true, y_score)

    baseline_ap = float(y_true.mean())
    ap_lift = ap - baseline_ap

    return {
        "roc_auc": float(roc_auc),
        "fpr": fpr,
        "tpr": tpr,
        "ap": float(ap),
        "precision": precision,
        "recall": recall,
        "baseline_ap": baseline_ap,
        "ap_lift": float(ap_lift),
        "n_pos": int((y_true == 1).sum()),
        "n_neg": int((y_true == 0).sum()),
    }


def make_plot_directories(dataset_plot_dir: Path) -> None:
    (dataset_plot_dir / "roc").mkdir(parents=True, exist_ok=True)
    (dataset_plot_dir / "pr").mkdir(parents=True, exist_ok=True)


def plot_roc_curve_for_tf(
    dataset_name: str,
    tf: str,
    curves_by_method: Dict[str, dict],
    out_path: Path,
) -> None:
    plt.figure(figsize=(8, 8), facecolor="white")

    sorted_methods = sorted(
        curves_by_method.items(),
        key=lambda item: item[1]["roc_auc"],
        reverse=True,
    )

    for method_name, payload in sorted_methods:
        method_style = get_method_style(method_name)

        plt.plot(
            payload["fpr"],
            payload["tpr"],
            linewidth=2,
            color=method_style["color"],
            label=f"{method_style['label']} ({payload['roc_auc']:.4f})",
        )

    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1.5, color="grey")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("False Positive Rate", fontsize=14)
    plt.ylabel("True Positive Rate", fontsize=14)
    plt.title(f"{dataset_name} | {tf}", fontsize=16)
    plt.legend(loc="lower right", frameon=True, fontsize=20, borderpad=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close()


def plot_pr_curve_for_tf(
    dataset_name: str,
    tf: str,
    curves_by_method: Dict[str, dict],
    out_path: Path,
) -> None:
    plt.figure(figsize=(8, 8), facecolor="white")

    baseline_ap = float(np.median([payload["baseline_ap"] for payload in curves_by_method.values()]))

    sorted_methods = sorted(
        curves_by_method.items(),
        key=lambda item: item[1]["ap"],
        reverse=True,
    )

    for method_name, payload in sorted_methods:
        method_style = get_method_style(method_name)

        plt.plot(
            payload["recall"],
            payload["precision"],
            linewidth=2,
            color=method_style["color"],
            label=(f"{method_style['label']} (AP={payload['ap']:.4f}, lift={payload['ap_lift']:.4f})"),
        )

    plt.axhline(
        baseline_ap,
        linestyle="--",
        linewidth=1.5,
        color="grey",
        label=f"Baseline prevalence={baseline_ap:.4f}",
    )
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("Recall", fontsize=14)
    plt.ylabel("Precision", fontsize=14)
    plt.title(f"{dataset_name} | {tf}", fontsize=16)
    plt.legend(loc="best", frameon=True, fontsize=20, borderpad=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close()


def get_shared_cells(
    tf: str,
    selected_cells: pd.Index,
    method_scores: Dict[str, pd.DataFrame],
    required_methods: List[str],
) -> pd.Index:
    shared_cells = selected_cells.copy()

    for method_name in required_methods:
        score_df = method_scores[method_name]

        if tf not in score_df.columns:
            return pd.Index([])

        method_valid_cells = score_df.index[
            score_df.index.isin(selected_cells) & pd.to_numeric(score_df[tf], errors="coerce").notna()
        ]

        shared_cells = shared_cells.intersection(method_valid_cells)

    return shared_cells


def evaluate_tf(
    tf: str,
    dataset_name: str,
    is_activation: bool,
    condition_clean: pd.Series,
    control_cells: pd.Index,
    method_scores: Dict[str, pd.DataFrame],
    dataset_plot_dir: Path,
) -> List[dict]:
    rows: List[dict] = []

    try:
        marked_cells = pd.Index(condition_clean.index[condition_clean == tf])
        if len(marked_cells) < MIN_POS_CELLS:
            return rows

        selected_cells = marked_cells.append(control_cells)

        shared_cells = get_shared_cells(
            tf=tf,
            selected_cells=selected_cells,
            method_scores=method_scores,
            required_methods=REQUIRED_METHODS,
        )

        if len(shared_cells) == 0:
            return rows

        y_true_series = pd.Series(0, index=shared_cells, dtype=int)
        y_true_series.loc[y_true_series.index.isin(marked_cells)] = 1
        y_true = y_true_series.to_numpy(dtype=int)

        if (y_true == 1).sum() < MIN_POS_CELLS or (y_true == 0).sum() < MIN_NEG_CELLS:
            return rows

        curves_by_method = {}

        for method_name in REQUIRED_METHODS:
            y_score = pd.to_numeric(
                method_scores[method_name].loc[shared_cells, tf],
                errors="coerce",
            ).to_numpy(dtype=float)

            if not np.isfinite(y_score).all():
                return rows

            if not is_activation:
                y_score = -y_score

            curves_by_method[method_name] = compute_curves(y_true, y_score)

        if len(curves_by_method) != len(REQUIRED_METHODS):
            return rows

        tf_file = sanitize_filename(tf)
        roc_out = dataset_plot_dir / "roc" / f"{tf_file}_roc.{SAVE_FORMAT}"
        pr_out = dataset_plot_dir / "pr" / f"{tf_file}_pr.{SAVE_FORMAT}"

        plot_roc_curve_for_tf(
            dataset_name=dataset_name,
            tf=tf,
            curves_by_method=curves_by_method,
            out_path=roc_out,
        )

        plot_pr_curve_for_tf(
            dataset_name=dataset_name,
            tf=tf,
            curves_by_method=curves_by_method,
            out_path=pr_out,
        )

        for method_name, metrics in curves_by_method.items():
            method_style = get_method_style(method_name)

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


def main() -> None:
    set_publication_style()
    OUTPUT_PLOT_DIR.mkdir(parents=True, exist_ok=True)

    adata_files_with_params = load_adata_files_with_params(PRIOR_TYPE)
    dataset_names = list(adata_files_with_params.keys())
    all_rows: List[dict] = []

    for dataset_name in dataset_names:
        params = adata_files_with_params[dataset_name]
        is_activation = params["is_activation"]
        common_tfs = params["common_perturbed_tfs"]

        print(f"\n{'=' * 60}\nProcessing {dataset_name}\n{'=' * 60}")

        adata = load_processed_adata(dataset_name)
        condition_clean = adata.obs["condition_clean"]
        control_cells = pd.Index(condition_clean.index[condition_clean == "control"])

        method_scores = load_method_scores(
            dataset_name=dataset_name,
            common_tfs=common_tfs,
            required_methods=REQUIRED_METHODS,
        )

        print("Loaded methods:", sorted(method_scores.keys()))

        dataset_plot_dir = OUTPUT_PLOT_DIR / dataset_name
        make_plot_directories(dataset_plot_dir)

        for tf in tqdm(common_tfs, desc=f"Plotting TF curves for {dataset_name}"):
            tf_rows = evaluate_tf(
                tf=tf,
                dataset_name=dataset_name,
                is_activation=is_activation,
                condition_clean=condition_clean,
                control_cells=control_cells,
                method_scores=method_scores,
                dataset_plot_dir=dataset_plot_dir,
            )

            if tf_rows:
                all_rows.extend(tf_rows)

    if all_rows:
        summary_df = pd.DataFrame(all_rows)
        out_file = OUTPUT_PLOT_DIR / f"tf_curve_metrics_{RUN_TAG}.tsv"
        summary_df.to_csv(out_file, sep="\t", index=False)
        print(f"\nSaved summary metrics: {out_file}")

    print(f"\nPlots saved under: {OUTPUT_PLOT_DIR}")


if __name__ == "__main__":
    main()
