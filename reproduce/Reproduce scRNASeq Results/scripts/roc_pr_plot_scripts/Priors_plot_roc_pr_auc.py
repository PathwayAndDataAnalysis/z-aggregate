from __future__ import annotations

import os

# Prevent BLAS/OpenMP over-subscription when using many processes.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import gc
import multiprocessing as mp
import re
import sys
import warnings
from pathlib import Path

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

SCRIPTS_DIR = Path(__file__).resolve().parent.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from utility_functions import (
    preprocess_adata,
    get_single_perturbation,
    set_publication_style,
    load_adata_files_with_params,
)

warnings.filterwarnings("ignore", category=FutureWarning)


# =========================================================
# CONFIG
# =========================================================
ANALYSIS_DIR = SCRIPTS_DIR.parent
DATASET_DIR = ANALYSIS_DIR / "scRNASeq"
SCORES_DIR = ANALYSIS_DIR / "scores"
OUTPUT_PLOT_DIR = ANALYSIS_DIR / "results" / "Priors_ROC_plots"

PRIOR_TYPES = [
    "causalpath-priors",
    "collectri",
    "dorothea",
    "ensemble-priors",
]

WEIGHT_TYPE = "UNIFORM"
RUN_TAG = f"prior-knowledge_{WEIGHT_TYPE}_matchedTFs"
METHODS = [f"z-agg_{prior}_{WEIGHT_TYPE}" for prior in PRIOR_TYPES]

METHOD_LABELS = {
    f"z-agg_causalpath-priors_{WEIGHT_TYPE}": "CausalPath",
    f"z-agg_collectri_{WEIGHT_TYPE}": "CollecTRI",
    f"z-agg_dorothea_{WEIGHT_TYPE}": "DoRothEA",
    f"z-agg_ensemble-priors_{WEIGHT_TYPE}": "Ensemble Priors",
}

METHOD_COLORS = {
    f"z-agg_causalpath-priors_{WEIGHT_TYPE}": "tab:green",
    f"z-agg_collectri_{WEIGHT_TYPE}": "tab:orange",
    f"z-agg_dorothea_{WEIGHT_TYPE}": "tab:blue",
    f"z-agg_ensemble-priors_{WEIGHT_TYPE}": "tab:red",
}

TARGET_DATASETS: list[str] | None = None

MIN_POS_CELLS = 5
MIN_CONTROL_CELLS = 5

SAVE_FORMAT = "svg"
DPI = 300
N_PROCESSES = max(
    1,
    int(os.environ.get("ROC_PR_N_PROCESSES", min(8, os.cpu_count() or 1))),
)


# =========================================================
# WORKER GLOBALS
# =========================================================
_WORKER = {
    "dataset_name": None,
    "is_activation": None,
    "condition": None,
    "control_cells": None,
    "method_scores": None,
    "plot_dir": None,
}


def init_worker(
    dataset_name: str,
    is_activation: bool,
    condition: pd.Series,
    control_cells: pd.Index,
    method_scores: dict[str, pd.DataFrame],
    plot_dir: str,
) -> None:
    _WORKER["dataset_name"] = dataset_name
    _WORKER["is_activation"] = is_activation
    _WORKER["condition"] = condition
    _WORKER["control_cells"] = control_cells
    _WORKER["method_scores"] = method_scores
    _WORKER["plot_dir"] = plot_dir

    set_publication_style()


# =========================================================
# HELPERS
# =========================================================
def clean_index(index) -> pd.Index:
    return pd.Index(index.astype(str)).str.strip()


def sanitize_filename(text: str) -> str:
    text = str(text)
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text


def method_to_prior(method: str) -> str:
    return method.replace("z-agg_", "").replace(f"_{WEIGHT_TYPE}", "")


def load_params_by_prior() -> dict:
    return {prior: load_adata_files_with_params(prior_type=prior) for prior in PRIOR_TYPES}


def get_common_datasets(params_by_prior: dict) -> list[str]:
    dataset_sets = [set(params_by_prior[prior]) for prior in PRIOR_TYPES]
    return sorted(set.intersection(*dataset_sets))


def get_matched_tfs(dataset_name: str, params_by_prior: dict) -> list[str]:
    tf_sets = [set(params_by_prior[prior][dataset_name]["common_perturbed_tfs"]) for prior in PRIOR_TYPES]
    return sorted(set.intersection(*tf_sets))


def load_scores(dataset_name: str, matched_tfs: list[str]) -> dict[str, pd.DataFrame]:
    score_dir = SCORES_DIR / dataset_name

    if not score_dir.exists():
        raise FileNotFoundError(f"Missing score directory: {score_dir}")

    method_scores = {}

    for prior in PRIOR_TYPES:
        method = f"z-agg_{prior}_{WEIGHT_TYPE}"
        score_path = score_dir / f"{dataset_name}_{method}.parquet"

        if not score_path.exists():
            raise FileNotFoundError(f"Missing score file: {score_path}")

        scores = pd.read_parquet(score_path)
        scores.index = clean_index(scores.index)
        scores.columns = clean_index(scores.columns)

        method_scores[method] = scores.reindex(columns=matched_tfs)

        print(f"   Loaded {method}: {method_scores[method].shape}")

    missing = [method for method in METHODS if method not in method_scores]

    if missing:
        raise ValueError(f"Missing score files for {dataset_name}: {missing}")

    return method_scores


def load_processed_adata(dataset_name: str) -> sc.AnnData:
    dataset_path = DATASET_DIR / f"{dataset_name}.h5ad"

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    adata = sc.read_h5ad(dataset_path)
    adata = preprocess_adata(adata, do_scale=False)
    adata.obs_names = clean_index(adata.obs_names)

    if "perturbation" not in adata.obs.columns:
        raise ValueError(f"'perturbation' column missing for {dataset_name}")

    condition = adata.obs["perturbation"].apply(get_single_perturbation)
    condition = condition.astype("string").str.strip()
    condition = condition.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})

    adata = adata[condition.notna()].copy()
    adata.obs["condition_clean"] = condition.loc[adata.obs_names].astype(str).str.strip()

    return adata


def compute_curves(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    roc_auc = roc_auc_score(y_true, y_score)
    fpr, tpr, _ = roc_curve(y_true, y_score)

    pr_auc = average_precision_score(y_true, y_score)
    precision, recall, _ = precision_recall_curve(y_true, y_score)

    baseline = float(y_true.mean())

    return {
        "roc_auc": float(roc_auc),
        "fpr": fpr,
        "tpr": tpr,
        "pr_auc": float(pr_auc),
        "precision": precision,
        "recall": recall,
        "baseline": baseline,
        "lift": float(pr_auc - baseline),
        "n_pos": int((y_true == 1).sum()),
        "n_control": int((y_true == 0).sum()),
    }


# =========================================================
# PLOTTING
# =========================================================
def plot_roc(
    dataset_name: str,
    tf: str,
    curves_by_method: dict[str, dict],
    out_path: Path,
) -> None:
    plt.figure(figsize=(8, 8), facecolor="white")

    sorted_methods = sorted(
        curves_by_method.items(),
        key=lambda item: item[1]["roc_auc"],
        reverse=True,
    )

    for method, metrics in sorted_methods:
        plt.plot(
            metrics["fpr"],
            metrics["tpr"],
            linewidth=2,
            color=METHOD_COLORS.get(method),
            label=f"{METHOD_LABELS.get(method, method)} ({metrics['roc_auc']:.4f})",
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


def plot_pr(
    dataset_name: str,
    tf: str,
    curves_by_method: dict[str, dict],
    out_path: Path,
) -> None:
    plt.figure(figsize=(8, 8), facecolor="white")

    baseline = float(np.median([m["baseline"] for m in curves_by_method.values()]))

    sorted_methods = sorted(
        curves_by_method.items(),
        key=lambda item: item[1]["pr_auc"],
        reverse=True,
    )

    for method, metrics in sorted_methods:
        plt.plot(
            metrics["recall"],
            metrics["precision"],
            linewidth=2,
            color=METHOD_COLORS.get(method),
            label=(f"{METHOD_LABELS.get(method, method)} (AP={metrics['pr_auc']:.4f}, lift={metrics['lift']:.4f})"),
        )

    plt.axhline(
        baseline,
        linestyle="--",
        linewidth=1.5,
        color="grey",
        label=f"Baseline prevalence={baseline:.4f}",
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


# =========================================================
# TF PROCESSING
# =========================================================
def process_tf(tf: str) -> list[dict]:
    dataset_name = _WORKER["dataset_name"]
    is_activation = _WORKER["is_activation"]
    condition = _WORKER["condition"]
    control_cells = _WORKER["control_cells"]
    method_scores = _WORKER["method_scores"]
    plot_dir = Path(_WORKER["plot_dir"])

    rows = []

    try:
        perturbed_cells = pd.Index(condition.index[condition == tf])

        if len(perturbed_cells) < MIN_POS_CELLS:
            return rows

        selected_cells = pd.Index(perturbed_cells.tolist() + control_cells.tolist()).drop_duplicates()

        shared_cells = selected_cells.copy()

        for method in METHODS:
            scores = method_scores[method]

            if tf not in scores.columns:
                return rows

            tf_scores = pd.to_numeric(scores[tf], errors="coerce")
            valid_cells = scores.index[scores.index.isin(selected_cells) & tf_scores.notna()]

            shared_cells = shared_cells.intersection(valid_cells)

        if len(shared_cells) == 0:
            return rows

        y_true_series = pd.Series(0, index=shared_cells, dtype=int)
        y_true_series.loc[y_true_series.index.isin(perturbed_cells)] = 1
        y_true = y_true_series.to_numpy(dtype=int)

        n_pos = int((y_true == 1).sum())
        n_control = int((y_true == 0).sum())

        if n_pos < MIN_POS_CELLS or n_control < MIN_CONTROL_CELLS:
            return rows

        curves_by_method = {}

        for method in METHODS:
            y_score = pd.to_numeric(
                method_scores[method].loc[shared_cells, tf],
                errors="coerce",
            ).to_numpy(dtype=float)

            if not np.isfinite(y_score).all():
                return rows

            # For CRISPRi/inhibition datasets, flip sign so higher score means
            # stronger evidence for the perturbed cells.
            if not is_activation:
                y_score = -y_score

            curves_by_method[method] = compute_curves(y_true, y_score)

        if len(curves_by_method) != len(METHODS):
            return rows

        tf_file = sanitize_filename(tf)

        roc_path = plot_dir / "roc" / f"{tf_file}_roc.{SAVE_FORMAT}"
        pr_path = plot_dir / "pr" / f"{tf_file}_pr.{SAVE_FORMAT}"

        plot_roc(dataset_name, tf, curves_by_method, roc_path)
        plot_pr(dataset_name, tf, curves_by_method, pr_path)

        for method, metrics in curves_by_method.items():
            rows.append(
                {
                    "Dataset": dataset_name,
                    "TF": tf,
                    "Prior_Type": method_to_prior(method),
                    "Method": method,
                    "Method_Label": METHOD_LABELS.get(method, method),
                    "N_Pos": metrics["n_pos"],
                    "N_Control": metrics["n_control"],
                    "ROC_AUC": metrics["roc_auc"],
                    "PR_AUC": metrics["pr_auc"],
                    "AP_Baseline": metrics["baseline"],
                    "AP_Lift": metrics["lift"],
                    "ROC_Plot": str(roc_path),
                    "PR_Plot": str(pr_path),
                }
            )

        return rows

    except Exception as error:
        print(f"[ERROR] {dataset_name} | {tf}: {error}")
        return rows


def get_chunksize(n_items: int, n_processes: int) -> int:
    return max(1, n_items // max(1, n_processes * 4))


# =========================================================
# MAIN
# =========================================================
def main() -> None:
    set_publication_style()
    OUTPUT_PLOT_DIR.mkdir(parents=True, exist_ok=True)

    params_by_prior = load_params_by_prior()
    dataset_names = get_common_datasets(params_by_prior)

    if TARGET_DATASETS is not None:
        target_set = set(TARGET_DATASETS)
        dataset_names = [name for name in dataset_names if name in target_set]

    available_datasets = {path.stem for path in DATASET_DIR.glob("*.h5ad")}
    missing_datasets = [name for name in dataset_names if name not in available_datasets]
    for dataset_name in missing_datasets:
        print(f"Skipping {dataset_name}: dataset file is not available locally")
    dataset_names = [name for name in dataset_names if name in available_datasets]

    n_processes = min(N_PROCESSES, os.cpu_count() or 1)

    print(f"Using {n_processes} worker processes")
    print(f"Datasets shared across priors: {len(dataset_names)}")

    all_rows = []

    for dataset_name in dataset_names:
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

        try:
            method_scores = load_scores(dataset_name, matched_tfs)
            adata = load_processed_adata(dataset_name)
        except (FileNotFoundError, ValueError) as error:
            print(f"Skipping {dataset_name}: {error}")
            continue

        control_cells = pd.Index(adata.obs.index[adata.obs["condition_clean"] == "control"])

        if len(control_cells) < MIN_CONTROL_CELLS:
            print(f"Skipping {dataset_name}: fewer than {MIN_CONTROL_CELLS} control cells")
            del adata, method_scores
            gc.collect()
            continue

        dataset_plot_dir = OUTPUT_PLOT_DIR / dataset_name
        (dataset_plot_dir / "roc").mkdir(parents=True, exist_ok=True)
        (dataset_plot_dir / "pr").mkdir(parents=True, exist_ok=True)

        ctx = mp.get_context("fork")
        chunksize = get_chunksize(len(matched_tfs), n_processes)

        with ctx.Pool(
            processes=n_processes,
            initializer=init_worker,
            initargs=(
                dataset_name,
                is_activation,
                adata.obs["condition_clean"],
                control_cells,
                method_scores,
                str(dataset_plot_dir),
            ),
        ) as pool:
            iterator = pool.imap_unordered(
                process_tf,
                matched_tfs,
                chunksize=chunksize,
            )

            for tf_rows in tqdm(
                iterator,
                total=len(matched_tfs),
                desc=f"Plotting prior curves for {dataset_name}",
            ):
                if tf_rows:
                    all_rows.extend(tf_rows)

        del adata, method_scores
        gc.collect()

    if all_rows:
        summary_df = pd.DataFrame(all_rows)
        output_file = OUTPUT_PLOT_DIR / f"tf_curve_metrics_{RUN_TAG}.tsv"
        summary_df.to_csv(output_file, sep="\t", index=False)
        print(f"\nSaved summary metrics: {output_file}")

    print(f"\nPlots saved under: {OUTPUT_PLOT_DIR}")


if __name__ == "__main__":
    main()
