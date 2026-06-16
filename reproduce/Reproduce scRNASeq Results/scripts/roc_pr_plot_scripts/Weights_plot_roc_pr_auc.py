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
from typing import Dict, List, Optional
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

ANALYSIS_DIR = SCRIPTS_DIR.parent
DATASET_DIR = ANALYSIS_DIR / "scRNASeq"
BASE_RESULT_DIR = ANALYSIS_DIR / "scores"
OUTPUT_PLOT_DIR = ANALYSIS_DIR / "results" / "Weights_ROC_plots"

PRIOR_TYPE = "causalpath-priors"
RUN_TAG = f"{PRIOR_TYPE}_weight-strategies"

WEIGHT_TYPES = [
    "UNIFORM",
    "CORRELATION",
    "SPECIFICITY",
    "NONZERORATE",
]

REQUIRED_METHODS = [f"z-agg_{w}" for w in WEIGHT_TYPES]

METHOD_LABELS = {
    "z-agg_UNIFORM": "Uniform",
    "z-agg_CORRELATION": "Correlation",
    "z-agg_SPECIFICITY": "Specificity",
    "z-agg_NONZERORATE": "Non-zero rate",
}

METHOD_COLORS = {
    "z-agg_UNIFORM": "#0072B2",  # blue
    "z-agg_CORRELATION": "#009E73",  # bluish green
    "z-agg_SPECIFICITY": "#E69F00",  # orange
    "z-agg_NONZERORATE": "#CC79A7",  # magenta
}

TARGET_DATASETS: Optional[List[str]] = None

MIN_POS_CELLS = 5
MIN_CONTROL_CELLS = 5
SAVE_FORMAT = "svg"  # png or svg
DPI = 300

N_PROCESSES = max(
    1,
    int(os.environ.get("ROC_PR_N_PROCESSES", min(8, os.cpu_count() or 1))),
)

os.makedirs(OUTPUT_PLOT_DIR, exist_ok=True)
set_publication_style()


# =========================================================
# Worker globals
# =========================================================
_WORKER_DATASET_NAME: str | None = None
_WORKER_IS_ACTIVATION: bool | None = None
_WORKER_ADATA_OBS_NAMES: pd.Index | None = None
_WORKER_CONDITION_CLEAN: pd.Series | None = None
_WORKER_CONTROL_CELLS: pd.Index | None = None
_WORKER_METHOD_SCORES: Dict[str, pd.DataFrame] | None = None
_WORKER_DATASET_PLOT_DIR: str | None = None


# =========================================================
# Helpers
# =========================================================
def sanitize_filename(text: str) -> str:
    text = str(text)
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text


def load_method_scores(
    dataset_name: str,
    common_tfs: List[str],
    required_methods: List[str],
) -> Dict[str, pd.DataFrame]:
    output_dir = os.path.join(BASE_RESULT_DIR, dataset_name)
    if not os.path.exists(output_dir):
        raise FileNotFoundError(f"Results directory not found: {output_dir}")

    method_scores = {}
    score_prefix = f"{dataset_name}_z-agg_{PRIOR_TYPE}_"
    avail_files = [f for f in os.listdir(output_dir) if f.endswith(".parquet") and f.startswith(score_prefix)]

    method_scores = {}
    for f in avail_files:
        weight_type = f.removesuffix(".parquet").replace(score_prefix, "", 1)
        method_key = f"z-agg_{weight_type}"

        if method_key not in required_methods:
            continue

        path = os.path.join(output_dir, f)
        df_scores = pd.read_parquet(path)
        df_scores.index = pd.Index(df_scores.index.astype(str)).str.strip()
        df_scores.columns = pd.Index(df_scores.columns.astype(str)).str.strip()
        df_scores = df_scores.reindex(columns=common_tfs, fill_value=np.nan)
        method_scores[method_key] = df_scores

    missing = [m for m in required_methods if m not in method_scores]
    if missing:
        raise ValueError(f"Missing required method score files for {dataset_name}: {missing}")

    return method_scores


def load_processed_adata(dataset_name: str) -> sc.AnnData:
    dataset_path = os.path.join(DATASET_DIR, dataset_name + ".h5ad")
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


# =========================================================
# Plotting
# =========================================================
def plot_roc_curve_for_tf(
    dataset_name: str,
    tf: str,
    curves_by_method: Dict[str, dict],
    out_path: str,
):
    plt.figure(figsize=(8, 8), facecolor="white")

    sorted_methods = sorted(
        curves_by_method.items(),
        key=lambda kv: kv[1]["roc_auc"],
        reverse=True,
    )
    for method_name, payload in sorted_methods:
        method_label = METHOD_LABELS.get(method_name, method_name)
        plt.plot(
            payload["fpr"],
            payload["tpr"],
            linewidth=2,
            color=METHOD_COLORS.get(method_name),
            label=f"{method_label} ({payload['roc_auc']:.4f})",
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
    out_path: str,
):
    plt.figure(figsize=(8, 8), facecolor="white")
    baseline_values = [payload["baseline_ap"] for payload in curves_by_method.values()]
    baseline_ap = float(np.median(baseline_values))

    sorted_methods = sorted(
        curves_by_method.items(),
        key=lambda kv: kv[1]["ap"],
        reverse=True,
    )

    for method_name, payload in sorted_methods:
        method_label = METHOD_LABELS.get(method_name, method_name)
        plt.plot(
            payload["recall"],
            payload["precision"],
            linewidth=2,
            color=METHOD_COLORS.get(method_name),
            label=f"{method_label} (AP={payload['ap']:.4f}, lift={payload['ap_lift']:.4f})",
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


# =========================================================
# Multiprocessing
# =========================================================
def init_worker_dataset(
    dataset_name: str,
    is_activation: bool,
    adata_obs_names: pd.Index,
    condition_clean: pd.Series,
    control_cells: pd.Index,
    method_scores: Dict[str, pd.DataFrame],
    dataset_plot_dir: str,
):
    global _WORKER_DATASET_NAME
    global _WORKER_IS_ACTIVATION
    global _WORKER_ADATA_OBS_NAMES
    global _WORKER_CONDITION_CLEAN
    global _WORKER_CONTROL_CELLS
    global _WORKER_METHOD_SCORES
    global _WORKER_DATASET_PLOT_DIR

    _WORKER_DATASET_NAME = dataset_name
    _WORKER_IS_ACTIVATION = is_activation
    _WORKER_ADATA_OBS_NAMES = adata_obs_names
    _WORKER_CONDITION_CLEAN = condition_clean
    _WORKER_CONTROL_CELLS = control_cells
    _WORKER_METHOD_SCORES = method_scores
    _WORKER_DATASET_PLOT_DIR = dataset_plot_dir

    set_publication_style()


def process_tf(tf: str) -> List[dict]:
    dataset_name = _WORKER_DATASET_NAME
    is_activation = _WORKER_IS_ACTIVATION
    adata_obs_names = _WORKER_ADATA_OBS_NAMES
    condition_clean = _WORKER_CONDITION_CLEAN
    control_cells = _WORKER_CONTROL_CELLS
    method_scores = _WORKER_METHOD_SCORES
    dataset_plot_dir = _WORKER_DATASET_PLOT_DIR

    assert dataset_name is not None
    assert is_activation is not None
    assert adata_obs_names is not None
    assert condition_clean is not None
    assert control_cells is not None
    assert method_scores is not None
    assert dataset_plot_dir is not None

    rows: List[dict] = []

    try:
        marked_cells = pd.Index(condition_clean.index[condition_clean == tf])
        if len(marked_cells) < MIN_POS_CELLS:
            return rows

        selected_cells = marked_cells.append(control_cells)

        # Keep only cells present in every required method.
        shared_cells = selected_cells.copy()

        for method_name in REQUIRED_METHODS:
            score_df = method_scores[method_name]

            if tf not in score_df.columns:
                shared_cells = pd.Index([])
                break

            method_valid_cells = score_df.index[
                score_df.index.isin(selected_cells) & pd.to_numeric(score_df[tf], errors="coerce").notna()
            ]

            shared_cells = shared_cells.intersection(method_valid_cells)

        if len(shared_cells) == 0:
            return rows

        y_true_series = pd.Series(0, index=shared_cells, dtype=int)
        y_true_series.loc[y_true_series.index.isin(marked_cells)] = 1
        y_true = y_true_series.to_numpy(dtype=int)

        if (y_true == 1).sum() < MIN_POS_CELLS or (y_true == 0).sum() < MIN_CONTROL_CELLS:
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
        roc_out = Path(dataset_plot_dir) / "roc" / f"{tf_file}_roc.{SAVE_FORMAT}"
        pr_out = Path(dataset_plot_dir) / "pr" / f"{tf_file}_pr.{SAVE_FORMAT}"

        plot_roc_curve_for_tf(
            dataset_name=dataset_name,
            tf=tf,
            curves_by_method=curves_by_method,
            out_path=str(roc_out),
        )

        plot_pr_curve_for_tf(
            dataset_name=dataset_name,
            tf=tf,
            curves_by_method=curves_by_method,
            out_path=str(pr_out),
        )

        for method_name, metrics in curves_by_method.items():
            rows.append(
                {
                    "Dataset": dataset_name,
                    "TF": tf,
                    "Method": method_name,
                    "Method_Label": METHOD_LABELS.get(method_name, method_name),
                    "N_Pos": metrics["n_pos"],
                    "N_Control": metrics["n_neg"],
                    "ROC_AUC": metrics["roc_auc"],
                    "PR_AUC": metrics["ap"],
                    "AP_Baseline": metrics["baseline_ap"],
                    "AP_Lift": metrics["ap_lift"],
                    "ROC_Plot": str(roc_out),
                    "PR_Plot": str(pr_out),
                }
            )

        return rows
    except Exception as e:
        print(f"[ERROR] {dataset_name} | {tf}: {e}")
        return rows


def get_chunksize(n_items: int, n_processes: int) -> int:
    return max(1, n_items // max(1, n_processes * 4))


# =========================================================
# Main
# =========================================================
def main():
    set_publication_style()

    adata_files_with_params = load_adata_files_with_params(PRIOR_TYPE)

    dataset_names = list(adata_files_with_params.keys())
    if TARGET_DATASETS is not None:
        target_set = set(TARGET_DATASETS)
        dataset_names = [d for d in dataset_names if d in target_set]

    available_datasets = {path.stem for path in Path(DATASET_DIR).glob("*.h5ad")}
    missing_datasets = [name for name in dataset_names if name not in available_datasets]
    for dataset_name in missing_datasets:
        print(f"Skipping {dataset_name}: dataset file is not available locally")
    dataset_names = [name for name in dataset_names if name in available_datasets]

    all_rows = []
    n_processes = min(N_PROCESSES, os.cpu_count() or 1)

    print(f"Using {n_processes} worker processes")

    for dataset_name in dataset_names:
        params = adata_files_with_params[dataset_name]
        is_activation = params["is_activation"]
        common_tfs = params["common_perturbed_tfs"]

        print(f"\n{'=' * 60}\nProcessing {dataset_name}\n{'=' * 60}")

        try:
            method_scores = load_method_scores(dataset_name, common_tfs, REQUIRED_METHODS)
            adata = load_processed_adata(dataset_name)
        except (FileNotFoundError, ValueError) as error:
            print(f"Skipping {dataset_name}: {error}")
            continue
        print("Loaded methods:", sorted(method_scores.keys()))

        control_cells = pd.Index(adata.obs.index[adata.obs["condition_clean"] == "control"])
        dataset_plot_dir = Path(OUTPUT_PLOT_DIR) / dataset_name
        dataset_plot_dir.mkdir(parents=True, exist_ok=True)

        (dataset_plot_dir / "roc").mkdir(parents=True, exist_ok=True)
        (dataset_plot_dir / "pr").mkdir(parents=True, exist_ok=True)

        ctx = mp.get_context("fork")
        chunksize = get_chunksize(len(common_tfs), n_processes)

        with ctx.Pool(
            processes=n_processes,
            initializer=init_worker_dataset,
            initargs=(
                dataset_name,
                is_activation,
                adata.obs_names,
                adata.obs["condition_clean"],
                control_cells,
                method_scores,
                str(dataset_plot_dir),
            ),
        ) as pool:
            iterator = pool.imap_unordered(process_tf, common_tfs, chunksize=chunksize)
            for tf_rows in tqdm(
                iterator,
                total=len(common_tfs),
                desc=f"Plotting TF curves for {dataset_name}",
            ):
                if tf_rows:
                    all_rows.extend(tf_rows)

        del adata
        del method_scores
        gc.collect()

    if all_rows:
        summary_df = pd.DataFrame(all_rows)
        out_file = Path(OUTPUT_PLOT_DIR) / f"tf_curve_metrics_{RUN_TAG}.tsv"
        summary_df.to_csv(out_file, sep="\t", index=False)
        print(f"\nSaved summary metrics: {out_file}")

    print(f"\nPlots saved under: {OUTPUT_PLOT_DIR}")


if __name__ == "__main__":
    main()
