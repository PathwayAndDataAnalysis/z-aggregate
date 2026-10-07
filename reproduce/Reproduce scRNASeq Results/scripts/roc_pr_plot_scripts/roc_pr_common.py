"""Shared ROC/PR calculations and plotting helpers.

Dataset discovery, score-file selection, TF matching, filtering thresholds, and
summary schemas stay in the three comparison scripts.
"""

from __future__ import annotations

import re
import sys
from collections.abc import Collection
from pathlib import Path
from textwrap import fill

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from matplotlib.ticker import MultipleLocator
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from utility_functions import get_single_perturbation, preprocess_adata  # noqa: E402

# Finalized paper style shared by Methods, Weights, and Priors.
FIGURE_SIZE = (4, 4)
PLOT_STYLE = {
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.linewidth": 0.8,
    "axes.edgecolor": "0.25",
    "axes.facecolor": "white",
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": "0.8",
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
    "axes.spines.top": True,
    "axes.spines.right": True,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "xtick.direction": "out",
    "xtick.bottom": True,
    "ytick.direction": "out",
    "ytick.left": True,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "svg.fonttype": "path",
    "pdf.fonttype": 42,
}


def clean_index(index: pd.Index) -> pd.Index:
    return pd.Index(index.astype(str)).str.strip()


def load_score_table(path: Path, tfs: list[str]) -> pd.DataFrame:
    """Normalize score identifiers and retain the requested TF columns."""
    scores = pd.read_parquet(path)
    scores.index = clean_index(scores.index)
    scores.columns = clean_index(scores.columns)
    return scores.reindex(columns=tfs, fill_value=np.nan)


def sanitize_filename(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text))
    text = re.sub(r"_+", "_", text).strip("_")
    return text


def load_processed_adata(dataset_name: str, dataset_dir: Path) -> sc.AnnData:
    dataset_path = dataset_dir / f"{dataset_name}.h5ad"
    adata = sc.read_h5ad(dataset_path)
    adata = preprocess_adata(adata, do_scale=False)

    if "perturbation" not in adata.obs.columns:
        raise ValueError(
            f"'perturbation' column not found in adata.obs for {dataset_name}"
        )

    adata.obs["condition_clean"] = adata.obs["perturbation"].apply(
        get_single_perturbation
    )
    adata = adata[adata.obs["condition_clean"].notna()].copy()
    adata.obs["condition_clean"] = adata.obs["condition_clean"].astype(str).str.strip()
    adata.obs_names = clean_index(adata.obs_names)
    return adata


def compute_curves(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> dict:
    """Compute ROC AUC, average precision, prevalence, and AP lift."""
    roc_auc = roc_auc_score(y_true, y_score)
    fpr, tpr, _ = roc_curve(y_true, y_score)

    ap = average_precision_score(y_true, y_score)
    precision, recall, _ = precision_recall_curve(y_true, y_score)

    baseline_ap = float(y_true.mean())
    ap_lift = ap / baseline_ap if baseline_ap > 0 else np.nan

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


def get_shared_cells(
    tf: str,
    selected_cells: pd.Index,
    method_scores: dict[str, pd.DataFrame],
    required_methods: Collection[str],
) -> pd.Index:
    shared_cells = selected_cells.copy()

    for method_name in required_methods:
        score_df = method_scores[method_name]

        if tf not in score_df.columns:
            return pd.Index([])

        method_valid_cells = score_df.index[
            score_df.index.isin(selected_cells)
            & pd.to_numeric(score_df[tf], errors="coerce").notna()
        ]

        shared_cells = shared_cells.intersection(method_valid_cells)

    return shared_cells


def compute_tf_curves(
    tf: str,
    shared_cells: pd.Index,
    y_true: np.ndarray,
    method_scores: dict[str, pd.DataFrame],
    required_methods: Collection[str],
    is_activation: bool,
) -> dict[str, dict]:
    """Orient scores by perturbation direction; reject the TF if any score is nonfinite."""
    curves = {}
    for method in required_methods:
        y_score = pd.to_numeric(
            method_scores[method].loc[shared_cells, tf], errors="coerce"
        ).to_numpy(dtype=float)
        if not np.isfinite(y_score).all():
            return {}
        if not is_activation:
            y_score = -y_score
        curves[method] = compute_curves(y_true, y_score)
    return curves


def make_plot_directories(dataset_plot_dir: Path) -> None:
    (dataset_plot_dir / "roc").mkdir(parents=True, exist_ok=True)
    (dataset_plot_dir / "pr").mkdir(parents=True, exist_ok=True)


# Figure rendering
def plot_tf_curves(
    dataset_name: str,
    tf: str,
    curves_by_method: dict[str, dict],
    dataset_plot_dir: Path,
    styles: dict[str, dict[str, str]],
    save_format: str,
    dpi: int,
) -> None:
    """Render paper figures from the unchanged empirical ROC/PR coordinates."""
    dataset_label = fill(dataset_name, width=48)
    tf_file = sanitize_filename(tf)

    with plt.rc_context(PLOT_STYLE):
        for curve_type in ("roc", "pr"):
            fig, ax = plt.subplots(figsize=FIGURE_SIZE)
            try:
                # Reserve space for the dataset header and axis labels.
                fig.subplots_adjust(left=0.16, right=0.916, bottom=0.16, top=0.84)
                ax.set_box_aspect(1)
                ax.set_anchor("W")
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.xaxis.set_major_locator(MultipleLocator(0.2))
                ax.yaxis.set_major_locator(MultipleLocator(0.2))
                ax.tick_params(length=3, width=0.8, pad=3)
                ax.text(
                    0.04,
                    0.95,
                    tf,
                    transform=ax.transAxes,
                    fontsize=12,
                    fontweight="medium",
                    color="0.15",
                    ha="left",
                    va="top",
                    zorder=5,
                )
                ax.text(
                    0,
                    1.03,
                    dataset_label,
                    transform=ax.transAxes,
                    fontsize=8,
                    color="0.35",
                    ha="left",
                    va="bottom",
                    linespacing=1.2,
                )

                metric = "roc_auc" if curve_type == "roc" else "ap"
                ranked_methods = sorted(
                    curves_by_method.items(),
                    key=lambda item: item[1][metric],
                    reverse=True,
                )
                for method_name, payload in ranked_methods:
                    style = styles[method_name]
                    if curve_type == "roc":
                        x, y = payload["fpr"], payload["tpr"]
                        label = f"{style['label']} ({payload['roc_auc']:.4f})"
                    else:
                        x, y = payload["recall"], payload["precision"]
                        label = f"{style['label']} (AP = {payload['ap']:.3g})"
                    ax.plot(
                        x,
                        y,
                        color=style["color"],
                        alpha=0.6,
                        linestyle="-",
                        linewidth=1.2,
                        label=label,
                    )

                if curve_type == "roc":
                    ax.plot(
                        [0, 1],
                        [0, 1],
                        color="0.6",
                        linestyle=(0, (4, 3)),
                        linewidth=1,
                        label="_nolegend_",
                        zorder=0,
                    )
                    ax.set_xlabel("False positive rate", fontsize=7)
                    ax.set_ylabel("True positive rate", fontsize=7)
                else:
                    baseline = float(
                        np.median([c["baseline_ap"] for c in curves_by_method.values()])
                    )
                    ax.axhline(
                        baseline,
                        color="0.6",
                        linestyle=(0, (4, 3)),
                        linewidth=1,
                        label=f"Prevalence = {baseline:.3g}",
                        zorder=0,
                    )
                    ax.set_xlabel("Recall", fontsize=7)
                    ax.set_ylabel("Precision", fontsize=7)

                ax.legend(
                    loc="lower right" if curve_type == "roc" else "best",
                    frameon=True,
                    facecolor="white",
                    edgecolor="0.85",
                    framealpha=0.95,
                    fontsize=8,
                    handlelength=2.4,
                    handletextpad=0.6,
                    labelspacing=0.7,
                    borderpad=0.5,
                )
                out_path = (
                    dataset_plot_dir
                    / curve_type
                    / f"{tf_file}_{curve_type}.{save_format}"
                )
                fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.06)
            finally:
                plt.close(fig)
