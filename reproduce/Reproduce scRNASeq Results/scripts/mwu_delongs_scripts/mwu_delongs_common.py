"""Shared MWU/DeLong operations; statistical implementations live in utility_functions."""

from __future__ import annotations

import gc
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from utility_functions import (  # noqa: E402
    apply_fdr_bh,
    bh_correct_by_method,
    delong_roc_test,
    get_single_perturbation,
    mann_whitney_perturbed_vs_control,
    normalize_index,
    preprocess_adata,
    read_adata_file,
)


def prepare_adata(adata_path: Path, dataset_name: str):
    """Apply the existing preprocessing and retain single-perturbation conditions."""
    adata = read_adata_file(str(adata_path))
    adata = preprocess_adata(adata, do_scale=False)
    adata.obs_names = normalize_index(adata.obs_names)

    adata.obs["condition_clean"] = adata.obs["perturbation"].apply(
        get_single_perturbation
    )
    adata = adata[adata.obs["condition_clean"].notna()].copy()
    adata.obs["condition_clean"] = adata.obs["condition_clean"].astype(str).str.strip()
    return adata


def read_score_table(path: Path, tf_list: list[str]) -> pd.DataFrame:
    """Normalize score axes and retain the requested TFs, including missing columns."""
    scores = pd.read_parquet(path)
    scores.index = normalize_index(scores.index)
    scores.columns = normalize_index(scores.columns)
    return scores.reindex(columns=tf_list, fill_value=np.nan)


def build_score_matrix(
    tf: str,
    eval_cells: pd.Index,
    scores: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Drop all-missing TF columns before restricting to paired, nonmissing cells."""
    score_mat = pd.DataFrame(index=eval_cells)

    for name, df_scores in scores.items():
        if tf not in df_scores.columns:
            continue

        score_mat[name] = pd.to_numeric(
            df_scores.loc[eval_cells, tf],
            errors="coerce",
        )

    return score_mat.dropna(axis=1, how="all").dropna(axis=0, how="any")


def rank_auc_scores(y: np.ndarray, score_mat: pd.DataFrame) -> list[dict]:
    """Rank finite score columns by ROC AUC; ties retain the input column order."""
    results = []
    for name in score_mat.columns:
        pred = score_mat[name].to_numpy(dtype=float)
        if not np.isfinite(pred).all():
            continue
        try:
            auc = roc_auc_score(y, pred)
        except Exception:
            continue
        results.append({"name": name, "auc": float(auc), "pred": pred})
    return sorted(results, key=lambda item: item["auc"], reverse=True)


def correct_mwu(
    rows: list[dict],
    *,
    groups: list[str],
    group_col: str,
    columns: list[str],
    alpha: float,
    min_pvalue: float,
) -> pd.DataFrame:
    """Apply BH within each group in one dataset and retain the output schema."""
    if not rows:
        return pd.DataFrame(columns=columns)
    df = bh_correct_by_method(
        pd.DataFrame(rows),
        methods=groups,
        method_col=group_col,
        alpha=alpha,
        min_pvalue=min_pvalue,
    )
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"MWU output is missing columns: {missing}")
    return df[columns].sort_values(
        ["Dataset", group_col, "Adjusted_P_Value", "P_Value"],
        ascending=[True, True, True, True],
    )


def compute_mwu_for_dataset(
    dataset_name: str,
    params: dict,
    condition_clean: pd.Series,
    scores: dict[str, pd.DataFrame],
    *,
    groups: list[str],
    group_col: str,
    columns: list[str],
    alpha: float,
    min_pvalue: float,
) -> pd.DataFrame:
    """Evaluate Methods/Weights independently, correcting within each dataset/group."""
    control_cells = pd.Index(condition_clean.index[condition_clean == "control"])
    if len(control_cells) < 2:
        return pd.DataFrame(columns=columns)

    rows: list[dict] = []
    tf_list = list(params["common_perturbed_tfs"])

    for tf in tqdm(tf_list, desc=f"MWU {dataset_name}"):
        perturbed_cells = pd.Index(condition_clean.index[condition_clean == tf])
        if len(perturbed_cells) < 2:
            continue

        for name, df_scores in scores.items():
            result = mann_whitney_perturbed_vs_control(
                series_scores=df_scores[tf],
                perturbed_cells=perturbed_cells,
                control_cells=control_cells,
                is_activation=params["is_activation"],
                min_pvalue=min_pvalue,
            )
            if result is None:
                continue

            rows.append(
                {
                    "Dataset": dataset_name,
                    "TF": tf,
                    group_col: name,
                    **result,
                }
            )

    return correct_mwu(
        rows,
        groups=groups,
        group_col=group_col,
        columns=columns,
        alpha=alpha,
        min_pvalue=min_pvalue,
    )


def compute_delong_top2_for_dataset(
    dataset_name: str,
    params: dict,
    condition_clean: pd.Series,
    scores: dict[str, pd.DataFrame],
    mwu_df: pd.DataFrame,
    *,
    group_col: str,
    empty_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Compare the top two AUCs only when the top group passes MWU after BH."""
    if mwu_df.empty:
        return pd.DataFrame(columns=empty_columns)

    mwu_lookup = mwu_df.set_index(["Dataset", "TF", group_col])

    rows: list[dict] = []
    tf_list = list(params["common_perturbed_tfs"])

    for tf in tqdm(tf_list, desc=f"DeLong {dataset_name}"):
        eval_cells = pd.Index(
            condition_clean.index[
                (condition_clean == tf) | (condition_clean == "control")
            ]
        )

        if len(eval_cells) == 0:
            continue

        y_true = (condition_clean.loc[eval_cells] == tf).astype(int)
        if y_true.nunique() < 2:
            continue

        score_mat = build_score_matrix(tf, eval_cells, scores)
        if score_mat.shape[1] < 2 or score_mat.empty:
            continue

        y = y_true.loc[score_mat.index].to_numpy(dtype=int)
        if np.unique(y).size < 2:
            continue

        # Direction correction for ROC/DeLong:
        # CRISPRa: higher score = stronger perturbation.
        # CRISPRi: lower score = stronger perturbation, so flip.
        if not params["is_activation"]:
            score_mat = -score_mat

        results = rank_auc_scores(y, score_mat)
        if len(results) < 2:
            continue

        top1 = results[0]
        top2 = results[1]

        top_key = (dataset_name, tf, top1["name"])
        second_key = (dataset_name, tf, top2["name"])

        if top_key not in mwu_lookup.index:
            continue

        top_mwu_adj = float(mwu_lookup.loc[top_key, "Adjusted_P_Value"])
        top_mwu_sig = bool(mwu_lookup.loc[top_key, "Significant_FDR_BH"])

        # Required rule:
        # Run DeLong only if the top ROC group is significant in MWU after BH.
        # The second group does not need to be MWU-significant.
        if not top_mwu_sig:
            continue

        if second_key in mwu_lookup.index:
            second_mwu_adj = float(mwu_lookup.loc[second_key, "Adjusted_P_Value"])
            second_mwu_sig = bool(mwu_lookup.loc[second_key, "Significant_FDR_BH"])
        else:
            second_mwu_adj = np.nan
            second_mwu_sig = False

        _, _, z, p = delong_roc_test(y, top1["pred"], top2["pred"])

        rows.append(
            {
                "Dataset": dataset_name,
                "TF": tf,
                f"Top_{group_col}": top1["name"],
                "Top_ROC_AUC": top1["auc"],
                "Top_MWU_Adjusted_P_Value": top_mwu_adj,
                "Top_MWU_Significant_FDR_BH": top_mwu_sig,
                "Top_N_Cells": int(len(y)),
                "Top_N_Pos": int(np.sum(y)),
                "Top_N_Control": int(len(y) - np.sum(y)),
                f"Second_{group_col}": top2["name"],
                "Second_ROC_AUC": top2["auc"],
                "Second_MWU_Adjusted_P_Value": second_mwu_adj,
                "Second_MWU_Significant_FDR_BH": second_mwu_sig,
                "AUC_Diff": float(top1["auc"] - top2["auc"]),
                "DeLong_Z": z,
                "DeLong_P_Value": p,
            }
        )

    return pd.DataFrame(rows)


def correct_delong_by_dataset(
    delong_raw: pd.DataFrame, *, columns: list[str], alpha: float
) -> pd.DataFrame:
    if delong_raw.empty:
        return pd.DataFrame(columns=columns)

    corrected_frames: list[pd.DataFrame] = []

    # Required rule:
    # BH correction is applied separately within each dataset.
    for _, dataset_df in delong_raw.groupby("Dataset", sort=True):
        dataset_df = apply_fdr_bh(
            dataset_df,
            p_col="DeLong_P_Value",
            adjusted_col="DeLong_P_Value_FDR_BH",
            significant_col="DeLong_Significant_FDR_BH",
            alpha=alpha,
        )

        dataset_df["DeLong_Significant_FDR_BH"] = (
            dataset_df["DeLong_Significant_FDR_BH"].fillna(False).astype(bool)
        )

        corrected_frames.append(dataset_df)

    corrected = pd.concat(corrected_frames, ignore_index=True)

    for col in columns:
        if col not in corrected.columns:
            corrected[col] = np.nan

    return corrected[columns].sort_values(
        ["Dataset", "Top_ROC_AUC", "AUC_Diff"],
        ascending=[True, False, False],
    )


def make_performance_summary(
    merged_mwu: pd.DataFrame,
    merged_delong: pd.DataFrame,
    *,
    groups: list[str],
    group_col: str,
) -> pd.DataFrame:
    """Summarize MWU results and gated DeLong comparisons for Methods/Weights."""
    summary = pd.DataFrame({group_col: groups})

    if not merged_mwu.empty:
        mwu_summary = merged_mwu.groupby(group_col, as_index=False).agg(
            MWU_Tests=("TF", "size"),
            MWU_Significant=("Significant_FDR_BH", "sum"),
            MWU_Mean_ROC_AUC=("ROC_AUC", "mean"),
            MWU_Median_ROC_AUC=("ROC_AUC", "median"),
            MWU_Mean_PR_AUC=("PR_AUC", "mean"),
            MWU_Median_PR_AUC=("PR_AUC", "median"),
        )
        summary = summary.merge(mwu_summary, on=group_col, how="left")

    if not merged_delong.empty:
        top_summary = (
            merged_delong.groupby(f"Top_{group_col}", as_index=False)
            .agg(
                DeLong_Top2_Tests=("TF", "size"),
                Best_Count_DeLong_Significant=("DeLong_Significant_FDR_BH", "sum"),
                Top_Mean_ROC_AUC=("Top_ROC_AUC", "mean"),
                Top_Median_ROC_AUC=("Top_ROC_AUC", "median"),
                Mean_AUC_Diff=("AUC_Diff", "mean"),
                Median_AUC_Diff=("AUC_Diff", "median"),
            )
            .rename(columns={f"Top_{group_col}": group_col})
        )

        second_summary = (
            merged_delong.groupby(f"Second_{group_col}", as_index=False)
            .agg(**{f"Second_{group_col}_Count": ("TF", "size")})
            .rename(columns={f"Second_{group_col}": group_col})
        )

        summary = summary.merge(top_summary, on=group_col, how="left")
        summary = summary.merge(second_summary, on=group_col, how="left")

    count_cols = [
        "MWU_Tests",
        "MWU_Significant",
        "DeLong_Top2_Tests",
        "Best_Count_DeLong_Significant",
        f"Second_{group_col}_Count",
    ]

    for col in count_cols:
        if col not in summary.columns:
            summary[col] = 0
        summary[col] = summary[col].fillna(0).astype(int)

    numeric_cols = [
        col for col in summary.columns if col not in [group_col, *count_cols]
    ]
    for col in numeric_cols:
        summary[col] = pd.to_numeric(summary[col], errors="coerce")

    return summary.sort_values(
        ["Best_Count_DeLong_Significant", "DeLong_Top2_Tests", "MWU_Significant"],
        ascending=[False, False, False],
    )
