from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm

from utility_functions import (
    read_adata_file,
    preprocess_adata,
    get_single_perturbation,
    load_adata_files_with_params,
    mann_whitney_perturbed_vs_control,
    bh_correct_by_method,
    delong_roc_test,
    apply_fdr_bh,
    save_tsv,
    normalize_index,
)

SCRATCH_ROOT = Path("/hpcstor6/scratch01/k/kisan.thapa001/z_agg_data")
ADATA_DIR = SCRATCH_ROOT / "scRNASeq"
SCORES_DIR = SCRATCH_ROOT / "scores"

OUT_DIR = SCRATCH_ROOT / "Weights_MWU-Delongs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PRIOR_TYPE = "causalpath"
METHOD_NAME = "z-aggregate"

WEIGHTS = [
    "UNIFORM",
    "CORRELATION",
    "SPECIFICITY",
    "NONZERORATE",
]

MWU_ALPHA = 0.1
DELONG_ALPHA = 0.1
MIN_PVALUE = 1e-300

MWU_COLS = [
    "Dataset",
    "TF",
    "Weight",
    "N_Pert",
    "N_Control",
    "U_Stat",
    "MWU_AUC_Effect",
    "ROC_AUC",
    "PR_AUC",
    "AP_Baseline",
    "AP_Lift",
    "P_Value",
    "Adjusted_P_Value",
    "Significant_FDR_BH",
    "Mean_Diff",
    "Score",
]

DELONG_COLS = [
    "Dataset",
    "TF",
    "Mode",
    "Top_Weight",
    "Top_ROC_AUC",
    "Top_MWU_Adjusted_P_Value",
    "Top_MWU_Significant_FDR_BH",
    "Top_N_Cells",
    "Top_N_Pos",
    "Top_N_Control",
    "Second_Weight",
    "Second_ROC_AUC",
    "Second_MWU_Adjusted_P_Value",
    "Second_MWU_Significant_FDR_BH",
    "AUC_Diff",
    "DeLong_Z",
    "DeLong_P_Value",
    "DeLong_P_Value_FDR_BH",
    "DeLong_Significant_FDR_BH",
]


def load_weight_scores(
    dataset_name: str,
    tf_list: list[str],
) -> dict[str, pd.DataFrame] | None:
    score_dir = SCORES_DIR / dataset_name
    if not score_dir.exists():
        print(f"Skipping {dataset_name}: missing score directory {score_dir}")
        return None

    scores: dict[str, pd.DataFrame] = {}
    loaded_paths: dict[str, Path] = {}

    for weight in WEIGHTS:
        path = score_dir / f"{dataset_name}_{METHOD_NAME}_{PRIOR_TYPE}_{weight}.parquet"

        if not path.exists():
            print(f"Skipping {dataset_name}: missing score file for {weight}: {path.name}")
            return None

        df = pd.read_parquet(path)
        df.index = normalize_index(df.index)
        df.columns = normalize_index(df.columns)

        scores[weight] = df.reindex(columns=tf_list, fill_value=np.nan)
        loaded_paths[weight] = path

        print(f"  loaded {weight}: {path.name}")

    if len(set(loaded_paths.values())) != len(loaded_paths):
        raise RuntimeError(
            f"Duplicate score files detected for {dataset_name}. "
            "Each weight strategy must load a different parquet file."
        )

    return scores


def load_dataset(
    dataset_name: str,
    params: dict,
) -> tuple[pd.Series, dict[str, pd.DataFrame]] | None:
    adata_path = ADATA_DIR / f"{dataset_name}.h5ad"

    if not adata_path.exists():
        print(f"Skipping {dataset_name}: missing {adata_path}")
        return None

    adata = read_adata_file(str(adata_path))
    adata = preprocess_adata(adata, do_scale=False)
    adata.obs_names = normalize_index(adata.obs_names)

    if "perturbation" not in adata.obs.columns:
        print(f"Skipping {dataset_name}: missing perturbation column")
        del adata
        gc.collect()
        return None

    adata.obs["condition_clean"] = adata.obs["perturbation"].apply(get_single_perturbation)
    adata = adata[adata.obs["condition_clean"].notna()].copy()
    adata.obs["condition_clean"] = adata.obs["condition_clean"].astype(str).str.strip()

    tf_list = list(params["common_perturbed_tfs"])

    weight_scores = load_weight_scores(dataset_name, tf_list)
    if weight_scores is None:
        del adata
        gc.collect()
        return None

    shared_cells = adata.obs_names.copy()
    for df in weight_scores.values():
        shared_cells = shared_cells.intersection(df.index)

    if len(shared_cells) == 0:
        print(f"Skipping {dataset_name}: no shared cells across AnnData and score files")
        del adata, weight_scores
        gc.collect()
        return None

    adata = adata[shared_cells].copy()
    condition_clean = adata.obs["condition_clean"].astype(str).str.strip()

    for weight in WEIGHTS:
        weight_scores[weight] = weight_scores[weight].loc[shared_cells, tf_list]

    return condition_clean, weight_scores


def compute_mwu_for_dataset(
    dataset_name: str,
    params: dict,
    condition_clean: pd.Series,
    weight_scores: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    control_cells = pd.Index(condition_clean.index[condition_clean == "control"])
    if len(control_cells) < 2:
        return pd.DataFrame(columns=MWU_COLS)

    rows: list[dict] = []
    tf_list = list(params["common_perturbed_tfs"])

    for tf in tqdm(tf_list, desc=f"MWU {dataset_name}"):
        perturbed_cells = pd.Index(condition_clean.index[condition_clean == tf])
        if len(perturbed_cells) < 2:
            continue

        for weight, df_scores in weight_scores.items():
            result = mann_whitney_perturbed_vs_control(
                series_scores=df_scores[tf],
                perturbed_cells=perturbed_cells,
                control_cells=control_cells,
                is_activation=params["is_activation"],
                min_pvalue=MIN_PVALUE,
            )

            if result is None:
                continue

            rows.append(
                {
                    "Dataset": dataset_name,
                    "TF": tf,
                    "Weight": weight,
                    **result,
                }
            )

    if not rows:
        return pd.DataFrame(columns=MWU_COLS)

    df = pd.DataFrame(rows)

    df = bh_correct_by_method(
        df,
        methods=WEIGHTS,
        alpha=MWU_ALPHA,
        method_col="Weight",
        min_pvalue=MIN_PVALUE,
    )

    missing = [col for col in MWU_COLS if col not in df.columns]
    if missing:
        raise ValueError(f"MWU output is missing columns: {missing}")

    return df[MWU_COLS].sort_values(
        ["Dataset", "Weight", "Adjusted_P_Value", "P_Value"],
        ascending=[True, True, True, True],
    )


def build_score_matrix(
    tf: str,
    eval_cells: pd.Index,
    weight_scores: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    score_mat = pd.DataFrame(index=eval_cells)

    for weight, df_scores in weight_scores.items():
        if tf not in df_scores.columns:
            continue

        score_mat[weight] = pd.to_numeric(
            df_scores.loc[eval_cells, tf],
            errors="coerce",
        )

    return score_mat.dropna(axis=1, how="all").dropna(axis=0, how="any")


def compute_delong_top2_for_dataset(
    dataset_name: str,
    params: dict,
    condition_clean: pd.Series,
    weight_scores: dict[str, pd.DataFrame],
    mwu_df: pd.DataFrame,
) -> pd.DataFrame:
    if mwu_df.empty:
        return pd.DataFrame()

    mwu_lookup = mwu_df.set_index(["Dataset", "TF", "Weight"])

    rows: list[dict] = []
    tf_list = list(params["common_perturbed_tfs"])

    for tf in tqdm(tf_list, desc=f"DeLong {dataset_name}"):
        eval_cells = pd.Index(condition_clean.index[(condition_clean == tf) | (condition_clean == "control")])

        if len(eval_cells) == 0:
            continue

        y_true = (condition_clean.loc[eval_cells] == tf).astype(int)
        if y_true.nunique() < 2:
            continue

        score_mat = build_score_matrix(tf, eval_cells, weight_scores)
        if score_mat.shape[1] < 2 or score_mat.empty:
            continue

        y = y_true.loc[score_mat.index].to_numpy(dtype=int)
        if np.unique(y).size < 2:
            continue

        if not params["is_activation"]:
            score_mat = -score_mat

        weight_results: list[dict] = []

        for weight in score_mat.columns:
            pred = score_mat[weight].to_numpy(dtype=float)

            if not np.isfinite(pred).all():
                continue

            try:
                auc = roc_auc_score(y, pred)
            except Exception:
                continue

            weight_results.append(
                {
                    "weight": weight,
                    "auc": float(auc),
                    "pred": pred,
                }
            )

        if len(weight_results) < 2:
            continue

        weight_results = sorted(
            weight_results,
            key=lambda item: item["auc"],
            reverse=True,
        )

        top1 = weight_results[0]
        top2 = weight_results[1]

        top_key = (dataset_name, tf, top1["weight"])
        second_key = (dataset_name, tf, top2["weight"])

        if top_key not in mwu_lookup.index:
            continue

        top_mwu_adj = float(mwu_lookup.loc[top_key, "Adjusted_P_Value"])
        top_mwu_sig = bool(mwu_lookup.loc[top_key, "Significant_FDR_BH"])

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
                "Top_Weight": top1["weight"],
                "Top_ROC_AUC": top1["auc"],
                "Top_MWU_Adjusted_P_Value": top_mwu_adj,
                "Top_MWU_Significant_FDR_BH": top_mwu_sig,
                "Top_N_Cells": int(len(y)),
                "Top_N_Pos": int(np.sum(y)),
                "Top_N_Control": int(len(y) - np.sum(y)),
                "Second_Weight": top2["weight"],
                "Second_ROC_AUC": top2["auc"],
                "Second_MWU_Adjusted_P_Value": second_mwu_adj,
                "Second_MWU_Significant_FDR_BH": second_mwu_sig,
                "AUC_Diff": float(top1["auc"] - top2["auc"]),
                "DeLong_Z": z,
                "DeLong_P_Value": p,
            }
        )

    return pd.DataFrame(rows)


def correct_delong_by_dataset(delong_raw: pd.DataFrame) -> pd.DataFrame:
    if delong_raw.empty:
        return pd.DataFrame(columns=DELONG_COLS)

    corrected_frames: list[pd.DataFrame] = []
    
    for _, dataset_df in delong_raw.groupby("Dataset", sort=True):
        dataset_df = apply_fdr_bh(
            dataset_df,
            p_col="DeLong_P_Value",
            adjusted_col="DeLong_P_Value_FDR_BH",
            significant_col="DeLong_Significant_FDR_BH",
            alpha=DELONG_ALPHA,
        )

        dataset_df["DeLong_Significant_FDR_BH"] = dataset_df["DeLong_Significant_FDR_BH"].fillna(False).astype(bool)

        corrected_frames.append(dataset_df)

    corrected = pd.concat(corrected_frames, ignore_index=True)

    for col in DELONG_COLS:
        if col not in corrected.columns:
            corrected[col] = np.nan

    return corrected[DELONG_COLS].sort_values(
        ["Dataset", "Top_ROC_AUC", "AUC_Diff"],
        ascending=[True, False, False],
    )


def make_weight_performance_summary(
    merged_mwu: pd.DataFrame,
    merged_delong: pd.DataFrame,
) -> pd.DataFrame:
    summary = pd.DataFrame({"Weight": WEIGHTS})

    if not merged_mwu.empty:
        mwu_summary = merged_mwu.groupby("Weight", as_index=False).agg(
            MWU_Tests=("TF", "size"),
            MWU_Significant=("Significant_FDR_BH", "sum"),
            MWU_Mean_ROC_AUC=("ROC_AUC", "mean"),
            MWU_Median_ROC_AUC=("ROC_AUC", "median"),
            MWU_Mean_PR_AUC=("PR_AUC", "mean"),
            MWU_Median_PR_AUC=("PR_AUC", "median"),
        )

        summary = summary.merge(mwu_summary, on="Weight", how="left")

    if not merged_delong.empty:
        top_summary = (
            merged_delong.groupby("Top_Weight", as_index=False)
            .agg(
                DeLong_Top2_Tests=("TF", "size"),
                Best_Count_DeLong_Significant=("DeLong_Significant_FDR_BH", "sum"),
                Top_Mean_ROC_AUC=("Top_ROC_AUC", "mean"),
                Top_Median_ROC_AUC=("Top_ROC_AUC", "median"),
                Mean_AUC_Diff=("AUC_Diff", "mean"),
                Median_AUC_Diff=("AUC_Diff", "median"),
            )
            .rename(columns={"Top_Weight": "Weight"})
        )

        second_summary = (
            merged_delong.groupby("Second_Weight", as_index=False)
            .agg(Second_Weight_Count=("TF", "size"))
            .rename(columns={"Second_Weight": "Weight"})
        )

        summary = summary.merge(top_summary, on="Weight", how="left")
        summary = summary.merge(second_summary, on="Weight", how="left")

    count_cols = [
        "MWU_Tests",
        "MWU_Significant",
        "DeLong_Top2_Tests",
        "Best_Count_DeLong_Significant",
        "Second_Weight_Count",
    ]

    for col in count_cols:
        if col not in summary.columns:
            summary[col] = 0
        summary[col] = summary[col].fillna(0).astype(int)

    numeric_cols = [col for col in summary.columns if col not in ["Weight", *count_cols]]
    for col in numeric_cols:
        summary[col] = pd.to_numeric(summary[col], errors="coerce")

    return summary.sort_values(
        ["Best_Count_DeLong_Significant", "DeLong_Top2_Tests", "MWU_Significant"],
        ascending=[False, False, False],
    )


def main() -> None:
    adata_files_with_params = load_adata_files_with_params(PRIOR_TYPE)

    all_mwu: list[pd.DataFrame] = []
    all_delong_raw: list[pd.DataFrame] = []

    for dataset_name, params in adata_files_with_params.items():
        print(f"\n{'=' * 80}\n{dataset_name}\n{'=' * 80}")

        loaded = load_dataset(dataset_name, params)
        if loaded is None:
            continue

        condition_clean, weight_scores = loaded

        mwu_df = compute_mwu_for_dataset(
            dataset_name=dataset_name,
            params=params,
            condition_clean=condition_clean,
            weight_scores=weight_scores,
        )

        if not mwu_df.empty:
            all_mwu.append(mwu_df)

        delong_df = compute_delong_top2_for_dataset(
            dataset_name=dataset_name,
            params=params,
            condition_clean=condition_clean,
            weight_scores=weight_scores,
            mwu_df=mwu_df,
        )

        if not delong_df.empty:
            all_delong_raw.append(delong_df)

        del condition_clean, weight_scores, mwu_df, delong_df
        gc.collect()

    if all_mwu:
        merged_mwu = pd.concat(all_mwu, ignore_index=True).sort_values(
            ["Dataset", "Weight", "Adjusted_P_Value", "P_Value"],
            ascending=[True, True, True, True],
        )
    else:
        merged_mwu = pd.DataFrame(columns=MWU_COLS)

    save_tsv(merged_mwu, OUT_DIR / "MWU_merged.tsv")

    if all_delong_raw:
        delong_raw = pd.concat(all_delong_raw, ignore_index=True)
        merged_delong = correct_delong_by_dataset(delong_raw)
    else:
        merged_delong = pd.DataFrame(columns=DELONG_COLS)

    save_tsv(merged_delong, OUT_DIR / "DeLong_top2_merged.tsv")

    weight_summary = make_weight_performance_summary(
        merged_mwu=merged_mwu,
        merged_delong=merged_delong,
    )

    save_tsv(weight_summary, OUT_DIR / "Weight_performance_summary.tsv")


if __name__ == "__main__":
    main()
