from __future__ import annotations

import gc
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm

SCRIPTS_DIR = Path(__file__).resolve().parent.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

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

ANALYSIS_DIR = SCRIPTS_DIR.parent
ADATA_DIR = ANALYSIS_DIR / "scRNASeq"
SCORES_DIR = ANALYSIS_DIR / "scores"
RESULTS_DIR = ANALYSIS_DIR / "results"

MWU_OUT_DIR = RESULTS_DIR / "Methods_MWU-Delongs" / "MWU"
DELONG_OUT_DIR = RESULTS_DIR / "Methods_MWU-Delongs" / "DeLong_top2"

PRIOR_TYPE = "causalpath-priors"
WEIGHT_TYPE = "UNIFORM"

MWU_ALPHA = 0.1
DELONG_ALPHA = 0.1
MIN_PVALUE = 1e-300

METHODS = {
    "z-agg": "z-agg",
    "viper": "VIPER",
    "ulm": "ULM",
    "zscore": "zscore",
}
REQUIRED_METHODS = list(METHODS)

METHOD_FILE_PATTERNS = {
    "z-agg": [f"*z-agg*{PRIOR_TYPE}*{WEIGHT_TYPE}*.parquet"],
    "viper": [f"*viper*{PRIOR_TYPE}*{WEIGHT_TYPE}*.parquet"],
    "ulm": [f"*ulm*{PRIOR_TYPE}*{WEIGHT_TYPE}*.parquet"],
    "zscore": [f"*zscore*{PRIOR_TYPE}*{WEIGHT_TYPE}*.parquet"],
}

MWU_OUT_DIR.mkdir(parents=True, exist_ok=True)
DELONG_OUT_DIR.mkdir(parents=True, exist_ok=True)

DELONG_COLS = [
    "Dataset",
    "TF",
    "Mode",
    "Top_Method",
    "Top_ROC_AUC",
    "Top_MWU_Adjusted_P_Value",
    "Top_N_Cells",
    "Top_N_Pos",
    "Top_N_Control",
    "Second_Method",
    "Second_ROC_AUC",
    "AUC_Diff",
    "DeLong_Z",
    "DeLong_P_Value",
    "DeLong_P_Value_FDR_BH",
    "DeLong_Significant_FDR_BH",
]


def load_method_scores(dataset_name: str, tf_list: list[str]) -> dict[str, pd.DataFrame] | None:
    score_dir = SCORES_DIR / dataset_name
    if not score_dir.exists():
        print(f"Skipping {dataset_name}: missing score directory {score_dir}")
        return None

    scores: dict[str, pd.DataFrame] = {}

    for method_name in REQUIRED_METHODS:
        matches: list[Path] = []
        for pattern in METHOD_FILE_PATTERNS[method_name]:
            matches.extend(sorted(score_dir.glob(pattern)))

        matches = sorted(set(matches), key=lambda path: (len(path.name), path.name))
        if not matches:
            print(f"Skipping {dataset_name}: missing score file for {method_name}")
            return None

        path = matches[0]
        df = pd.read_parquet(path)
        df.index = normalize_index(df.index)
        df.columns = normalize_index(df.columns)
        scores[method_name] = df.reindex(columns=tf_list, fill_value=np.nan)

        print(f"  loaded {method_name}: {path.name}")

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
    method_scores = load_method_scores(dataset_name, tf_list)
    if method_scores is None:
        del adata
        gc.collect()
        return None

    shared_cells = adata.obs_names.copy()
    for df in method_scores.values():
        shared_cells = shared_cells.intersection(df.index)

    if len(shared_cells) == 0:
        print(f"Skipping {dataset_name}: no shared cells across AnnData and score files")
        del adata, method_scores
        gc.collect()
        return None

    adata = adata[shared_cells].copy()
    condition_clean = adata.obs["condition_clean"].astype(str).str.strip()

    for method_name in REQUIRED_METHODS:
        method_scores[method_name] = method_scores[method_name].loc[shared_cells, tf_list]

    return condition_clean, method_scores


def compute_mwu_for_dataset(
    dataset_name: str,
    params: dict,
    condition_clean: pd.Series,
    method_scores: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    control_cells = pd.Index(condition_clean.index[condition_clean == "control"])
    if len(control_cells) < 2:
        return pd.DataFrame()

    rows: list[dict] = []
    tf_list = list(params["common_perturbed_tfs"])

    for tf in tqdm(tf_list, desc=f"MWU {dataset_name}"):
        perturbed_cells = pd.Index(condition_clean.index[condition_clean == tf])
        if len(perturbed_cells) < 2:
            continue

        for method_name, df_scores in method_scores.items():
            result = mann_whitney_perturbed_vs_control(
                series_scores=df_scores[tf],
                perturbed_cells=perturbed_cells,
                control_cells=control_cells,
                is_activation=params["is_activation"],
                min_pvalue=MIN_PVALUE,
            )
            if result is None:
                continue

            rows.append({"Dataset": dataset_name, "TF": tf, "Method": method_name, **result})

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = bh_correct_by_method(
        df,
        methods=REQUIRED_METHODS,
        alpha=MWU_ALPHA,
        min_pvalue=MIN_PVALUE,
    )

    cols = [
        "Dataset",
        "TF",
        "Method",
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

    missing = [col for col in cols if col not in df.columns]
    if missing:
        raise ValueError(
            f"MWU output is missing columns. Check mann_whitney_perturbed_vs_control return values: {missing}"
        )

    return df[cols].sort_values(["Dataset", "Method", "Adjusted_P_Value", "P_Value"])


def build_score_matrix(
    tf: str,
    eval_cells: pd.Index,
    method_scores: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    score_mat = pd.DataFrame(index=eval_cells)

    for method_name, df_scores in method_scores.items():
        if tf not in df_scores.columns:
            continue
        score_mat[method_name] = pd.to_numeric(df_scores.loc[eval_cells, tf], errors="coerce")

    return score_mat.dropna(axis=1, how="all").dropna(axis=0, how="any")


def compute_delong_top2_for_dataset(
    dataset_name: str,
    params: dict,
    condition_clean: pd.Series,
    method_scores: dict[str, pd.DataFrame],
    mwu_df: pd.DataFrame,
) -> pd.DataFrame:
    if mwu_df.empty or "Adjusted_P_Value" not in mwu_df.columns:
        return pd.DataFrame()

    sig = mwu_df[mwu_df["Adjusted_P_Value"].le(MWU_ALPHA)].copy()
    if sig.empty:
        return pd.DataFrame()

    sig_keys = set(zip(sig["Dataset"], sig["TF"], sig["Method"]))
    sig_p = sig.set_index(["Dataset", "TF", "Method"])["Adjusted_P_Value"].to_dict()

    candidate_tfs = sorted(set(params["common_perturbed_tfs"]) & set(sig["TF"]) & set(condition_clean.unique()))

    rows: list[dict] = []
    mode_label = "CRISPRa" if params["is_activation"] else "CRISPRi"

    for tf in tqdm(candidate_tfs, desc=f"DeLong {dataset_name}"):
        eval_cells = pd.Index(condition_clean.index[(condition_clean == tf) | (condition_clean == "control")])
        if len(eval_cells) == 0:
            continue

        y_true = (condition_clean.loc[eval_cells] == tf).astype(int)
        if y_true.nunique() < 2:
            continue

        score_mat = build_score_matrix(tf, eval_cells, method_scores)
        if score_mat.shape[1] < 2 or score_mat.empty:
            continue

        y = y_true.loc[score_mat.index].to_numpy(dtype=int)
        if np.unique(y).size < 2:
            continue

        if not params["is_activation"]:
            score_mat = -score_mat

        method_results: list[dict] = []
        for method_name in score_mat.columns:
            pred = score_mat[method_name].to_numpy(dtype=float)
            if not np.isfinite(pred).all():
                continue

            try:
                auc = roc_auc_score(y, pred)
            except Exception:
                continue

            method_results.append({"method": method_name, "auc": float(auc), "pred": pred})

        if len(method_results) < 2:
            continue

        method_results = sorted(method_results, key=lambda item: item["auc"], reverse=True)
        top1, top2 = method_results[0], method_results[1]

        top_key = (dataset_name, tf, top1["method"])
        if top_key not in sig_keys:
            continue

        _, _, z, p = delong_roc_test(y, top1["pred"], top2["pred"])

        rows.append(
            {
                "Dataset": dataset_name,
                "TF": tf,
                "Mode": mode_label,
                "Top_Method": top1["method"],
                "Top_ROC_AUC": top1["auc"],
                "Top_MWU_Adjusted_P_Value": sig_p.get(top_key, np.nan),
                "Top_N_Cells": int(len(y)),
                "Top_N_Pos": int(np.sum(y)),
                "Top_N_Control": int(len(y) - np.sum(y)),
                "Second_Method": top2["method"],
                "Second_ROC_AUC": top2["auc"],
                "AUC_Diff": float(top1["auc"] - top2["auc"]),
                "DeLong_Z": z,
                "DeLong_P_Value": p,
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    adata_files_with_params = load_adata_files_with_params(PRIOR_TYPE)

    all_mwu: list[pd.DataFrame] = []
    all_delong_raw: list[pd.DataFrame] = []

    for dataset_name, params in adata_files_with_params.items():
        print(f"\n{'=' * 80}\n{dataset_name}\n{'=' * 80}")

        loaded = load_dataset(dataset_name, params)
        if loaded is None:
            continue

        condition_clean, method_scores = loaded

        mwu_df = compute_mwu_for_dataset(
            dataset_name=dataset_name,
            params=params,
            condition_clean=condition_clean,
            method_scores=method_scores,
        )
        if not mwu_df.empty:
            save_tsv(mwu_df, MWU_OUT_DIR / f"MWU_{dataset_name}.tsv")
            all_mwu.append(mwu_df)

        delong_df = compute_delong_top2_for_dataset(
            dataset_name=dataset_name,
            params=params,
            condition_clean=condition_clean,
            method_scores=method_scores,
            mwu_df=mwu_df,
        )
        if not delong_df.empty:
            all_delong_raw.append(delong_df)

        del condition_clean, method_scores, mwu_df, delong_df
        gc.collect()

    if all_mwu:
        merged_mwu = pd.concat(all_mwu, ignore_index=True).sort_values(
            ["Dataset", "Method", "Adjusted_P_Value", "P_Value"]
        )
        save_tsv(merged_mwu, MWU_OUT_DIR / "MWU_merged.tsv")

    if all_delong_raw:
        corrected_delong_by_dataset = []

        for dataset_name, dataset_df in pd.concat(all_delong_raw, ignore_index=True).groupby("Dataset", sort=True):
            dataset_df = apply_fdr_bh(
                dataset_df,
                p_col="DeLong_P_Value",
                adjusted_col="DeLong_P_Value_FDR_BH",
                significant_col="DeLong_Significant_FDR_BH",
                alpha=DELONG_ALPHA,
            ).sort_values(
                ["Top_ROC_AUC", "AUC_Diff"],
                ascending=[False, False],
            )

            dataset_df = dataset_df[DELONG_COLS]
            save_tsv(dataset_df, DELONG_OUT_DIR / f"DeLong_top2_{dataset_name}.tsv")
            corrected_delong_by_dataset.append(dataset_df)

        merged_delong = pd.concat(corrected_delong_by_dataset, ignore_index=True).sort_values(
            ["Dataset", "Top_ROC_AUC", "AUC_Diff"],
            ascending=[True, False, False],
        )

        save_tsv(merged_delong, DELONG_OUT_DIR / "DeLong_top2_merged.tsv")


if __name__ == "__main__":
    main()
