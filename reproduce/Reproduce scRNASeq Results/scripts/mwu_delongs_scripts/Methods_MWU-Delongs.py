"""Compare activity-inference methods with MWU and gated top-two DeLong tests."""

from __future__ import annotations

import gc
from pathlib import Path

import pandas as pd

import mwu_delongs_common as common
from utility_functions import load_adata_files_with_params, save_tsv

ANALYSIS_DIR = Path(__file__).resolve().parents[2]

ADATA_DIR = ANALYSIS_DIR / "scRNASeq"
SCORES_DIR = ANALYSIS_DIR / "scores"
OUT_DIR = ANALYSIS_DIR / "results" / "Methods_MWU-Delongs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PRIOR_TYPE = "causalpath"
WEIGHT_TYPE = "UNIFORM"

MWU_ALPHA = 0.1
DELONG_ALPHA = 0.1
MIN_PVALUE = 1e-300

REQUIRED_METHODS = ["z-aggregate", "viper", "ulm", "zscore"]

MWU_COLS = [
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

DELONG_COLS = [
    "Dataset",
    "TF",
    "Top_Method",
    "Top_ROC_AUC",
    "Top_MWU_Adjusted_P_Value",
    "Top_MWU_Significant_FDR_BH",
    "Top_N_Cells",
    "Top_N_Pos",
    "Top_N_Control",
    "Second_Method",
    "Second_ROC_AUC",
    "Second_MWU_Adjusted_P_Value",
    "Second_MWU_Significant_FDR_BH",
    "AUC_Diff",
    "DeLong_Z",
    "DeLong_P_Value",
    "DeLong_P_Value_FDR_BH",
    "DeLong_Significant_FDR_BH",
]


def load_method_scores(
    dataset_name: str,
    tf_list: list[str],
) -> dict[str, pd.DataFrame] | None:
    score_dir = SCORES_DIR / dataset_name

    if not score_dir.exists():
        print(f"Skipping {dataset_name}: missing score directory {score_dir}")
        return None

    scores: dict[str, pd.DataFrame] = {}

    for method_name in REQUIRED_METHODS:
        filename = f"{dataset_name}_{method_name}_{PRIOR_TYPE}_{WEIGHT_TYPE}.parquet"
        path = score_dir / filename

        if not path.exists():
            print(
                f"Skipping {dataset_name}: missing score file for {method_name}: {filename}"
            )
            return None

        scores[method_name] = common.read_score_table(path, tf_list)

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

    adata = common.prepare_adata(adata_path, dataset_name)
    if adata is None:
        return None

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
        print(
            f"Skipping {dataset_name}: no shared cells across AnnData and score files"
        )
        del adata, method_scores
        gc.collect()
        return None

    adata = adata[shared_cells].copy()
    condition_clean = adata.obs["condition_clean"].astype(str).str.strip()

    for method_name in REQUIRED_METHODS:
        method_scores[method_name] = method_scores[method_name].loc[
            shared_cells, tf_list
        ]

    return condition_clean, method_scores


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

        mwu_df = common.compute_mwu_for_dataset(
            dataset_name=dataset_name,
            params=params,
            condition_clean=condition_clean,
            scores=method_scores,
            groups=REQUIRED_METHODS,
            group_col="Method",
            columns=MWU_COLS,
            alpha=MWU_ALPHA,
            min_pvalue=MIN_PVALUE,
        )

        if not mwu_df.empty:
            all_mwu.append(mwu_df)

        delong_df = common.compute_delong_top2_for_dataset(
            dataset_name=dataset_name,
            params=params,
            condition_clean=condition_clean,
            scores=method_scores,
            mwu_df=mwu_df,
            group_col="Method",
            empty_columns=[
                col for col in DELONG_COLS if col != "DeLong_P_Value_FDR_BH"
            ],
        )

        if not delong_df.empty:
            all_delong_raw.append(delong_df)

        del condition_clean, method_scores, mwu_df, delong_df
        gc.collect()

    if all_mwu:
        merged_mwu = pd.concat(all_mwu, ignore_index=True).sort_values(
            ["Dataset", "Method", "Adjusted_P_Value", "P_Value"],
            ascending=[True, True, True, True],
        )
    else:
        merged_mwu = pd.DataFrame(columns=MWU_COLS)

    save_tsv(merged_mwu, OUT_DIR / "MWU_merged.tsv")

    if all_delong_raw:
        delong_raw = pd.concat(all_delong_raw, ignore_index=True)
        merged_delong = common.correct_delong_by_dataset(
            delong_raw, columns=DELONG_COLS, alpha=DELONG_ALPHA
        )
    else:
        merged_delong = pd.DataFrame(columns=DELONG_COLS)

    save_tsv(merged_delong, OUT_DIR / "DeLong_top2_merged.tsv")

    method_summary = common.make_performance_summary(
        merged_mwu=merged_mwu,
        merged_delong=merged_delong,
        groups=REQUIRED_METHODS,
        group_col="Method",
    )
    save_tsv(method_summary, OUT_DIR / "Method_performance_summary.tsv")


if __name__ == "__main__":
    main()
