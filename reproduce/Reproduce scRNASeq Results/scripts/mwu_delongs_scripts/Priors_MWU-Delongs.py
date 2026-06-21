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
RESULT_ROOT = SCRATCH_ROOT / "Priors_MWU-Delongs"
RESULT_ROOT.mkdir(parents=True, exist_ok=True)

METHOD_NAME = "z-aggregate"
WEIGHT_TYPE = "UNIFORM"

MWU_ALPHA = 0.1
DELONG_ALPHA = 0.1
MIN_PVALUE = 1e-300

BASE_PRIORS = [
    "causalpath",
    "collectri",
    "dorothea",
]

ALL_PRIORS = [
    "causalpath",
    "collectri",
    "dorothea",
    "ensemble",
]

DELONG_RUN_CONFIGS = [
    {
        "include_ensemble": False,
        "using_common_tfs_only": True,
    },
    {
        "include_ensemble": False,
        "using_common_tfs_only": False,
    },
    {
        "include_ensemble": True,
        "using_common_tfs_only": True,
    },
    {
        "include_ensemble": True,
        "using_common_tfs_only": False,
    },
]

MWU_COLS = [
    "Dataset",
    "TF",
    "Prior",
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
    "Comparison_Type",
    "N_Available_Priors",
    "N_MWU_Significant_Options",
    "Top_Prior",
    "Top_ROC_AUC",
    "Top_MWU_Adjusted_P_Value",
    "Top_MWU_Significant_FDR_BH",
    "Top_N_Cells",
    "Top_N_Pos",
    "Top_N_Control",
    "Second_Prior",
    "Second_ROC_AUC",
    "Second_MWU_Adjusted_P_Value",
    "Second_MWU_Significant_FDR_BH",
    "AUC_Diff",
    "DeLong_Z",
    "DeLong_P_Value",
    "DeLong_P_Value_FDR_BH",
    "DeLong_Significant_FDR_BH",
]


def selected_delong_priors(include_ensemble: bool) -> list[str]:
    return ALL_PRIORS if include_ensemble else BASE_PRIORS


def delong_run_label(include_ensemble: bool, using_common_tfs_only: bool) -> str:
    ensemble_label = "With-Ensemble" if include_ensemble else "No-Ensemble"
    tf_label = "CommonTFsOnly" if using_common_tfs_only else "AllAvailableTFs"
    return f"{ensemble_label}_{tf_label}"


def load_params_by_prior() -> dict[str, dict]:
    return {prior: load_adata_files_with_params(prior) for prior in ALL_PRIORS}


def all_datasets(params_by_prior: dict[str, dict]) -> list[str]:
    dataset_sets = [set(params_by_prior[prior]) for prior in ALL_PRIORS]
    return sorted(set.union(*dataset_sets))


def common_datasets(
    params_by_prior: dict[str, dict],
    priors: list[str],
) -> list[str]:
    dataset_sets = [set(params_by_prior[prior]) for prior in priors]
    return sorted(set.intersection(*dataset_sets))


def union_datasets(
    params_by_prior: dict[str, dict],
    priors: list[str],
) -> list[str]:
    dataset_sets = [set(params_by_prior[prior]) for prior in priors]
    return sorted(set.union(*dataset_sets))


def delong_datasets(
    params_by_prior: dict[str, dict],
    priors: list[str],
    using_common_tfs_only: bool,
) -> list[str]:
    if using_common_tfs_only:
        return common_datasets(params_by_prior, priors)
    return union_datasets(params_by_prior, priors)


def priors_available_for_dataset(
    dataset_name: str,
    params_by_prior: dict[str, dict],
    priors: list[str],
) -> list[str]:
    return [prior for prior in priors if dataset_name in params_by_prior[prior]]


def matched_tfs_for_dataset(
    dataset_name: str,
    params_by_prior: dict[str, dict],
    priors: list[str],
) -> list[str]:
    tf_sets = [set(params_by_prior[prior][dataset_name]["common_perturbed_tfs"]) for prior in priors]
    return sorted(set.intersection(*tf_sets))


def union_tfs_for_dataset(
    dataset_name: str,
    params_by_prior: dict[str, dict],
    priors: list[str],
) -> list[str]:
    tf_sets = [
        set(params_by_prior[prior][dataset_name]["common_perturbed_tfs"])
        for prior in priors
        if dataset_name in params_by_prior[prior]
    ]

    if not tf_sets:
        return []

    return sorted(set.union(*tf_sets))


def common_tfs_across_all_resources_by_dataset(
    params_by_prior: dict[str, dict],
) -> dict[str, set[str]]:
    """TFs present in all four prior-resource common-TF lists, by dataset."""
    common_map: dict[str, set[str]] = {}

    for dataset_name in common_datasets(params_by_prior, ALL_PRIORS):
        tf_set = set(
            matched_tfs_for_dataset(
                dataset_name=dataset_name,
                params_by_prior=params_by_prior,
                priors=ALL_PRIORS,
            )
        )
        if tf_set:
            common_map[dataset_name] = tf_set

    return common_map


def filter_mwu_to_common_tfs_only(
    merged_mwu: pd.DataFrame,
    params_by_prior: dict[str, dict],
) -> pd.DataFrame:
    if merged_mwu.empty:
        return pd.DataFrame(columns=MWU_COLS)

    common_map = common_tfs_across_all_resources_by_dataset(params_by_prior)
    if not common_map:
        return pd.DataFrame(columns=MWU_COLS)

    common_pairs = [(dataset_name, tf) for dataset_name, tf_set in common_map.items() for tf in tf_set]

    common_pair_index = pd.MultiIndex.from_tuples(
        common_pairs,
        names=["Dataset", "TF"],
    )

    row_pair_index = pd.MultiIndex.from_frame(merged_mwu[["Dataset", "TF"]])
    common_mask = row_pair_index.isin(common_pair_index)

    out = merged_mwu.loc[common_mask & merged_mwu["Prior"].isin(ALL_PRIORS)].copy()

    if out.empty:
        return pd.DataFrame(columns=MWU_COLS)

    prior_counts = out.drop_duplicates(["Dataset", "TF", "Prior"]).groupby(["Dataset", "TF"])["Prior"].nunique()
    valid_pair_index = prior_counts[prior_counts.eq(len(ALL_PRIORS))].index

    row_pair_index = pd.MultiIndex.from_frame(out[["Dataset", "TF"]])
    out = out.loc[row_pair_index.isin(valid_pair_index)].copy()

    if out.empty:
        return pd.DataFrame(columns=MWU_COLS)

    return out[MWU_COLS].sort_values(
        ["Dataset", "Prior", "Adjusted_P_Value", "P_Value"],
        ascending=[True, True, True, True],
    )


def delong_tfs_for_dataset(
    dataset_name: str,
    params_by_prior: dict[str, dict],
    priors: list[str],
    using_common_tfs_only: bool,
) -> list[str]:
    if using_common_tfs_only:
        return matched_tfs_for_dataset(dataset_name, params_by_prior, priors)

    return union_tfs_for_dataset(dataset_name, params_by_prior, priors)


def load_single_prior_score(
    dataset_name: str,
    prior: str,
    tf_list: list[str],
) -> pd.DataFrame | None:
    score_dir = SCORES_DIR / dataset_name
    if not score_dir.exists():
        print(f"Skipping {dataset_name} [{prior}]: missing score directory {score_dir}")
        return None

    path = score_dir / f"{dataset_name}_{METHOD_NAME}_{prior}_{WEIGHT_TYPE}.parquet"
    if not path.exists():
        print(f"Skipping {dataset_name} [{prior}]: missing score file {path.name}")
        return None

    df = pd.read_parquet(path)
    df.index = normalize_index(df.index)
    df.columns = normalize_index(df.columns)

    return df.reindex(columns=tf_list, fill_value=np.nan)


def load_prior_scores(
    dataset_name: str,
    tf_list: list[str],
    priors: list[str],
    require_all_priors: bool,
) -> dict[str, pd.DataFrame] | None:
    score_dir = SCORES_DIR / dataset_name
    if not score_dir.exists():
        print(f"Skipping {dataset_name}: missing score directory {score_dir}")
        return None

    scores: dict[str, pd.DataFrame] = {}
    loaded_paths: dict[str, Path] = {}

    for prior in priors:
        path = score_dir / f"{dataset_name}_{METHOD_NAME}_{prior}_{WEIGHT_TYPE}.parquet"

        if not path.exists():
            msg = f"missing score file for {prior}: {path.name}"
            if require_all_priors:
                print(f"Skipping {dataset_name}: {msg}")
                return None

            print(f"  Skipping {dataset_name} [{prior}]: {msg}")
            continue

        df = pd.read_parquet(path)
        df.index = normalize_index(df.index)
        df.columns = normalize_index(df.columns)

        scores[prior] = df.reindex(columns=tf_list, fill_value=np.nan)
        loaded_paths[prior] = path

        print(f"  loaded {prior}: {path.name}")

    if not scores:
        print(f"Skipping {dataset_name}: no selected prior score files loaded")
        return None

    if len(set(loaded_paths.values())) != len(loaded_paths):
        raise RuntimeError(
            f"Duplicate score files detected for {dataset_name}. Each prior must load a different parquet file."
        )

    return scores


def load_dataset_for_independent_mwu(
    dataset_name: str,
    params_by_prior: dict[str, dict],
) -> tuple[pd.Series, dict[str, pd.DataFrame], dict[str, list[str]], bool] | None:
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

    condition_clean = adata.obs["condition_clean"].astype(str).str.strip().copy()

    prior_scores: dict[str, pd.DataFrame] = {}
    tf_by_prior: dict[str, list[str]] = {}
    is_activation = None

    for prior in ALL_PRIORS:
        if dataset_name not in params_by_prior[prior]:
            continue

        params = params_by_prior[prior][dataset_name]
        tf_list = list(params["common_perturbed_tfs"])

        score_df = load_single_prior_score(
            dataset_name=dataset_name,
            prior=prior,
            tf_list=tf_list,
        )

        if score_df is None:
            continue

        prior_scores[prior] = score_df
        tf_by_prior[prior] = tf_list

        if is_activation is None:
            is_activation = bool(params["is_activation"])

    del adata
    gc.collect()

    if not prior_scores:
        print(f"Skipping {dataset_name}: no prior score files loaded")
        return None

    return condition_clean, prior_scores, tf_by_prior, bool(is_activation)


def load_dataset_for_delong(
    dataset_name: str,
    params_by_prior: dict[str, dict],
    priors: list[str],
    using_common_tfs_only: bool,
) -> tuple[pd.Series, dict[str, pd.DataFrame], list[str], bool] | None:
    adata_path = ADATA_DIR / f"{dataset_name}.h5ad"
    if not adata_path.exists():
        print(f"Skipping {dataset_name}: missing {adata_path}")
        return None

    available_priors = priors_available_for_dataset(
        dataset_name=dataset_name,
        params_by_prior=params_by_prior,
        priors=priors,
    )

    if using_common_tfs_only and len(available_priors) != len(priors):
        missing = sorted(set(priors) - set(available_priors))
        print(f"Skipping {dataset_name}: missing priors for common-TF DeLong: {missing}")
        return None

    if not available_priors:
        print(f"Skipping {dataset_name}: no selected DeLong priors available")
        return None

    tf_list = delong_tfs_for_dataset(
        dataset_name=dataset_name,
        params_by_prior=params_by_prior,
        priors=priors if using_common_tfs_only else available_priors,
        using_common_tfs_only=using_common_tfs_only,
    )

    if not tf_list:
        tf_msg = "common" if using_common_tfs_only else "available"
        print(f"Skipping {dataset_name}: no {tf_msg} TFs across selected priors")
        return None

    ref_prior = available_priors[0]
    is_activation = bool(params_by_prior[ref_prior][dataset_name]["is_activation"])

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

    score_priors = priors if using_common_tfs_only else available_priors

    prior_scores = load_prior_scores(
        dataset_name=dataset_name,
        tf_list=tf_list,
        priors=score_priors,
        require_all_priors=using_common_tfs_only,
    )

    if prior_scores is None:
        del adata
        gc.collect()
        return None

    shared_cells = adata.obs_names.copy()
    for df in prior_scores.values():
        shared_cells = shared_cells.intersection(df.index)

    if len(shared_cells) == 0:
        print(f"Skipping {dataset_name}: no shared cells across AnnData and score files")
        del adata, prior_scores
        gc.collect()
        return None

    adata = adata[shared_cells].copy()
    condition_clean = adata.obs["condition_clean"].astype(str).str.strip().copy()

    for prior in list(prior_scores):
        prior_scores[prior] = prior_scores[prior].loc[shared_cells, tf_list]

    return condition_clean, prior_scores, tf_list, is_activation


def compute_mwu_for_dataset(
    dataset_name: str,
    tf_by_prior: dict[str, list[str]],
    is_activation: bool,
    condition_clean: pd.Series,
    prior_scores: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    control_cells = pd.Index(condition_clean.index[condition_clean == "control"])
    if len(control_cells) < 2:
        return pd.DataFrame(columns=MWU_COLS)

    rows: list[dict] = []

    for prior, df_scores in prior_scores.items():
        tf_list = tf_by_prior[prior]

        for tf in tqdm(tf_list, desc=f"MWU {dataset_name} [{prior}]"):
            perturbed_cells = pd.Index(condition_clean.index[condition_clean == tf])
            if len(perturbed_cells) < 2:
                continue

            if tf not in df_scores.columns:
                continue

            result = mann_whitney_perturbed_vs_control(
                series_scores=df_scores[tf],
                perturbed_cells=perturbed_cells,
                control_cells=control_cells,
                is_activation=is_activation,
                min_pvalue=MIN_PVALUE,
            )

            if result is None:
                continue

            rows.append(
                {
                    "Dataset": dataset_name,
                    "TF": tf,
                    "Prior": prior,
                    **result,
                }
            )

    if not rows:
        return pd.DataFrame(columns=MWU_COLS)

    df = pd.DataFrame(rows)

    df = bh_correct_by_method(
        df,
        methods=list(prior_scores.keys()),
        alpha=MWU_ALPHA,
        method_col="Prior",
        min_pvalue=MIN_PVALUE,
    )

    missing = [col for col in MWU_COLS if col not in df.columns]
    if missing:
        raise ValueError(f"MWU output is missing columns: {missing}")

    return df[MWU_COLS].sort_values(
        ["Dataset", "Prior", "Adjusted_P_Value", "P_Value"],
        ascending=[True, True, True, True],
    )


def compute_mwu_once(
    params_by_prior: dict[str, dict],
) -> tuple[
    dict[str, pd.DataFrame],
    pd.DataFrame,
    dict[str, pd.DataFrame],
    pd.DataFrame,
]:
    all_mwu: list[pd.DataFrame] = []
    mwu_by_dataset: dict[str, pd.DataFrame] = {}

    print("\nRunning MWU once for all priors.")

    for dataset_name in all_datasets(params_by_prior):
        print(f"\n{'=' * 80}\nMWU {dataset_name}\n{'=' * 80}")

        loaded = load_dataset_for_independent_mwu(dataset_name, params_by_prior)
        if loaded is None:
            continue

        condition_clean, prior_scores, tf_by_prior, is_activation = loaded

        mwu_df = compute_mwu_for_dataset(
            dataset_name=dataset_name,
            tf_by_prior=tf_by_prior,
            is_activation=is_activation,
            condition_clean=condition_clean,
            prior_scores=prior_scores,
        )

        if not mwu_df.empty:
            all_mwu.append(mwu_df)
            mwu_by_dataset[dataset_name] = mwu_df

        del condition_clean, prior_scores, tf_by_prior, mwu_df
        gc.collect()

    if all_mwu:
        merged_mwu = pd.concat(all_mwu, ignore_index=True).sort_values(
            ["Dataset", "Prior", "Adjusted_P_Value", "P_Value"],
            ascending=[True, True, True, True],
        )
    else:
        merged_mwu = pd.DataFrame(columns=MWU_COLS)

    save_tsv(merged_mwu, RESULT_ROOT / "MWU_merged_AllAvailableTFs.tsv")

    merged_mwu_common = filter_mwu_to_common_tfs_only(
        merged_mwu=merged_mwu,
        params_by_prior=params_by_prior,
    )
    save_tsv(merged_mwu_common, RESULT_ROOT / "MWU_merged_CommonTFsOnly.tsv")

    mwu_common_by_dataset = {
        dataset_name: dataset_df.copy() for dataset_name, dataset_df in merged_mwu_common.groupby("Dataset", sort=False)
    }

    return mwu_by_dataset, merged_mwu, mwu_common_by_dataset, merged_mwu_common


def build_score_matrix(
    tf: str,
    eval_cells: pd.Index,
    prior_scores: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    score_mat = pd.DataFrame(index=eval_cells)

    for prior, df_scores in prior_scores.items():
        if tf not in df_scores.columns:
            continue

        score_mat[prior] = pd.to_numeric(
            df_scores.loc[eval_cells, tf],
            errors="coerce",
        )

    return score_mat.dropna(axis=1, how="all").dropna(axis=0, how="any")


def get_mwu_value(
    mwu_lookup: pd.DataFrame,
    key: tuple[str, str, str],
    column: str,
    default=np.nan,
):
    if key not in mwu_lookup.index:
        return default

    row = mwu_lookup.loc[key]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]

    return row.get(column, default)


def get_mwu_significant(
    mwu_lookup: pd.DataFrame,
    key: tuple[str, str, str],
) -> bool:
    value = get_mwu_value(mwu_lookup, key, "Significant_FDR_BH", False)
    if pd.isna(value):
        return False
    return bool(value)


def add_single_available_row(
    rows: list[dict],
    dataset_name: str,
    tf: str,
    prior_result: dict,
    mwu_lookup: pd.DataFrame,
    sig_count_by_tf: dict,
    y: np.ndarray,
) -> None:
    prior = prior_result["prior"]
    top_key = (dataset_name, tf, prior)

    rows.append(
        {
            "Dataset": dataset_name,
            "TF": tf,
            "Comparison_Type": "single_available_significant",
            "N_Available_Priors": 1,
            "N_MWU_Significant_Options": int(sig_count_by_tf.get((dataset_name, tf), 0)),
            "Top_Prior": prior,
            "Top_ROC_AUC": prior_result["auc"],
            "Top_MWU_Adjusted_P_Value": get_mwu_value(
                mwu_lookup,
                top_key,
                "Adjusted_P_Value",
                np.nan,
            ),
            "Top_MWU_Significant_FDR_BH": True,
            "Top_N_Cells": int(len(y)),
            "Top_N_Pos": int(np.sum(y)),
            "Top_N_Control": int(len(y) - np.sum(y)),
            "Second_Prior": np.nan,
            "Second_ROC_AUC": np.nan,
            "Second_MWU_Adjusted_P_Value": np.nan,
            "Second_MWU_Significant_FDR_BH": False,
            "AUC_Diff": np.nan,
            "DeLong_Z": np.nan,
            "DeLong_P_Value": np.nan,
        }
    )


def compute_delong_top2_for_dataset(
    dataset_name: str,
    tf_list: list[str],
    is_activation: bool,
    condition_clean: pd.Series,
    prior_scores: dict[str, pd.DataFrame],
    mwu_df: pd.DataFrame,
    delong_priors: list[str],
    run_label: str,
    using_common_tfs_only: bool,
) -> pd.DataFrame:
    if mwu_df.empty:
        return pd.DataFrame()

    loaded_delong_priors = [prior for prior in delong_priors if prior in prior_scores]
    if not loaded_delong_priors:
        return pd.DataFrame()

    mwu_df = mwu_df[mwu_df["Prior"].isin(loaded_delong_priors)].copy()
    if mwu_df.empty:
        return pd.DataFrame()

    mwu_lookup = mwu_df.drop_duplicates(["Dataset", "TF", "Prior"]).set_index(["Dataset", "TF", "Prior"])

    sig_count_by_tf = (
        mwu_df[mwu_df["Significant_FDR_BH"].fillna(False).astype(bool)]
        .groupby(["Dataset", "TF"])["Prior"]
        .nunique()
        .to_dict()
    )

    mwu_candidate_tfs = set(mwu_df["TF"].astype(str).str.strip())
    candidate_tfs = sorted(set(tf_list) & set(condition_clean.unique()) & mwu_candidate_tfs)
    rows: list[dict] = []

    allow_single_available = not using_common_tfs_only

    for tf in tqdm(candidate_tfs, desc=f"DeLong {dataset_name} [{run_label}]"):
        eval_cells = pd.Index(condition_clean.index[(condition_clean == tf) | (condition_clean == "control")])

        if len(eval_cells) == 0:
            continue

        y_true = (condition_clean.loc[eval_cells] == tf).astype(int)
        if y_true.nunique() < 2:
            continue

        score_mat = build_score_matrix(tf, eval_cells, prior_scores)
        if score_mat.empty:
            continue

        usable_priors = [prior for prior in loaded_delong_priors if prior in score_mat.columns]
        if not usable_priors:
            continue

        score_mat = score_mat[usable_priors]
        y = y_true.loc[score_mat.index].to_numpy(dtype=int)

        if np.unique(y).size < 2:
            continue

        if not is_activation:
            score_mat = -score_mat

        prior_results: list[dict] = []

        for prior in score_mat.columns:
            pred = score_mat[prior].to_numpy(dtype=float)

            if not np.isfinite(pred).all():
                continue

            try:
                auc = roc_auc_score(y, pred)
            except Exception:
                continue

            prior_results.append(
                {
                    "prior": prior,
                    "auc": float(auc),
                    "pred": pred,
                }
            )

        if not prior_results:
            continue

        prior_results = sorted(
            prior_results,
            key=lambda item: item["auc"],
            reverse=True,
        )

        if len(prior_results) == 1:
            if not allow_single_available:
                continue

            top1 = prior_results[0]
            top_key = (dataset_name, tf, top1["prior"])

            if not get_mwu_significant(mwu_lookup, top_key):
                continue

            add_single_available_row(
                rows=rows,
                dataset_name=dataset_name,
                tf=tf,
                prior_result=top1,
                mwu_lookup=mwu_lookup,
                sig_count_by_tf=sig_count_by_tf,
                y=y,
            )
            continue

        top1, top2 = prior_results[0], prior_results[1]
        top_key = (dataset_name, tf, top1["prior"])
        second_key = (dataset_name, tf, top2["prior"])

        if not get_mwu_significant(mwu_lookup, top_key):
            continue

        _, _, z, p = delong_roc_test(y, top1["pred"], top2["pred"])

        rows.append(
            {
                "Dataset": dataset_name,
                "TF": tf,
                "Comparison_Type": "top2_delong",
                "N_Available_Priors": int(len(prior_results)),
                "N_MWU_Significant_Options": int(sig_count_by_tf.get((dataset_name, tf), 0)),
                "Top_Prior": top1["prior"],
                "Top_ROC_AUC": top1["auc"],
                "Top_MWU_Adjusted_P_Value": get_mwu_value(
                    mwu_lookup,
                    top_key,
                    "Adjusted_P_Value",
                    np.nan,
                ),
                "Top_MWU_Significant_FDR_BH": True,
                "Top_N_Cells": int(len(y)),
                "Top_N_Pos": int(np.sum(y)),
                "Top_N_Control": int(len(y) - np.sum(y)),
                "Second_Prior": top2["prior"],
                "Second_ROC_AUC": top2["auc"],
                "Second_MWU_Adjusted_P_Value": get_mwu_value(
                    mwu_lookup,
                    second_key,
                    "Adjusted_P_Value",
                    np.nan,
                ),
                "Second_MWU_Significant_FDR_BH": get_mwu_significant(
                    mwu_lookup,
                    second_key,
                ),
                "AUC_Diff": float(top1["auc"] - top2["auc"]),
                "DeLong_Z": z,
                "DeLong_P_Value": p,
            }
        )

    return pd.DataFrame(rows)


def correct_delong_fdr(dataset_df: pd.DataFrame) -> pd.DataFrame:
    dataset_df = dataset_df.copy()

    dataset_df["DeLong_P_Value_FDR_BH"] = np.nan
    dataset_df["DeLong_Significant_FDR_BH"] = False

    tested_mask = dataset_df["Comparison_Type"].eq("top2_delong")

    if tested_mask.any():
        corrected = apply_fdr_bh(
            dataset_df.loc[tested_mask].copy(),
            p_col="DeLong_P_Value",
            adjusted_col="DeLong_P_Value_FDR_BH",
            significant_col="DeLong_Significant_FDR_BH",
            alpha=DELONG_ALPHA,
        )

        dataset_df.loc[tested_mask, "DeLong_P_Value_FDR_BH"] = corrected["DeLong_P_Value_FDR_BH"].to_numpy()

        dataset_df.loc[tested_mask, "DeLong_Significant_FDR_BH"] = corrected["DeLong_Significant_FDR_BH"].to_numpy()

    single_mask = dataset_df["Comparison_Type"].eq("single_available_significant")
    dataset_df.loc[single_mask, "DeLong_Significant_FDR_BH"] = True
    dataset_df["DeLong_Significant_FDR_BH"] = dataset_df["DeLong_Significant_FDR_BH"].fillna(False).astype(bool)

    return dataset_df


def correct_delong_by_dataset(delong_raw: pd.DataFrame) -> pd.DataFrame:
    if delong_raw.empty:
        return pd.DataFrame(columns=DELONG_COLS)

    corrected_frames: list[pd.DataFrame] = []

    # Dataset-level BH correction for DeLong p-values.
    for _, dataset_df in delong_raw.groupby("Dataset", sort=True):
        corrected_frames.append(correct_delong_fdr(dataset_df))

    corrected = pd.concat(corrected_frames, ignore_index=True)

    for col in DELONG_COLS:
        if col not in corrected.columns:
            corrected[col] = np.nan

    return corrected[DELONG_COLS].sort_values(
        ["Dataset", "Top_ROC_AUC", "AUC_Diff"],
        ascending=[True, False, False],
    )


def make_prior_performance_summary(
    merged_delong: pd.DataFrame,
    priors: list[str],
) -> pd.DataFrame:
    summary = pd.DataFrame({"Prior": priors})

    if not merged_delong.empty:
        top_summary = (
            merged_delong.groupby("Top_Prior", as_index=False)
            .agg(
                Top_Count=("TF", "size"),
                Best_Count_DeLong_Significant=("DeLong_Significant_FDR_BH", "sum"),
                Top2_DeLong_Tests=(
                    "Comparison_Type",
                    lambda x: int((x == "top2_delong").sum()),
                ),
                Single_Available_Significant_Count=(
                    "Comparison_Type",
                    lambda x: int((x == "single_available_significant").sum()),
                ),
                Top_Mean_ROC_AUC=("Top_ROC_AUC", "mean"),
                Top_Median_ROC_AUC=("Top_ROC_AUC", "median"),
                Mean_AUC_Diff=("AUC_Diff", "mean"),
                Median_AUC_Diff=("AUC_Diff", "median"),
            )
            .rename(columns={"Top_Prior": "Prior"})
        )

        second_summary = (
            merged_delong.dropna(subset=["Second_Prior"])
            .groupby("Second_Prior", as_index=False)
            .agg(Second_Prior_Count=("TF", "size"))
            .rename(columns={"Second_Prior": "Prior"})
        )

        summary = summary.merge(top_summary, on="Prior", how="left")
        summary = summary.merge(second_summary, on="Prior", how="left")

    count_cols = [
        "Top_Count",
        "Best_Count_DeLong_Significant",
        "Top2_DeLong_Tests",
        "Single_Available_Significant_Count",
        "Second_Prior_Count",
    ]

    for col in count_cols:
        if col not in summary.columns:
            summary[col] = 0
        summary[col] = summary[col].fillna(0).astype(int)

    numeric_cols = [col for col in summary.columns if col not in ["Prior", *count_cols]]
    for col in numeric_cols:
        summary[col] = pd.to_numeric(summary[col], errors="coerce")

    return summary.sort_values(
        ["Best_Count_DeLong_Significant", "Top_Count", "Top2_DeLong_Tests"],
        ascending=[False, False, False],
    )


def run_delong_config(
    params_by_prior: dict[str, dict],
    mwu_by_dataset: dict[str, pd.DataFrame],
    include_ensemble: bool,
    using_common_tfs_only: bool,
) -> pd.DataFrame:
    delong_priors = selected_delong_priors(include_ensemble)
    run_label = delong_run_label(include_ensemble, using_common_tfs_only)

    print(f"\n{'#' * 80}")
    print(f"DeLong configuration: {run_label}")
    print(f"DeLong priors: {delong_priors}")
    print(f"{'#' * 80}")

    all_delong_raw: list[pd.DataFrame] = []

    dataset_names_for_delong = delong_datasets(
        params_by_prior=params_by_prior,
        priors=delong_priors,
        using_common_tfs_only=using_common_tfs_only,
    )

    for dataset_name in dataset_names_for_delong:
        if dataset_name not in mwu_by_dataset:
            continue

        print(f"\n{'=' * 80}\nDeLong {dataset_name} [{run_label}]\n{'=' * 80}")

        loaded = load_dataset_for_delong(
            dataset_name=dataset_name,
            params_by_prior=params_by_prior,
            priors=delong_priors,
            using_common_tfs_only=using_common_tfs_only,
        )

        if loaded is None:
            continue

        condition_clean, prior_scores, tf_list, is_activation = loaded

        delong_df = compute_delong_top2_for_dataset(
            dataset_name=dataset_name,
            tf_list=tf_list,
            is_activation=is_activation,
            condition_clean=condition_clean,
            prior_scores=prior_scores,
            mwu_df=mwu_by_dataset[dataset_name],
            delong_priors=delong_priors,
            run_label=run_label,
            using_common_tfs_only=using_common_tfs_only,
        )

        if not delong_df.empty:
            all_delong_raw.append(delong_df)

        del condition_clean, prior_scores, tf_list, delong_df
        gc.collect()

    if all_delong_raw:
        delong_raw = pd.concat(all_delong_raw, ignore_index=True)
        merged_delong = correct_delong_by_dataset(delong_raw)
    else:
        merged_delong = pd.DataFrame(columns=DELONG_COLS)

    delong_file = RESULT_ROOT / f"DeLong_top2_{run_label}.tsv"
    save_tsv(merged_delong, delong_file)

    summary = make_prior_performance_summary(
        merged_delong=merged_delong,
        priors=delong_priors,
    )

    summary_file = RESULT_ROOT / f"Prior_performance_summary_{run_label}.tsv"
    save_tsv(summary, summary_file)

    return merged_delong


def main() -> None:
    print(f"Output directory: {RESULT_ROOT}")
    print("DeLong configurations:")

    for cfg in DELONG_RUN_CONFIGS:
        print(
            "  - "
            + delong_run_label(
                include_ensemble=bool(cfg["include_ensemble"]),
                using_common_tfs_only=bool(cfg["using_common_tfs_only"]),
            )
        )

    params_by_prior = load_params_by_prior()
    (
        mwu_by_dataset,
        merged_mwu,
        mwu_common_by_dataset,
        merged_mwu_common,
    ) = compute_mwu_once(params_by_prior)

    if merged_mwu.empty or not mwu_by_dataset:
        print("No MWU results available. Skipping all DeLong configurations.")
        return

    for cfg in DELONG_RUN_CONFIGS:
        using_common_tfs_only = bool(cfg["using_common_tfs_only"])
        mwu_for_run = mwu_common_by_dataset if using_common_tfs_only else mwu_by_dataset

        if using_common_tfs_only and (merged_mwu_common.empty or not mwu_for_run):
            print("No common-TF-only MWU results available. Skipping CommonTFsOnly DeLong configuration.")
            continue

        run_delong_config(
            params_by_prior=params_by_prior,
            mwu_by_dataset=mwu_for_run,
            include_ensemble=bool(cfg["include_ensemble"]),
            using_common_tfs_only=using_common_tfs_only,
        )


if __name__ == "__main__":
    main()
