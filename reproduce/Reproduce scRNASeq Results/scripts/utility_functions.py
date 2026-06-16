import os
import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib
from pathlib import Path

matplotlib.use("Agg")
from anndata import AnnData
from scipy.stats import median_abs_deviation
from enum import Enum
from scipy.sparse import issparse
import decoupler as dc
import re
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import ast
from typing import Dict
from scipy import stats
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
from sklearn.metrics import average_precision_score, roc_auc_score

ANALYSIS_DIR = Path(__file__).resolve().parent.parent
REPOSITORY_ROOT = ANALYSIS_DIR.parents[1]
PRIOR_DATA_DIR = REPOSITORY_ROOT / "data"

# Dataset-specific directionality.
# Activation datasets: higher score should indicate stronger perturbation.
# Inhibition datasets: lower score should indicate stronger perturbation,
# so scores are sign-flipped before ROC AUC / PR AUC.
ACTIVATED_DATASETS = {"TianKampmann2021_CRISPRa", "NormanWeissman2019_filtered"}
COMMON_TF_DIR = ANALYSIS_DIR / "common_tfs"


class WeightType(str, Enum):
    UNIFORM = "Uniform_Weight"
    CORRELATION = "Correlation_Weight"
    SPECIFICITY = "Specificity_Weight"
    NON_ZERO_RATIO = "Non_Zero_Ratio_Weight"
    EXISTING = "Existing_Weight"


def save_tsv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, sep="\t", index=False)
    print(f"saved: {path}")


def normalize_index(index) -> pd.Index:
    return pd.Index(index.astype(str)).str.strip()


def read_adata_file(gene_exp_file: str, is_cells_x_genes: bool = True) -> AnnData:
    if not os.path.exists(gene_exp_file):
        raise FileNotFoundError(f"File not found: {gene_exp_file}")

    _, ext = os.path.splitext(gene_exp_file)
    ext = ext.lower()

    # Handle H5AD (Native AnnData)
    if ext == ".h5ad":
        adata = sc.read_h5ad(gene_exp_file)

    # Handle Text Formats (CSV/TSV)
    elif ext in [".csv", ".tsv", ".txt"]:
        sep = "\t" if ext in [".tsv", ".txt"] else ","
        try:
            df = pd.read_csv(gene_exp_file, sep=sep, index_col=0)
        except Exception as e:
            raise ValueError(f"Could not parse file: {e}")
        if not is_cells_x_genes:
            df = df.T
        adata = sc.AnnData(df)
    else:
        raise ValueError(f"Unsupported format: {ext}")

    # print(f"--- Data loaded successfully. Shape: {adata.shape}")
    return adata


def preprocess_adata(
    adata: AnnData,
    do_scale: bool,
) -> AnnData:
    adata_copy = adata.copy()
    adata_copy.obs_names = pd.Index(adata_copy.obs_names.astype(str)).str.strip()
    adata_copy.var_names = pd.Index(adata_copy.var_names.astype(str)).str.strip()
    adata_copy.var_names_make_unique()

    # print(f"--- Shape before basic filtering: {adata_copy.shape}")
    n_cells, n_genes = adata_copy.shape
    min_genes = int(0.01 * n_genes)  # 1% of genes expressed per cell
    min_cells = int(0.001 * n_cells)  # 0.1% of cells expressing the gene
    target_sum = 1e4

    sc.pp.filter_cells(adata_copy, min_genes=min_genes)
    sc.pp.filter_genes(adata_copy, min_cells=min_cells)
    # print(f"--- Shape after basic filtering: {adata_copy.shape}")

    # Adaptive Mitochondrial gene filtering
    adata_copy.var["mt"] = adata_copy.var_names.str.upper().str.startswith("MT-")
    sc.pp.calculate_qc_metrics(
        adata_copy,
        qc_vars=["mt"],
        percent_top=None,
        log1p=False,
        inplace=True,
    )
    mt_pcts = adata_copy.obs["pct_counts_mt"].values
    median_mt = np.median(mt_pcts)
    # scale='normal' applies the 1.4826 scaling factor automatically
    mad_mt = median_abs_deviation(mt_pcts, scale="normal")
    # Calculate the adaptive cutoff (Median + 3 MADs)
    mt_cutoff = median_mt + (3 * mad_mt)
    mt_cutoff = max(mt_cutoff, 10.0)  # Never filter stricter than 10%
    mt_cutoff = min(mt_cutoff, 25.0)  # Never allow more than 25% MT
    # Print statistics so you can track what happened
    # print("Mitochondrial Filtering Dataset summary:")
    # print(f"  - Median MT: {median_mt:.2f}%")
    # print(f"  - MAD MT:    {mad_mt:.2f}%")
    # print(f"  - Cutoff:    {mt_cutoff:.2f}%")
    cells_before = adata_copy.n_obs
    # Actually filter the dataset
    adata_copy = adata_copy[adata_copy.obs["pct_counts_mt"] < mt_cutoff].copy()
    cells_after = adata_copy.n_obs
    removed = cells_before - cells_after
    # print(f"  - Cells before: {cells_before:,}")
    # print(f"  - Cells after:  {cells_after:,}")
    # print(f"  - Removed:      {removed:,} high-MT cells\n")

    sc.pp.normalize_total(adata_copy, target_sum=target_sum)
    sc.pp.log1p(adata_copy)

    if do_scale:
        sc.pp.scale(adata_copy)

    # print(f"- Preprocessing complete. Final shape: {adata_copy.shape}")
    return adata_copy


def get_single_perturbation(label):
    CONTROL_RE = re.compile(
        r"^(?:control|ctrl|negctrl|negative[_ -]?control|non[_ -]?targeting|ntc|nt)$",
        re.IGNORECASE,
    )
    GUIDE_RE = re.compile(r"^(?:g\d+|\d+)$", re.IGNORECASE)

    def is_control_token(s: str) -> bool:
        return bool(CONTROL_RE.fullmatch(str(s).strip()))

    if pd.isna(label):
        return "control"

    label = str(label).strip()
    if not label or label.lower() == "nan":
        return "control"

    # Remove trailing guide ID like _g1 or _1 for whole-label control cases
    label_no_guide = re.sub(r"_(?:g\d+|\d+)$", "", label, flags=re.IGNORECASE)

    if is_control_token(label_no_guide):
        return "control"

    parts = label.split("_")

    if len(parts) == 1:
        return re.sub(r"g\d+$", "", parts[0], flags=re.IGNORECASE)

    if len(parts) != 2:
        return None

    p1, p2 = parts

    # Case: TP53_g1 or TP53_1
    if GUIDE_RE.fullmatch(p2):
        base = re.sub(r"g\d+$", "", p1, flags=re.IGNORECASE)
        return "control" if is_control_token(base) else base

    p1_ctrl = is_control_token(p1)
    p2_ctrl = is_control_token(p2)

    if p1_ctrl and p2_ctrl:
        return "control"
    if p1_ctrl and not p2_ctrl:
        return p2
    if p2_ctrl and not p1_ctrl:
        return p1
    if p1 == p2:
        return p1

    return None


def read_prior_network_file(prior_type: str) -> pd.DataFrame:
    # causalpath-priors
    if prior_type == "causalpath-priors":
        prior_file = PRIOR_DATA_DIR / "causalpath-priors.tsv"
        df = pd.read_csv(
            prior_file,
            sep="\t",
            header=None,
            names=["source", "interaction", "target"],
        )

    elif prior_type == "collectri":
        collectri = dc.op.collectri(organism="human", license="academic")
        df = collectri[["source", "target", "weight"]].copy()
        df = df[~df["target"].str.startswith("hsa-", na=False)]
        df.rename(columns={"weight": "interaction"}, inplace=True)

    elif prior_type == "dorothea":
        dorothea = dc.op.dorothea(organism="human", license="academic")
        df = dorothea[["source", "target", "weight"]].copy()
        df.rename(columns={"weight": "interaction"}, inplace=True)

    elif prior_type == "ensemble-priors":
        prior_file = PRIOR_DATA_DIR / "ensemble-priors.tsv"
        df = pd.read_csv(
            prior_file,
            sep="\t",
            header=None,
            names=["source", "interaction", "target"],
        )

    elif os.path.exists(prior_type):
        with open(prior_type, "r") as f:
            first_line = f.readline().lower()
        if "source" in first_line and "target" in first_line:
            df = pd.read_csv(prior_type, sep="\t")
            df.columns = df.columns.str.lower()
        else:
            df = pd.read_csv(prior_type, sep="\t", header=None)
            if df.shape[1] == 3:
                df.columns = ["source", "interaction", "target"]
            elif df.shape[1] >= 4:
                df = df.iloc[:, :4]
                df.columns = ["source", "interaction", "target", "weight"]

    else:
        raise ValueError(f"Unsupported prior type: {prior_type}")

    interaction_map = {
        "upregulates-expression": 1,
        "downregulates-expression": -1,
        "up": 1,
        "down": -1,
    }

    col = df["interaction"]
    # handle strings
    if col.dtype == "object":
        col = col.astype(str).str.lower().str.strip()
        col = col.replace(interaction_map)

    # convert to numeric
    col = pd.to_numeric(col, errors="coerce")
    col = np.sign(col)
    col = col.replace(0, np.nan)
    df["interaction"] = col.astype("Int64")
    cols_to_keep = ["source", "interaction", "target"]
    if "weight" in df.columns:
        cols_to_keep.append("weight")
    df = df[cols_to_keep]
    df = df.dropna(subset=["interaction", "source", "target"])

    return df


def compute_network_weights(
    adata: AnnData,
    prior_network: pd.DataFrame,
    weight_type: WeightType = WeightType.UNIFORM,
) -> pd.DataFrame:
    # print(f"Computing weights using strategy: {weight_type.value}")

    if weight_type == WeightType.UNIFORM:
        # print("   Uniform weights: using interaction as weight (no overlap filtering).")
        net = prior_network.copy()
        net["weight"] = net["interaction"]
        net = net[["source", "interaction", "target", "weight"]].fillna(0.0)
        # print("   Weights computed successfully.")
        return net

    initial_edges = len(prior_network)
    mask = prior_network["target"].isin(set(adata.var_names))
    net = prior_network[mask].copy()
    final_edges = len(net)

    if initial_edges > 0:
        coverage_pct = (final_edges / initial_edges) * 100
    else:
        coverage_pct = 0.0

    # print(
    #     f"   Network Overlap: {final_edges}/{initial_edges} edges ({coverage_pct:.2f}%) target genes present in dataset."
    # )

    if net.empty:
        adata_examples = list(adata.var_names[:5])
        net_examples = list(prior_network["target"].head(5).values)
        error_msg = (
            "No overlapping genes found between network targets and adata genes.\n"
            f"   Dataset Genes (example): {adata_examples}\n"
            f"   Network Targets (example): {net_examples}\n"
            "   Please check gene formats (e.g. UPPER case vs Title Case, Symbols vs Ensembl IDs)."
        )
        raise ValueError(error_msg)

    if weight_type == WeightType.CORRELATION:
        print("   Calculating Spearman correlations (TF mRNA vs Target mRNA)...")
        tf_target_corr = {}
        unique_tfs = net["source"].unique()

        for tf in tqdm(unique_tfs, desc="   Correlating TFs", unit="TF"):
            if tf not in adata.var_names:
                continue

            tf_vec = adata[:, tf].X.toarray().flatten() if issparse(adata[:, tf].X) else adata[:, tf].X.flatten()
            targets = net.loc[net["source"] == tf, "target"].unique()

            target_mat = adata[:, targets].X.toarray() if issparse(adata[:, targets].X) else adata[:, targets].X

            df_temp = pd.DataFrame(target_mat, columns=targets)
            corrs = df_temp.corrwith(pd.Series(tf_vec), method="spearman")
            tf_target_corr[tf] = corrs.to_dict()

        net["weight"] = net.apply(
            lambda row: tf_target_corr.get(row["source"], {}).get(row["target"], 0.0),
            axis=1,
        )

    elif weight_type == WeightType.SPECIFICITY:
        print("   Calculating specificity weights (1 / TF_count per gene)...")
        target_counts = net.groupby("target")["source"].transform("count")
        net["weight"] = 1.0 / target_counts
        net["weight"] = net["weight"] * net["interaction"]

    elif weight_type == WeightType.NON_ZERO_RATE:
        print("   Calculating nonzero rate weights...")
        n_cells = adata.n_obs
        if issparse(adata.X):
            detection_rates = (adata.X > 0).sum(axis=0).A1 / n_cells
        else:
            detection_rates = (adata.X > 0).sum(axis=0) / n_cells
        gene_reliability_map = dict(zip(adata.var_names, detection_rates))
        net["weight"] = net["target"].map(gene_reliability_map) * net["interaction"]

    elif weight_type == WeightType.EXISTING:
        if "weight" not in net.columns:
            print("'weight' column not found in priors. Falling back to Uniform weights.")
            net["weight"] = 1.0
        else:
            net["weight"] = net["weight"].abs().fillna(1.0)
    else:
        raise ValueError(f"Unknown weight type: {weight_type}")

    net = net[["source", "interaction", "target", "weight"]].fillna(0.0)
    print("   Weights computed successfully.")
    return net


def set_publication_style():
    sns.set_theme(style="whitegrid")
    plt.rcParams.update(
        {
            "font.size": 13,
            "axes.labelsize": 13,
            "axes.titlesize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 10,
            "legend.title_fontsize": 11,
            "figure.figsize": (8, 6),
            "figure.dpi": 120,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def load_adata_files_with_params(prior_type: str) -> Dict[str, dict]:
    prior_file_path = os.path.join(COMMON_TF_DIR, f"common_tfs_{prior_type}.tsv")
    df = pd.read_csv(prior_file_path, sep="\t")

    adata_files_with_params = {}
    for _, row in df.iterrows():
        dataset_name = row["dataset"]
        common_tfs = ast.literal_eval(row["common_tfs"])
        common_tfs = [str(tf).strip() for tf in common_tfs]
        adata_files_with_params[dataset_name] = {
            "is_activation": dataset_name in ACTIVATED_DATASETS,
            "common_perturbed_tfs": common_tfs,
        }
    return adata_files_with_params


def get_method_key(filename: str, dataset_name: str, weight_type: str, prior_type: str) -> str:
    lower = filename.lower()

    if "z-agg" in lower:
        return f"z-agg_{weight_type}"
    elif "viper" in lower:
        return f"viper_{weight_type}"
    elif "ulm" in lower:
        return f"ulm_{weight_type}"
    elif "zscore" in lower:
        return f"zscore_{weight_type}"
    else:
        return filename.replace(f"{dataset_name}_", "").replace(".parquet", "").replace(f"{prior_type}_", "")


# Mann Whitney U Test Code
def mann_whitney_perturbed_vs_control(
    series_scores: pd.Series,
    perturbed_cells: pd.Index,
    control_cells: pd.Index,
    is_activation: bool,
    min_pvalue: float = 1e-300,
) -> dict | None:
    """One-sided MWU test: perturbed TF cells vs control cells."""
    valid_pert = perturbed_cells[perturbed_cells.isin(series_scores.index)]
    valid_ctrl = control_cells[control_cells.isin(series_scores.index)]

    vals_pert = pd.to_numeric(series_scores.loc[valid_pert], errors="coerce").dropna()
    vals_ctrl = pd.to_numeric(series_scores.loc[valid_ctrl], errors="coerce").dropna()

    n_pert = int(len(vals_pert))
    n_ctrl = int(len(vals_ctrl))
    if n_pert < 2 or n_ctrl < 2:
        return None

    alternative = "greater" if is_activation else "less"
    try:
        u_stat, p_value = mannwhitneyu(vals_pert, vals_ctrl, alternative=alternative)
    except Exception:
        return None

    p_value = float(np.clip(p_value, min_pvalue, 1.0))
    roc_pr_metrics = compute_roc_pr_metrics_perturbed_vs_control(
        series_scores=series_scores,
        perturbed_cells=perturbed_cells,
        control_cells=control_cells,
        is_activation=is_activation,
    )
    return {
        "N_Pert": n_pert,
        "N_Control": n_ctrl,
        "U_Stat": float(u_stat),
        "MWU_AUC_Effect": float(u_stat / (n_pert * n_ctrl)),
        "P_Value": p_value,
        "Mean_Diff": float(vals_pert.mean() - vals_ctrl.mean()),
        "Score": float(-np.log(p_value)),
        **roc_pr_metrics,
    }


def bh_correct_by_method(
    df: pd.DataFrame,
    methods: list[str],
    alpha: float = 0.1,
    p_col: str = "P_Value",
    method_col: str = "Method",
    adjusted_col: str = "Adjusted_P_Value",
    significant_col: str = "Significant_FDR_BH",
    min_pvalue: float = 1e-300,
) -> pd.DataFrame:
    """BH correction within each method block of one result table."""
    out = df.copy()
    out[p_col] = pd.to_numeric(out[p_col], errors="coerce").clip(lower=min_pvalue, upper=1.0)
    out[adjusted_col] = np.nan
    out[significant_col] = False

    for method in methods:
        mask = out[method_col].eq(method) & out[p_col].notna()
        if not mask.any():
            continue
        reject, adj, _, _ = multipletests(out.loc[mask, p_col].to_numpy(float), alpha=alpha, method="fdr_bh")
        out.loc[mask, adjusted_col] = adj
        out.loc[mask, significant_col] = reject

    return out


def apply_fdr_bh(
    df: pd.DataFrame,
    p_col: str,
    adjusted_col: str,
    significant_col: str,
    alpha: float = 0.1,
) -> pd.DataFrame:
    """BH correction over all non-null p-values in df[p_col]."""
    out = df.copy()
    out[p_col] = pd.to_numeric(out[p_col], errors="coerce")
    out[adjusted_col] = np.nan
    out[significant_col] = False

    mask = out[p_col].notna()
    if mask.any():
        reject, adj, _, _ = multipletests(out.loc[mask, p_col].to_numpy(float), alpha=alpha, method="fdr_bh")
        out.loc[mask, adjusted_col] = adj
        out.loc[mask, significant_col] = reject

    return out


# Delongs Test Code
def compute_midrank(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    order = np.argsort(x)
    sorted_x = x[order]
    n = len(x)
    ranks = np.zeros(n, dtype=float)

    i = 0
    while i < n:
        j = i
        while j < n and sorted_x[j] == sorted_x[i]:
            j += 1
        ranks[i:j] = 0.5 * (i + j - 1) + 1
        i = j

    out = np.empty(n, dtype=float)
    out[order] = ranks
    return out


def fast_delong(predictions_sorted_transposed: np.ndarray, label_1_count: int) -> tuple[np.ndarray, np.ndarray]:
    m = int(label_1_count)
    n = predictions_sorted_transposed.shape[1] - m
    if m <= 0 or n <= 0:
        raise ValueError("DeLong requires at least one positive and one negative sample.")

    pos = predictions_sorted_transposed[:, :m]
    neg = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.zeros((k, m), dtype=float)
    ty = np.zeros((k, n), dtype=float)
    tz = np.zeros((k, m + n), dtype=float)

    for r in range(k):
        tx[r] = compute_midrank(pos[r])
        ty[r] = compute_midrank(neg[r])
        tz[r] = compute_midrank(predictions_sorted_transposed[r])

    aucs = tz[:, :m].sum(axis=1) / (m * n) - (m + 1.0) / (2.0 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    cov = np.atleast_2d(np.cov(v01)) / m + np.atleast_2d(np.cov(v10)) / n
    return aucs, cov


def delong_roc_test(
    y_true: np.ndarray,
    pred1: np.ndarray,
    pred2: np.ndarray,
    var_epsilon: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Two-sided DeLong test for two correlated ROC AUCs."""
    y_true = np.asarray(y_true).astype(int)
    pred1 = np.asarray(pred1, dtype=float)
    pred2 = np.asarray(pred2, dtype=float)

    keep = np.isfinite(pred1) & np.isfinite(pred2)
    y_true, pred1, pred2 = y_true[keep], pred1[keep], pred2[keep]

    if not np.array_equal(np.unique(y_true), np.array([0, 1])):
        raise ValueError("y_true must contain exactly labels 0 and 1.")

    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))
    if n_pos < 2 or n_neg < 2:
        return np.array([np.nan, np.nan]), np.full((2, 2), np.nan), np.nan, np.nan

    order = np.argsort(-y_true)
    preds_sorted = np.vstack([pred1[order], pred2[order]])
    aucs, cov = fast_delong(preds_sorted, n_pos)

    auc_diff = aucs[0] - aucs[1]
    var = cov[0, 0] + cov[1, 1] - 2.0 * cov[0, 1]
    if var < 0 and abs(var) < var_epsilon:
        var = 0.0

    if var <= var_epsilon:
        if abs(auc_diff) <= var_epsilon:
            return aucs, cov, 0.0, 1.0
        return aucs, cov, np.nan, np.nan

    z = float(auc_diff / np.sqrt(var))
    p_value = float(2.0 * stats.norm.sf(abs(z)))
    return aucs, cov, z, p_value


def compute_roc_pr_metrics_perturbed_vs_control(
    series_scores: pd.Series,
    perturbed_cells: pd.Index,
    control_cells: pd.Index,
    is_activation: bool,
) -> dict:
    valid_perturbed = perturbed_cells[perturbed_cells.isin(series_scores.index)]
    valid_control = control_cells[control_cells.isin(series_scores.index)]

    pert_scores = pd.to_numeric(series_scores.loc[valid_perturbed], errors="coerce").dropna()
    ctrl_scores = pd.to_numeric(series_scores.loc[valid_control], errors="coerce").dropna()

    empty_result = {
        "ROC_AUC": np.nan,
        "PR_AUC": np.nan,
        "AP_Baseline": np.nan,
        "AP_Lift": np.nan,
    }

    if len(pert_scores) == 0 or len(ctrl_scores) == 0:
        return empty_result

    y_true = np.r_[np.ones(len(pert_scores)), np.zeros(len(ctrl_scores))]
    y_score = np.r_[
        pert_scores.to_numpy(dtype=float),
        ctrl_scores.to_numpy(dtype=float),
    ]

    # Direction correction:
    # CRISPRa: higher score should indicate perturbation.
    # CRISPRi: lower score should indicate perturbation, so flip scores.
    if not is_activation:
        y_score = -y_score

    keep = np.isfinite(y_score)
    y_true = y_true[keep]
    y_score = y_score[keep]

    if np.unique(y_true).size < 2:
        return empty_result

    roc_auc = float(roc_auc_score(y_true, y_score))
    pr_auc = float(average_precision_score(y_true, y_score))
    ap_baseline = float(y_true.mean())
    ap_lift = float(pr_auc / ap_baseline) if ap_baseline > 0 else np.nan

    return {
        "ROC_AUC": roc_auc,
        "PR_AUC": pr_auc,
        "AP_Baseline": ap_baseline,
        "AP_Lift": ap_lift,
    }
