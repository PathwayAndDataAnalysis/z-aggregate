import os
import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from scipy.sparse import csr_matrix, issparse
from scipy.special import ndtr
from sklearn.utils.sparsefuncs import mean_variance_axis, inplace_csr_row_scale
import logging
from tqdm import tqdm
from .WeightType import WeightType

logger = logging.getLogger(__name__)


def read_adata_file(gene_exp_file: str) -> AnnData:
    logger.info(f"Reading expression data from: {gene_exp_file}")
    if not os.path.exists(gene_exp_file):
        raise FileNotFoundError(f"File not found: {gene_exp_file}")

    _, ext = os.path.splitext(gene_exp_file)
    ext = ext.lower()

    if ext == ".h5ad":
        adata = sc.read_h5ad(gene_exp_file)
    elif ext in [".csv", ".tsv", ".txt"]:
        sep = "\t" if ext in [".tsv", ".txt"] else ","
        df = pd.read_csv(gene_exp_file, sep=sep, index_col=0)
        adata = sc.AnnData(df)  # Assume  Cells x Genes input for texts
    else:
        raise ValueError(f"Unsupported format: {ext}")

    logger.info(f"   Loaded data shape: {adata.n_obs} cells x {adata.n_vars} genes")

    adata.obs_names = adata.obs_names.astype(str)
    adata.var_names = adata.var_names.astype(str)

    if not adata.obs_names.is_unique:
        logger.warning("Duplicate cell names found. Making unique.")
        adata.obs_names_make_unique()
    if not adata.var_names.is_unique:
        logger.warning("Duplicate gene names found. Making unique.")
        adata.var_names_make_unique()

    return adata


def preprocess_adata(adata: AnnData, min_genes=1000, min_cells=10, max_mt_pct=20.0) -> AnnData:
    logger.info("Starting Preprocessing...")
    n_cells_init, n_genes_init = adata.shape

    # Filter Cells
    sc.pp.filter_cells(adata, min_genes=min_genes)
    n_cells_step1 = adata.n_obs
    logger.info(f"   Filtered cells (min_genes={min_genes}): Removed {n_cells_init - n_cells_step1} cells.")

    # Filter Genes
    sc.pp.filter_genes(adata, min_cells=min_cells)
    n_genes_step1 = adata.n_vars
    logger.info(f"   Filtered genes (min_cells={min_cells}): Removed {n_genes_init - n_genes_step1} genes.")

    # Mito Filter
    adata.var["mt"] = adata.var_names.str.startswith(("MT-", "mt-"))
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True)

    n_cells_pre_mt = adata.n_obs
    adata = adata[adata.obs["pct_counts_mt"] < max_mt_pct, :].copy()
    n_cells_post_mt = adata.n_obs

    logger.info(f"   Mitochondrial filter (<{max_mt_pct}%): Removed {n_cells_pre_mt - n_cells_post_mt} cells.")

    # Normalize & Log
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    logger.info(f"Preprocessing complete. Final shape: {adata.n_obs} cells x {adata.n_vars} genes")
    return adata


def read_prior_network_file(prior_file: str) -> pd.DataFrame:
    logger.info(f"Loading prior network from: {prior_file}")
    if not os.path.exists(prior_file):
        raise FileNotFoundError(f"File not found: {prior_file}")

    _, ext = os.path.splitext(prior_file)
    sep = "\t" if ext.lower() in [".tsv", ".txt"] else ","

    try:
        df_raw = pd.read_csv(prior_file, sep=sep, header=None)
    except pd.errors.EmptyDataError:
        logger.error("Prior network file is empty.")
        return pd.DataFrame(columns=["source", "interaction", "target", "weight"])

    col_mappings = {
        "source": ["source", "tf", "regulator"],
        "interaction": ["interaction", "mor", "mode", "regulatory_effect", "direction"],
        "target": ["target", "gene", "target_gene"],
        "weight": ["weight", "score", "confidence", "likelihood"],
    }
    all_keywords = {alias for aliases in col_mappings.values() for alias in aliases}

    first_row = df_raw.iloc[0].astype(str).str.lower().str.strip().tolist()
    has_header = any(val in all_keywords for val in first_row)

    if has_header:
        df = pd.read_csv(prior_file, sep=sep, header=0)
        df.columns = df.columns.str.lower().str.strip()
    else:
        logger.debug("No header detected in network file. Inferring columns based on width.")
        df = df_raw
        width = df.shape[1]
        if width == 3:
            df.columns = ["source", "interaction", "target"]
        elif width == 4:
            df.columns = ["source", "interaction", "target", "weight"]
        else:
            raise ValueError(f"Unexpected column count: {width}. Expected 3 or 4.")

    rename_map = {}
    for standard, aliases in col_mappings.items():
        for alias in aliases:
            rename_map[alias] = standard
    df = df.rename(columns=rename_map)

    required = {"source", "interaction", "target"}
    if not required.issubset(df.columns):
        if len(df.columns) == 3:
            df.columns = ["source", "interaction", "target"]
        elif len(df.columns) == 4:
            df.columns = ["source", "interaction", "target", "weight"]
        else:
            raise ValueError(f"Mapping failed. Columns: {list(df.columns)}")

    # 6. Normalize Interaction values
    interaction_map = {"upregulates-expression": 1, "downregulates-expression": -1}

    if df["interaction"].dtype == "object":
        df["interaction"] = df["interaction"].str.lower().str.strip()
        df["interaction"] = df["interaction"].map(interaction_map)
        df["interaction"] = pd.to_numeric(df["interaction"], errors="coerce")

    logger.info(f"   Loaded {len(df)} interactions. Unique TFs: {df['source'].nunique()}")
    return df


def compute_network_weights(
    adata: AnnData, prior_network: pd.DataFrame, weight_type: WeightType = WeightType.UNIFORM
) -> pd.DataFrame:
    logger.info(f"Computing weights using strategy: {weight_type.value}")
    initial_edges = len(prior_network)
    mask = prior_network["target"].isin(set(adata.var_names))
    net = prior_network[mask].copy()
    final_edges = len(net)

    if initial_edges > 0:
        coverage_pct = (final_edges / initial_edges) * 100
    else:
        coverage_pct = 0.0

    logger.info(
        f"   Network Overlap: {final_edges}/{initial_edges} edges ({coverage_pct:.2f}%) target genes present in dataset.")

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

    if weight_type == WeightType.UNIFORM:
        net["weight"] = net["interaction"]

    elif weight_type == WeightType.CORRELATION:
        logger.info("   Calculating Spearman correlations (TF mRNA vs Target mRNA)...")
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
        logger.info("   Calculating specificity weights (1 / TF_count per gene)...")
        target_counts = net.groupby("target")["source"].transform("count")
        net["weight"] = 1.0 / target_counts
        net["weight"] = net["weight"] * net["interaction"]

    elif weight_type == WeightType.NON_ZERO_RATE:
        logger.info("   Calculating nonzero rate weights...")
        n_cells = adata.n_obs
        if issparse(adata.X):
            detection_rates = (adata.X > 0).sum(axis=0).A1 / n_cells
        else:
            detection_rates = (adata.X > 0).sum(axis=0) / n_cells
        gene_reliability_map = dict(zip(adata.var_names, detection_rates))
        net["weight"] = net["target"].map(gene_reliability_map) * net["interaction"]

    elif weight_type == WeightType.EXISTING:
        if "weight" not in net.columns:
            logger.warning("'weight' column not found in priors. Falling back to Uniform weights.")
            net["weight"] = 1.0
        else:
            net["weight"] = net["weight"].abs().fillna(1.0)
    else:
        raise ValueError(f"Unknown weight type: {weight_type}")

    net = net[["source", "interaction", "target", "weight"]].fillna(0.0)
    logger.info("   Weights computed successfully.")
    return net
