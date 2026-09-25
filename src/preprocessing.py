import logging
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from scipy.sparse import issparse
from scipy.stats import median_abs_deviation, rankdata
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


def preprocess_adata(
    adata: AnnData,
    do_scale: bool = False,
    *,
    min_genes: int | None = None,
    min_cells: int | None = None,
    max_mt_pct: float | None = None,
) -> AnnData:
    """Preprocess an AnnData object with adaptive or user-supplied QC thresholds.

    When no QC thresholds are supplied, adaptive thresholds are used: cells and
    genes are filtered at 1% and 0.1% detection rates, respectively, and the
    mitochondrial cutoff is median + 3 MAD, bounded to 10--25%. Supplying all
    three thresholds selects fixed-threshold preprocessing instead.
    """
    thresholds = (min_genes, min_cells, max_mt_pct)
    if all(value is None for value in thresholds):
        n_cells, n_genes = adata.shape
        return _run_preprocessing(
            adata,
            min_genes=int(0.01 * n_genes),
            min_cells=int(0.001 * n_cells),
            max_mt_pct=None,
            do_scale=do_scale,
            clean_names=True,
        )

    if any(value is None for value in thresholds):
        raise ValueError(
            "Fixed-threshold preprocessing requires min_genes, min_cells, "
            "and max_mt_pct."
        )

    return _run_preprocessing(
        adata,
        min_genes=min_genes,
        min_cells=min_cells,
        max_mt_pct=max_mt_pct,
        do_scale=do_scale,
        clean_names=False,
    )


def _run_preprocessing(
    adata: AnnData,
    *,
    min_genes: int,
    min_cells: int,
    max_mt_pct: float | None,
    do_scale: bool,
    clean_names: bool,
) -> AnnData:
    """Apply common filtering, transformation, and optional scaling steps."""
    adata_copy = adata.copy()
    if clean_names:
        adata_copy.obs_names = pd.Index(adata_copy.obs_names.astype(str)).str.strip()
        adata_copy.var_names = pd.Index(adata_copy.var_names.astype(str)).str.strip()
        adata_copy.var_names_make_unique()

    mode = "adaptive" if max_mt_pct is None else "fixed-threshold"
    logger.info("Starting %s preprocessing. Initial shape: %s", mode, adata_copy.shape)

    sc.pp.filter_cells(adata_copy, min_genes=min_genes)
    sc.pp.filter_genes(adata_copy, min_cells=min_cells)
    logger.info("Shape after basic filtering: %s", adata_copy.shape)

    adata_copy.var["mt"] = adata_copy.var_names.str.upper().str.startswith("MT-")
    sc.pp.calculate_qc_metrics(
        adata_copy, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True
    )

    if max_mt_pct is None:
        mt_pcts = adata_copy.obs["pct_counts_mt"].values
        median_mt = np.median(mt_pcts)
        mad_mt = median_abs_deviation(mt_pcts, scale="normal")
        max_mt_pct = min(max(median_mt + (3 * mad_mt), 10.0), 25.0)
        logger.info(
            "Adaptive mitochondrial cutoff: median=%.2f%%, MAD=%.2f%%, cutoff=%.2f%%.",
            median_mt,
            mad_mt,
            max_mt_pct,
        )

    cells_before = adata_copy.n_obs
    adata_copy = adata_copy[adata_copy.obs["pct_counts_mt"] < max_mt_pct].copy()
    logger.info(
        "Mitochondrial filter (<%s%%): removed %s cells.",
        max_mt_pct,
        cells_before - adata_copy.n_obs,
    )

    sc.pp.normalize_total(adata_copy, target_sum=1e4)
    sc.pp.log1p(adata_copy)
    if do_scale:
        sc.pp.scale(adata_copy)

    logger.info("Preprocessing complete. Final shape: %s", adata_copy.shape)
    return adata_copy


def read_prior_network_file(prior_type: str) -> pd.DataFrame:
    """
    Load prior network from local files.

    Supported:
      - causalpath
      - collectri
      - dorothea
      - ensemble
      - custom file path

    Expected output:
      source | interaction | target
    """

    data_dir = Path("data")
    prior_files = {
        "causalpath": data_dir / "causalpath.tsv",
        "collectri": data_dir / "collectri.tsv",
        "dorothea": data_dir / "dorothea.tsv",
        "ensemble": data_dir / "ensemble.tsv",
    }

    if prior_type in prior_files:
        prior_file = prior_files[prior_type]
    elif os.path.exists(prior_type):
        prior_file = Path(prior_type)
    else:
        raise ValueError(
            f"Unsupported prior_type: {prior_type}. "
            f"Use causalpath, collectri, dorothea, ensemble, or provide a valid file path."
        )

    sep = "\t" if prior_file.suffix.lower() in [".tsv", ".txt"] else ","

    with open(prior_file, "r") as f:
        first_line = f.readline().lower().strip()

    has_header = ("source" in first_line and "target" in first_line) or (
        "tf" in first_line and "gene" in first_line
    )

    if has_header:
        df = pd.read_csv(prior_file, sep=sep)
        df.columns = (
            df.columns.astype(str)
            .str.lower()
            .str.strip()
            .str.replace(" ", "_", regex=False)
        )

        df = df.rename(
            columns={
                "tf": "source",
                "regulator": "source",
                "gene": "target",
                "target_gene": "target",
                "mor": "interaction",
                "mode": "interaction",
                "direction": "interaction",
                "effect": "interaction",
                "sign": "interaction",
            }
        )
        if "interaction" not in df.columns and "weight" in df.columns:
            df = df.rename(columns={"weight": "interaction"})

    else:
        df = pd.read_csv(prior_file, sep=sep, header=None)
        if df.shape[1] == 3:
            df.columns = ["source", "interaction", "target"]
        elif df.shape[1] >= 4:
            df = df.iloc[:, :4]
            df.columns = ["source", "interaction", "target", "weight"]
        else:
            raise ValueError(
                f"Unexpected prior file format. Expected 3 or 4 columns, got {df.shape[1]}."
            )

    required_cols = {"source", "interaction", "target"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Missing required columns. Found columns: {list(df.columns)}")

    interaction_map = {
        "upregulates-expression": 1,
        "downregulates-expression": -1,
        "upregulates": 1,
        "downregulates": -1,
    }

    interaction = df["interaction"]

    if interaction.dtype == "object":
        interaction = (
            interaction.astype(str).str.lower().str.strip().replace(interaction_map)
        )

    interaction = pd.to_numeric(interaction, errors="coerce")
    interaction = np.sign(interaction)
    interaction = pd.Series(interaction, index=df.index).replace(0, np.nan)

    df["interaction"] = interaction

    df["source"] = df["source"].astype(str).str.strip()
    df["target"] = df["target"].astype(str).str.strip()

    cols_to_keep = ["source", "interaction", "target"]

    if "weight" in df.columns:
        df["weight"] = pd.to_numeric(df["weight"], errors="coerce")
        cols_to_keep.append("weight")

    df = df[cols_to_keep]
    df = df.dropna(subset=["source", "interaction", "target"])
    df = df[
        (df["source"] != "")
        & (df["target"] != "")
        & (df["source"].str.lower() != "nan")
        & (df["target"].str.lower() != "nan")
    ]
    df["interaction"] = df["interaction"].astype(int)
    df = df.drop_duplicates().reset_index(drop=True)
    return df


def _spearman_correlations(
    adata: AnnData, net: pd.DataFrame, n_jobs: int | None
) -> pd.Series:
    """Compute pairwise-complete TF-target correlations in network row order."""
    matrix = adata.X.tocsc(copy=False) if issparse(adata.X) else adata.X
    gene_positions = {gene: i for i, gene in enumerate(adata.var_names)}
    tf_targets = net.groupby("source", sort=False)["target"].unique()
    workers = min(n_jobs or os.cpu_count() or 1, len(tf_targets))
    logger.info("   Correlating %s TFs with %s workers.", len(tf_targets), workers)

    def gene_values(gene: str) -> np.ndarray:
        column = matrix[:, gene_positions[gene]]
        return np.asarray(column.toarray() if issparse(column) else column).ravel()

    def correlate_tf(item: tuple[str, np.ndarray]) -> tuple[str, dict[str, float]]:
        tf, targets = item
        if tf not in gene_positions:
            return tf, {}

        tf_values = gene_values(tf)
        tf_finite = np.isfinite(tf_values)
        tf_ranks = rankdata(tf_values) if tf_finite.all() else None
        tf_centered = tf_ranks - tf_ranks.mean() if tf_ranks is not None else None
        tf_sum_sq = float(np.sum(tf_centered**2)) if tf_centered is not None else 0.0
        result = {}

        for target in targets:
            target_values = gene_values(target)
            valid = tf_finite & np.isfinite(target_values)
            if valid.sum() < 2:
                result[target] = 0.0
                continue

            if valid.all():
                x_centered = tf_centered
                x_sum_sq = tf_sum_sq
            else:
                x_ranks = rankdata(tf_values[valid])
                x_centered = x_ranks - x_ranks.mean()
                x_sum_sq = float(np.sum(x_centered**2))

            y_ranks = rankdata(target_values[valid])
            y_centered = y_ranks - y_ranks.mean()
            y_sum_sq = float(np.sum(y_centered**2))
            denominator = np.sqrt(x_sum_sq * y_sum_sq)
            result[target] = (
                float(np.clip(np.sum(x_centered * y_centered) / denominator, -1, 1))
                if denominator > 0
                else 0.0
            )

        return tf, result

    tf_target_corr = {}
    if workers == 1:
        results = map(correlate_tf, tf_targets.items())
        for tf, corrs in tqdm(
            results, total=len(tf_targets), desc="   Correlating TFs", unit="TF"
        ):
            tf_target_corr[tf] = corrs
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            results = executor.map(correlate_tf, tf_targets.items())
            for tf, corrs in tqdm(
                results, total=len(tf_targets), desc="   Correlating TFs", unit="TF"
            ):
                tf_target_corr[tf] = corrs

    return pd.Series(
        (
            tf_target_corr.get(tf, {}).get(target, 0.0)
            for tf, target in zip(net["source"], net["target"])
        ),
        index=net.index,
        dtype=float,
    )


def compute_network_weights(
    adata: AnnData,
    prior_network: pd.DataFrame,
    weight_type: WeightType = WeightType.UNIFORM,
    n_jobs: int | None = None,
) -> pd.DataFrame:
    if n_jobs is not None and n_jobs < 1:
        raise ValueError("n_jobs must be a positive integer or None")
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
        f"   Network Overlap: {final_edges}/{initial_edges} edges ({coverage_pct:.2f}%) target genes present in dataset."
    )

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
        logger.info("   Uniform weights: assigning magnitude 1 to every edge.")
        net["weight"] = 1.0

    elif weight_type == WeightType.CORRELATION:
        logger.info("   Calculating Spearman correlations (TF mRNA vs Target mRNA)...")
        correlations = _spearman_correlations(adata, net, n_jobs)

        nonzero = correlations != 0
        net.loc[nonzero, "interaction"] = np.sign(correlations.loc[nonzero]).astype(int)
        net["weight"] = correlations.abs()

    elif weight_type == WeightType.SPECIFICITY:
        logger.info("   Calculating specificity weights (1 / TF_count per gene)...")
        target_counts = net.groupby("target")["source"].transform("count")
        net["weight"] = 1.0 / target_counts

    elif weight_type == WeightType.NONZERORATE:
        logger.info("   Calculating nonzero rate weights...")
        n_cells = adata.n_obs
        if issparse(adata.X):
            detection_rates = (adata.X > 0).sum(axis=0).A1 / n_cells
        else:
            detection_rates = (adata.X > 0).sum(axis=0) / n_cells
        gene_reliability_map = dict(zip(adata.var_names, detection_rates))
        net["weight"] = net["target"].map(gene_reliability_map)

    elif weight_type == WeightType.EXISTING:
        if "weight" not in net.columns:
            logger.warning(
                "'weight' column not found in priors. Falling back to Uniform weights."
            )
            net["weight"] = 1.0
        else:
            net["weight"] = net["weight"].abs().fillna(1.0)
    else:
        raise ValueError(f"Unknown weight type: {weight_type}")

    net = net[["source", "interaction", "target", "weight"]].copy()
    net["interaction"] = pd.to_numeric(net["interaction"], errors="raise")
    net["weight"] = pd.to_numeric(net["weight"], errors="coerce").fillna(0.0)

    if (~net["interaction"].isin((-1, 1))).any():
        raise ValueError("Network interactions must be -1 or +1.")
    if (~np.isfinite(net["weight"])).any() or (net["weight"] < 0).any():
        raise ValueError("Network weight magnitudes must be finite and non-negative.")

    net["interaction"] = net["interaction"].astype(int)
    net["weight"] = net["weight"].astype(float)
    logger.info("   Weights computed successfully.")
    return net
