import os
import pandas as pd
import numpy as np
import scanpy as sc
from anndata import AnnData
from scipy.sparse import issparse
from scipy.stats import median_abs_deviation
import logging
from pathlib import Path
import re
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

    return _preprocess_with_fixed_thresholds(
        adata,
        do_scale=do_scale,
        min_genes=min_genes,
        min_cells=min_cells,
        max_mt_pct=max_mt_pct,
    )


def _preprocess_with_fixed_thresholds(
    adata: AnnData,
    *,
    do_scale: bool,
    min_genes: int,
    min_cells: int,
    max_mt_pct: float,
) -> AnnData:
    """Apply preprocessing with explicitly supplied QC thresholds."""
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
        "downregulates": -1
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


def compute_network_weights(
    adata: AnnData,
    prior_network: pd.DataFrame,
    weight_type: WeightType = WeightType.UNIFORM,
) -> pd.DataFrame:
    logger.info(f"Computing weights using strategy: {weight_type.value}")

    if weight_type == WeightType.UNIFORM:
        logger.info(
            "   Uniform weights: using interaction as weight (no overlap filtering)."
        )
        net = prior_network.copy()
        net["weight"] = net["interaction"]
        net = net[["source", "interaction", "target", "weight"]].fillna(0.0)
        logger.info("   Weights computed successfully.")
        return net

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

    if weight_type == WeightType.CORRELATION:
        logger.info("   Calculating Spearman correlations (TF mRNA vs Target mRNA)...")
        tf_target_corr = {}
        unique_tfs = net["source"].unique()

        for tf in tqdm(unique_tfs, desc="   Correlating TFs", unit="TF"):
            if tf not in adata.var_names:
                continue

            tf_vec = (
                adata[:, tf].X.toarray().flatten()
                if issparse(adata[:, tf].X)
                else adata[:, tf].X.flatten()
            )
            targets = net.loc[net["source"] == tf, "target"].unique()

            target_mat = (
                adata[:, targets].X.toarray()
                if issparse(adata[:, targets].X)
                else adata[:, targets].X
            )

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
            logger.warning(
                "'weight' column not found in priors. Falling back to Uniform weights."
            )
            net["weight"] = 1.0
        else:
            net["weight"] = net["weight"].abs().fillna(1.0)
    else:
        raise ValueError(f"Unknown weight type: {weight_type}")

    net = net[["source", "interaction", "target", "weight"]].fillna(0.0)
    logger.info("   Weights computed successfully.")
    return net


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
