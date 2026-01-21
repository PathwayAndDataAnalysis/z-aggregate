import numpy as np
import pandas as pd
import scanpy as sc
import logging
from anndata import AnnData
from scipy.sparse import csr_matrix, issparse
from scipy.special import ndtr
from sklearn.utils.sparsefuncs import mean_variance_axis, inplace_csr_row_scale

logger = logging.getLogger(__name__)


def run_z_aggregate(adata, priors: pd.DataFrame, min_targets: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    X = adata.X
    if issparse(X):
        mean, var = mean_variance_axis(X, axis=0)
        std = np.sqrt(var)
    else:
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
    std[std == 0] = 1.0

    # Validate TFs
    tf_counts = priors["source"].value_counts()
    valid_tfs = tf_counts[tf_counts >= min_targets].index

    if len(valid_tfs) == 0:
        logger.warning(f"No TF had >= {min_targets} targets in dataset.")
        empty = pd.DataFrame(index=adata.obs_names)
        return empty, empty

    priors_filtered = priors[priors["source"].isin(valid_tfs)]

    # Build Matrix
    genes_cat = pd.Categorical(priors_filtered["target"], categories=adata.var_names)
    tfs_cat = pd.Categorical(priors_filtered["source"], categories=valid_tfs)

    valid_mask = genes_cat.codes != -1
    row_ind = genes_cat.codes[valid_mask]
    col_ind = tfs_cat.codes[valid_mask]

    raw_weights = priors_filtered["weight"].values[valid_mask]
    directions = priors_filtered["interaction"].values[valid_mask]
    data_val = raw_weights * directions

    W = csr_matrix((data_val, (row_ind, col_ind)), shape=(len(adata.var_names), len(valid_tfs)), dtype=np.float64)

    # Z-Score Calculation
    inv_std = (1.0 / std).astype(np.float64)
    W_scaled = W.copy()
    inplace_csr_row_scale(W_scaled, inv_std)

    term1 = X @ W_scaled
    if issparse(term1):
        term1 = term1.toarray()

    term2 = mean @ W_scaled
    numerator = term1 - term2

    sum_sq_weights = np.array(W.power(2).sum(axis=0)).flatten()
    denominator = np.sqrt(sum_sq_weights)
    denominator[denominator == 0] = 1.0

    final_z = numerator / denominator

    # Convert to P-Values
    abs_z = np.abs(final_z)
    p_values = 2 * ndtr(-abs_z)
    p_values = np.clip(p_values, 1e-300, 1.0)

    activation_dir = np.sign(final_z)

    min_val = np.finfo(p_values.dtype).tiny
    p_values = np.clip(p_values, min_val, 1.0)
    scores = -np.log(p_values) * activation_dir

    scores_df = pd.DataFrame(scores, index=adata.obs_names, columns=valid_tfs)
    pvalues_df = pd.DataFrame(p_values, index=adata.obs_names, columns=valid_tfs)

    return scores_df, pvalues_df
