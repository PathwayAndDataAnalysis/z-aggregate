import numpy as np
import pandas as pd
import logging
from anndata import AnnData
from scipy.sparse import csr_matrix, issparse
from scipy.special import ndtr
from sklearn.utils.sparsefuncs import mean_variance_axis, inplace_csr_row_scale

logger = logging.getLogger(__name__)


def run_z_aggregate(adata: AnnData, priors: pd.DataFrame, min_targets: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    X = adata.X

    if issparse(X):
        mean, var = mean_variance_axis(X, axis=0)
        mean = np.asarray(mean).ravel()
        var = np.asarray(var).ravel()
        std = np.sqrt(var)
    else:
        X = np.asarray(X, dtype=np.float64)
        mean = X.mean(axis=0)
        std = X.std(axis=0)

    std = np.asarray(std, dtype=np.float64).ravel()
    std = np.maximum(std, 1e-12)

    pri = priors.copy()

    # Keep only genes present in this dataset
    pri = pri[pri["target"].isin(adata.var_names)].copy()

    pri["interaction"] = pd.to_numeric(pri["interaction"], errors="raise")
    pri["weight"] = pd.to_numeric(pri["weight"], errors="raise")

    if not pri["interaction"].isin((-1, 1)).all():
        raise ValueError("Prior interactions must be -1 or +1.")
    if (~np.isfinite(pri["weight"])).any() or (pri["weight"] < 0).any():
        raise ValueError("Prior weights must be finite, non-negative magnitudes.")

    pri = pri[pri["weight"] > 0].copy()

    # Apply the regulatory direction
    pri["signed_weight"] = pri["weight"] * pri["interaction"]
    pri = pri.groupby(["source", "target"], as_index=False)["signed_weight"].sum()
    pri = pri[pri["signed_weight"] != 0].copy()

    # Count usable unique targets per TF after dataset intersection
    tf_counts = pri.groupby("source")["target"].nunique()
    valid_tfs = tf_counts[tf_counts >= min_targets].index.tolist()

    if len(valid_tfs) == 0:
        empty = pd.DataFrame(index=adata.obs_names)
        return empty, empty

    pri = pri[pri["source"].isin(valid_tfs)].copy()

    genes_cat = pd.Categorical(pri["target"], categories=adata.var_names)
    tfs_cat = pd.Categorical(pri["source"], categories=valid_tfs)

    row_ind = genes_cat.codes
    col_ind = tfs_cat.codes
    data_val = pri["signed_weight"].to_numpy(dtype=np.float64)

    W = csr_matrix(
        (data_val, (row_ind, col_ind)),
        shape=(len(adata.var_names), len(valid_tfs)),
        dtype=np.float64,
    )

    # Z-Score Calculation
    inv_std = (1.0 / std).astype(np.float64)
    W_scaled = W.copy()
    inplace_csr_row_scale(W_scaled, inv_std)

    term1 = X @ W_scaled
    if issparse(term1):
        term1 = term1.toarray()

    term2 = mean @ W_scaled
    if issparse(term2):
        term2 = term2.toarray()

    numerator = term1 - term2

    sum_sq_weights = np.asarray(W.power(2).sum(axis=0)).ravel()
    denominator = np.sqrt(np.maximum(sum_sq_weights, 1e-12))

    final_z = numerator / denominator

    abs_z = np.abs(final_z)
    p_values = 2 * ndtr(-abs_z)
    p_values = np.clip(p_values, 1e-300, 1.0)

    scores = -np.log(p_values) * np.sign(final_z)

    scores_df = pd.DataFrame(scores, index=adata.obs_names, columns=valid_tfs)
    pvalues_df = pd.DataFrame(p_values, index=adata.obs_names, columns=valid_tfs)

    scores_df = scores_df.astype(np.float64)
    pvalues_df = pvalues_df.astype(np.float64)

    return scores_df, pvalues_df
