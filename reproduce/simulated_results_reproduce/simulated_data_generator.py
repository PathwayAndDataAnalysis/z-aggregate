import json
import os
import sys
import numpy as np
import pandas as pd
import secrets


def generate_prior_heavytail(
    n_genes: int,
    n_tfs: int,
    # total_mu: float = 3.2,
    # total_sigma: float = 1.1,
    total_mu: float = 2.2933,  # Average Mean of Collectri, Dorothea, and CausalPath Priors
    total_sigma: float = 1.3826,  # Average Std of Collectri, Dorothea, and CausalPath Priors
    repression_alpha: float = 1.0,
    repression_beta: float = 10.0,
    weighted: bool = True,
    random_seed: int | None = None,
) -> pd.DataFrame:
    """
    Generate a TF-target prior with:
    - heavy-tailed total targets per TF
    - mostly activating edges
    - non-uniform target gene selection
    """

    rng = np.random.default_rng(random_seed)

    tfs = np.array([f"TF_{i+1}" for i in range(n_tfs)], dtype=object)
    genes = np.array([f"G_{i+1}" for i in range(n_genes)], dtype=object)

    # Gene-level target propensity: creates realistic overlap / gene in-degree heterogeneity
    gene_propensity = rng.lognormal(mean=0.0, sigma=1.0, size=n_genes)
    gene_prob = gene_propensity / gene_propensity.sum()

    rows = []

    # Heavy-tailed total targets per TF
    total_targets = np.rint(
        rng.lognormal(mean=total_mu, sigma=total_sigma, size=n_tfs)
    ).astype(int)
    total_targets = np.clip(total_targets, 1, n_genes)

    # Small repression fraction per TF
    repression_frac = rng.beta(a=repression_alpha, b=repression_beta, size=n_tfs)

    down_targets = rng.binomial(total_targets, repression_frac)
    up_targets = total_targets - down_targets

    for i, tf in enumerate(tfs):
        n_total = total_targets[i]
        n_up = up_targets[i]
        n_down = down_targets[i]

        # Sample unique target genes without replacement
        chosen_idx = rng.choice(
            n_genes,
            size=n_total,
            replace=False,
            p=gene_prob,
        )

        # Shuffle chosen genes and assign first n_up as activating, rest as repressing
        rng.shuffle(chosen_idx)
        up_idx = chosen_idx[:n_up]
        down_idx = chosen_idx[n_up:]

        for gi in up_idx:
            row = {
                "source": tf,
                "target": genes[gi],
                "interaction": 1,
            }
            if weighted:
                row["weight"] = rng.beta(2, 2)  # confidence-like weight in [0,1]
            else:
                row["weight"] = 1.0
            rows.append(row)

        for gi in down_idx:
            row = {
                "source": tf,
                "target": genes[gi],
                "interaction": -1,
            }
            if weighted:
                row["weight"] = rng.beta(2, 2)
            else:
                row["weight"] = 1.0
            rows.append(row)

    df = pd.DataFrame(rows)
    return df


def generate_ground_truth(n_cells, n_tfs, n_clusters=5, random_seed=None):
    rng = np.random.default_rng(random_seed)

    cell_ids = [f"Cell_{i+1}" for i in range(n_cells)]
    tf_names = [f"TF_{j+1}" for j in range(n_tfs)]

    clusters = rng.choice(n_clusters, size=n_cells)
    cluster_patterns = np.zeros((n_clusters, n_tfs), dtype=float)

    for k in range(n_clusters):
        active = rng.choice(n_tfs, size=max(1, n_tfs // 10), replace=False)
        inactive = rng.choice(
            np.setdiff1d(np.arange(n_tfs), active),
            size=max(1, n_tfs // 12),
            replace=False,
        )
        cluster_patterns[k, active] = rng.normal(1.5, 0.3, size=len(active))
        cluster_patterns[k, inactive] = rng.normal(-1.5, 0.3, size=len(inactive))

    latent = cluster_patterns[clusters] + rng.normal(0, 0.5, size=(n_cells, n_tfs))

    disc = np.zeros_like(latent, dtype=np.int8)
    disc[latent > 0.75] = 1
    disc[latent < -0.75] = -1

    return pd.DataFrame(disc, index=cell_ids, columns=tf_names)


def generate_gene_expression(
    n_cells: int,
    n_genes: int,
    tf_effect_factor: float,
    missing_percentage: int,
    random_seed: int,
    prior_dfs: pd.DataFrame,
    ground_truth_dfs: pd.DataFrame,
) -> pd.DataFrame:
    print("Generating gene expression data...")

    if n_cells <= 0 or n_genes <= 0:
        raise ValueError("n_cells and n_genes must be positive.")
    if prior_dfs.empty:
        raise ValueError("prior_df is empty.")
    if ground_truth_dfs.empty:
        raise ValueError("ground_truth_df is empty.")
    if not (0 <= missing_percentage <= 100):
        raise ValueError("missing_percentage must be in [0, 100].")

    required_cols = {"source", "target", "interaction", "weight"}
    if not required_cols.issubset(prior_dfs.columns):
        raise ValueError(f"prior_df must contain columns: {required_cols}")

    missing_ratio = float(missing_percentage) / 100.0
    rng = np.random.default_rng(random_seed)

    cell_ids = [f"Cell_{i + 1}" for i in range(n_cells)]
    gene_names = np.array([f"G_{i + 1}" for i in range(n_genes)], dtype=object)
    tf_names = list(ground_truth_dfs.columns)
    n_tfs = len(tf_names)

    # 1) Generate gene propensities (don't normalize yet)
    gene_prop = rng.lognormal(mean=0, sigma=3, size=n_genes)

    # 2) Initialize rate matrix (Gamma) to create cell-to-cell variability
    theta = 5
    gamma_shape = theta
    gamma_scale = gene_prop / theta
    rate_per_gene = rng.gamma(
        shape=gamma_shape, scale=gamma_scale, size=(n_cells, n_genes)
    )

    # 3) Build TF-target mappings
    gene_index_map = {g: i for i, g in enumerate(gene_names)}
    tf_name_to_idx = {name: i for i, name in enumerate(tf_names)}

    tf_to_up_idx = [[] for _ in range(n_tfs)]
    tf_to_down_idx = [[] for _ in range(n_tfs)]

    tf_to_up_weights = [[] for _ in range(n_tfs)]
    tf_to_down_weights = [[] for _ in range(n_tfs)]

    prior_tf = prior_dfs["source"].to_numpy()
    prior_target = prior_dfs["target"].to_numpy()
    prior_dir = prior_dfs["interaction"].to_numpy()
    prior_w = prior_dfs["weight"].to_numpy()

    for tf, tgt, direction, w in zip(prior_tf, prior_target, prior_dir, prior_w):
        gi = gene_index_map.get(tgt, None)
        ti = tf_name_to_idx.get(tf, None)
        if gi is None or ti is None:
            continue
        if direction == 1:
            tf_to_up_idx[ti].append(gi)
            tf_to_up_weights[ti].append(w)
        elif direction == -1:
            tf_to_down_idx[ti].append(gi)
            tf_to_down_weights[ti].append(w)

    # 4) Apply regulatory effects
    gt_states = ground_truth_dfs.to_numpy(copy=False)

    for ti in range(n_tfs):
        up_genes = np.array(tf_to_up_idx[ti], dtype=np.int64)
        up_w = np.array(tf_to_up_weights[ti], dtype=np.float64)

        down_genes = np.array(tf_to_down_idx[ti], dtype=np.int64)
        down_w = np.array(tf_to_down_weights[ti], dtype=np.float64)

        if up_genes.size == 0 and down_genes.size == 0:
            continue

        states = gt_states[:, ti]
        act_cells = np.where(states == 1)[0]
        inact_cells = np.where(states == -1)[0]

        if up_genes.size > 0:
            up_multipliers = 1 + (tf_effect_factor * up_w)
        if down_genes.size > 0:
            down_multipliers = 1 + (tf_effect_factor * down_w)

        if act_cells.size > 0:
            if up_genes.size > 0:
                rate_per_gene[np.ix_(act_cells, up_genes)] *= up_multipliers
            if down_genes.size > 0:
                rate_per_gene[np.ix_(act_cells, down_genes)] /= down_multipliers

        if inact_cells.size > 0:
            if up_genes.size > 0:
                rate_per_gene[np.ix_(inact_cells, up_genes)] /= up_multipliers
            if down_genes.size > 0:
                rate_per_gene[np.ix_(inact_cells, down_genes)] *= down_multipliers

    # 5) Normalize within cell
    cell_totals = rate_per_gene.sum(axis=1, keepdims=True)
    cell_totals = np.where(cell_totals == 0, 1, cell_totals)
    rate_per_gene = rate_per_gene / cell_totals

    # 6) Scale to library size
    target_libsize = 10.0e4
    libsize = rng.lognormal(mean=np.log(target_libsize), sigma=0.35, size=n_cells)
    mean_mat = rate_per_gene * libsize[:, None]

    # 7) Gamma-Poisson / Poisson sampling
    expr_counts = rng.poisson(mean_mat).astype(np.int64)

    # 8) Dropout: add exactly enough extra zeros to match missing_percentage
    total_entries = expr_counts.size
    target_n_zeros = int(round(missing_ratio * total_entries))

    current_zero_mask = expr_counts == 0
    current_n_zeros = int(current_zero_mask.sum())
    current_missing_ratio = current_n_zeros / total_entries

    print(
        f"Zero percentage before explicit dropout: {100.0 * current_missing_ratio:.2f}%"
    )

    if target_n_zeros > current_n_zeros:
        n_extra_zeros_needed = target_n_zeros - current_n_zeros

        # choose only among currently nonzero entries
        nonzero_flat_idx = np.flatnonzero(~current_zero_mask.ravel())

        if n_extra_zeros_needed > nonzero_flat_idx.size:
            n_extra_zeros_needed = nonzero_flat_idx.size

        chosen_flat_idx = rng.choice(
            nonzero_flat_idx,
            size=n_extra_zeros_needed,
            replace=False,
        )

        expr_counts_flat = expr_counts.ravel()
        expr_counts_flat[chosen_flat_idx] = 0
        expr_counts = expr_counts_flat.reshape(expr_counts.shape)

    elif target_n_zeros < current_n_zeros:
        print(
            "Requested missing percentage is lower than the zero percentage already "
            "produced by Gamma-Poisson sampling. No dropout added."
        )

    final_n_zeros = int((expr_counts == 0).sum())
    final_missing_ratio = final_n_zeros / total_entries

    print(f"Target zero percentage: {100.0 * missing_ratio:.2f}%")
    print(f"Zero percentage after dropout: {100.0 * final_missing_ratio:.2f}%")

    expr = pd.DataFrame(expr_counts, index=cell_ids, columns=gene_names)
    expr.clip(lower=0.0, inplace=True)

    return expr


def generate_gene_expression_neg_binomial(
    n_cells: int,
    n_genes: int,
    tf_effect_factor: float,
    missing_percentage: int,
    random_seed: int,
    prior_dfs: pd.DataFrame,
    ground_truth_dfs: pd.DataFrame,
) -> pd.DataFrame:
    print("Generating gene expression data...")

    if n_cells <= 0 or n_genes <= 0:
        raise ValueError("n_cells and n_genes must be positive.")
    if prior_dfs.empty:
        raise ValueError("prior_df is empty.")
    if ground_truth_dfs.empty:
        raise ValueError("ground_truth_df is empty.")
    if not (0 <= missing_percentage <= 100):
        raise ValueError("missing_percentage must be in [0, 100].")

    required_cols = {"source", "target", "interaction", "weight"}
    if not required_cols.issubset(prior_dfs.columns):
        raise ValueError(f"prior_df must contain columns: {required_cols}")

    missing_ratio = float(missing_percentage) / 100.0
    rng = np.random.default_rng(random_seed)

    cell_ids = [f"Cell_{i + 1}" for i in range(n_cells)]
    gene_names = np.array([f"G_{i + 1}" for i in range(n_genes)], dtype=object)
    tf_names = list(ground_truth_dfs.columns)
    n_tfs = len(tf_names)

    # 1) Generate gene propensities (heavy-tailed baseline abundance)
    gene_prop = rng.lognormal(mean=0.0, sigma=2.0, size=n_genes)

    # 2) Initialize rate matrix with gene-specific Gamma overdispersion
    #    This is still your same Gamma-Poisson idea, but with per-gene theta.
    #    Higher-abundance genes get slightly larger theta -> a bit less noisy.
    theta_per_gene = np.clip(2.0 + 0.4 * np.log1p(gene_prop), 1.5, 8.0)
    gamma_shape = theta_per_gene[None, :]
    gamma_scale = gene_prop[None, :] / theta_per_gene[None, :]
    rate_per_gene = rng.gamma(
        shape=gamma_shape,
        scale=gamma_scale,
        size=(n_cells, n_genes),
    )

    # Small residual non-prior biological variation
    residual_noise = rng.lognormal(mean=0.0, sigma=0.15, size=(n_cells, n_genes))
    rate_per_gene *= residual_noise

    # 3) Build TF-target mappings
    gene_index_map = {g: i for i, g in enumerate(gene_names)}
    tf_name_to_idx = {name: i for i, name in enumerate(tf_names)}

    tf_to_up_idx = [[] for _ in range(n_tfs)]
    tf_to_down_idx = [[] for _ in range(n_tfs)]

    tf_to_up_weights = [[] for _ in range(n_tfs)]
    tf_to_down_weights = [[] for _ in range(n_tfs)]

    prior_tf = prior_dfs["source"].to_numpy()
    prior_target = prior_dfs["target"].to_numpy()
    prior_dir = prior_dfs["interaction"].to_numpy()
    prior_w = prior_dfs["weight"].to_numpy()

    for tf, tgt, direction, w in zip(prior_tf, prior_target, prior_dir, prior_w):
        gi = gene_index_map.get(tgt, None)
        ti = tf_name_to_idx.get(tf, None)
        if gi is None or ti is None:
            continue
        if direction == 1:
            tf_to_up_idx[ti].append(gi)
            tf_to_up_weights[ti].append(w)
        elif direction == -1:
            tf_to_down_idx[ti].append(gi)
            tf_to_down_weights[ti].append(w)

    # 4) Apply regulatory effects
    gt_states = ground_truth_dfs.to_numpy(copy=False)

    for ti in range(n_tfs):
        up_genes = np.array(tf_to_up_idx[ti], dtype=np.int64)
        up_w = np.array(tf_to_up_weights[ti], dtype=np.float64)

        down_genes = np.array(tf_to_down_idx[ti], dtype=np.int64)
        down_w = np.array(tf_to_down_weights[ti], dtype=np.float64)

        if up_genes.size == 0 and down_genes.size == 0:
            continue

        states = gt_states[:, ti]
        act_cells = np.where(states == 1)[0]
        inact_cells = np.where(states == -1)[0]

        if up_genes.size > 0:
            up_multipliers = 1.0 + (tf_effect_factor * up_w)
        if down_genes.size > 0:
            down_multipliers = 1.0 + (tf_effect_factor * down_w)

        if act_cells.size > 0:
            if up_genes.size > 0:
                rate_per_gene[np.ix_(act_cells, up_genes)] *= up_multipliers
            if down_genes.size > 0:
                rate_per_gene[np.ix_(act_cells, down_genes)] /= down_multipliers

        if inact_cells.size > 0:
            if up_genes.size > 0:
                rate_per_gene[np.ix_(inact_cells, up_genes)] /= up_multipliers
            if down_genes.size > 0:
                rate_per_gene[np.ix_(inact_cells, down_genes)] *= down_multipliers

    # 5) Normalize within each cell
    cell_totals = rate_per_gene.sum(axis=1, keepdims=True)
    cell_totals = np.where(cell_totals == 0, 1.0, cell_totals)
    rate_per_gene = rate_per_gene / cell_totals

    # 6) Scale to cell-specific library sizes
    target_libsize = 2.0e4
    libsize = rng.lognormal(mean=np.log(target_libsize), sigma=0.45, size=n_cells)

    # Add cell-specific capture efficiency
    capture_eff = rng.beta(a=8, b=2, size=n_cells)
    mean_mat = rate_per_gene * (libsize * capture_eff)[:, None]

    # 7) Final Poisson sampling
    expr_counts = rng.poisson(mean_mat).astype(np.int64)

    # 8) Add exact extra dropout, but preferentially on low counts
    total_entries = expr_counts.size
    target_n_zeros = int(round(missing_ratio * total_entries))

    current_zero_mask = expr_counts == 0
    current_n_zeros = int(current_zero_mask.sum())
    current_missing_ratio = current_n_zeros / total_entries

    print(
        f"Zero percentage before explicit dropout: {100.0 * current_missing_ratio:.2f}%"
    )

    if target_n_zeros > current_n_zeros:
        n_extra_zeros_needed = target_n_zeros - current_n_zeros

        expr_flat = expr_counts.ravel()
        zero_flat = current_zero_mask.ravel()
        nonzero_flat_idx = np.flatnonzero(~zero_flat)

        if n_extra_zeros_needed > nonzero_flat_idx.size:
            n_extra_zeros_needed = nonzero_flat_idx.size

        # Preferentially drop lower counts
        nonzero_vals = expr_flat[nonzero_flat_idx].astype(np.float64)

        # Larger counts get much smaller dropout probability
        # +1 avoids division by zero
        drop_scores = 1.0 / np.sqrt(nonzero_vals + 1.0)
        drop_probs = drop_scores / drop_scores.sum()

        chosen_flat_idx = rng.choice(
            nonzero_flat_idx,
            size=n_extra_zeros_needed,
            replace=False,
            p=drop_probs,
        )

        expr_flat[chosen_flat_idx] = 0
        expr_counts = expr_flat.reshape(expr_counts.shape)

    elif target_n_zeros < current_n_zeros:
        print(
            "Requested missing percentage is lower than the zero percentage already "
            "produced before explicit dropout. No extra dropout added."
        )

    final_n_zeros = int((expr_counts == 0).sum())
    final_missing_ratio = final_n_zeros / total_entries

    print(f"Target zero percentage: {100.0 * missing_ratio:.2f}%")
    print(f"Zero percentage after dropout: {100.0 * final_missing_ratio:.2f}%")

    expr = pd.DataFrame(expr_counts, index=cell_ids, columns=gene_names)
    expr.clip(lower=0, inplace=True)

    return expr


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(
            "Usage: uv run src/sim_data_generator_optimized.py simulated_data/parameters.json"
        )
        sys.exit(1)

    param_file = sys.argv[1]
    # param_file = "simulated_data/simulated_data.json"

    with open(param_file, "r") as f:
        params = json.load(f)
        print(params)

    if not params.get("random_seed"):
        rand_seed = secrets.randbelow(2**32)
        params["random_seed"] = rand_seed

    # ------------------------------------------
    # STEP 1. Generate Prior
    # ------------------------------------------
    prior_df = generate_prior_heavytail(
        n_genes=params["n_genes"],
        n_tfs=params["n_tfs"],
        weighted=params["weighted"],
        random_seed=params["random_seed"],
    )

    # ------------------------------------------
    # STEP 2. Generate Ground Truth
    # ------------------------------------------
    ground_truth_df = generate_ground_truth(
        n_cells=params["n_cells"],
        n_tfs=params["n_tfs"],
        n_clusters=params["n_clusters"],
        random_seed=params["random_seed"],
    )

    # ------------------------------------------
    # STEP 3. Generate Gene Expression Data
    # ------------------------------------------
    gene_exp = generate_gene_expression(
        n_cells=params["n_cells"],
        n_genes=params["n_genes"],
        tf_effect_factor=params["tf_effect_factor"],
        missing_percentage=params["missing_percentage"],
        random_seed=params["random_seed"],
        prior_dfs=prior_df,
        ground_truth_dfs=ground_truth_df,
    )

    # gene_exp = generate_gene_expression_neg_binomial(
    #     n_cells=params["n_cells"],
    #     n_genes=params["n_genes"],
    #     tf_effect_factor=params["tf_effect_factor"],
    #     missing_percentage=params["missing_percentage"],
    #     random_seed=params["random_seed"],
    #     prior_dfs=prior_df,
    #     ground_truth_dfs=ground_truth_df,
    # )

    prior_df["interaction"] = prior_df["interaction"].map(
        {1: "upregulates-expression", -1: "downregulates-expression"}
    )
    os.makedirs(params["output_dir"], exist_ok=True)
    prior_df.to_csv(
        f"{params['output_dir']}/{params['output_prior_file']}",
        sep="\t",
        index=False,
        header=True,
    )
    print(f"Wrote prior to {params['output_prior_file']}")

    ground_truth_df.to_csv(
        f"{params['output_dir']}/{params['output_ground_truth_file']}",
        sep="\t",
        index=True,
    )
    print(f"Wrote ground truth to {params['output_ground_truth_file']}")

    gene_exp.to_csv(
        f"{params['output_dir']}/{params['output_exp_file']}", sep="\t", index=True
    )
    print(f"Wrote gene expression to {params['output_exp_file']}")
