import json
import os
import sys
import numpy as np
import pandas as pd
import secrets


def generate_prior_poisson(
    n_genes: int,
    n_tfs: int,
    lambda_param: float,
    overlap_alpha_factor: int,
    weighted: bool = True,
    random_seed: int = None,
) -> pd.DataFrame:
    print("Generating prior data (Poisson targets)...")

    if n_genes <= 0 or n_tfs <= 0:
        raise ValueError("n_genes and n_tfs must be positive.")
    if lambda_param <= 0:
        raise ValueError("lambda_param must be > 0.")

    rng = np.random.default_rng(random_seed)

    tfs = [f"TF_{i + 1}" for i in range(n_tfs)]
    genes = np.array([f"G_{i + 1}" for i in range(n_genes)], dtype=object)

    rows_tf = []
    rows_target = []
    rows_interaction = []

    gene_indices = np.arange(n_genes)

    # 1. Draw up/down target counts per TF from Poisson(λ)
    targets = rng.poisson(lam=lambda_param, size=n_tfs)
    n_up_targets = rng.poisson(lam=targets / 2)
    n_down_targets = targets - n_up_targets

    for i, tf in enumerate(tfs):
        n_up = int(n_up_targets[i])
        n_down = int(n_down_targets[i])

        # Permutation to avoid self-overlap within the same TF
        perm = rng.permutation(gene_indices)

        # Clip counts to available genes
        if n_up > n_genes:
            n_up = n_genes
        if n_down > n_genes - n_up:
            n_down = n_genes - n_up

        up_idx = perm[:n_up]
        down_idx = perm[n_up : n_up + n_down]

        if n_up:
            rows_tf.extend([tf] * n_up)
            rows_target.extend(genes.take(up_idx))
            rows_interaction.extend([1] * n_up)  # 1 represents upregulation
        if n_down:
            rows_tf.extend([tf] * n_down)
            rows_target.extend(genes.take(down_idx))
            rows_interaction.extend([-1] * n_down)  # -1 represents downregulation

    df = pd.DataFrame(
        {"source": rows_tf, "target": rows_target, "interaction": rows_interaction}
    )

    # 2. Handle Overlap Logic (Rebuilds the network structure if alpha > 0)
    if overlap_alpha_factor > 0:
        prior_overlapped_updated = {}
        overlapped_prior_net = []

        unique_tfs = df["source"].unique()
        if len(unique_tfs) >= 2:
            group_size = 5  # adjust if desired
            prior_grouped = df.groupby("source").size().to_dict()

            for i in range(0, len(unique_tfs), group_size):
                current_tfs = unique_tfs[i : i + group_size]

                # Pool all targets from this group of TFs
                current_targets = (
                    df[df["source"].isin(current_tfs)]["target"].unique().tolist()
                )

                # Assign random direction (+1/-1) to the pooled targets
                targets_directions = rng.choice([1, -1], size=len(current_targets))

                rng.shuffle(current_targets)
                current_targets_len = len(current_targets)

                for tf in current_tfs:
                    added_targets = prior_overlapped_updated.get(tf, [])
                    original_target_count = prior_grouped.get(tf, 0)

                    mean = (current_targets_len - 1) / 2
                    std = current_targets_len / (
                        len(current_tfs) * overlap_alpha_factor
                    )

                    # Re-sample targets based on Normal distribution to create hubs
                    while len(added_targets) < original_target_count:
                        random_idx = int(rng.normal(mean, std))

                        if 0 <= random_idx < current_targets_len:
                            target_candidate = current_targets[random_idx]

                            # Ensure unique (TF, Target) pairs
                            if target_candidate not in added_targets:
                                candidate_direction = targets_directions[random_idx]
                                added_targets.append(target_candidate)
                                overlapped_prior_net.append(
                                    {
                                        "source": tf,
                                        "target": target_candidate,
                                        "interaction": candidate_direction,
                                    }
                                )

                    prior_overlapped_updated[tf] = added_targets

            # Replace the original dataframe with the overlapped version
            df = pd.DataFrame(overlapped_prior_net)

    # 3. Clean duplicates based on topology (Source + Target)
    # We prioritize keeping the first occurrence before assigning weights
    df = df.drop_duplicates(subset=["source", "target"], keep="first").reset_index(
        drop=True
    )

    # 4. Generate Random Weights (0-1)
    if weighted:
        df["weight"] = rng.uniform(0.0, 2.0, size=len(df))
    else:
        df["weight"] = 1.0

    return df


def generate_ground_truth(
    n_cells: int,
    n_tfs: int,
    ground_truth_active_inactive_prob: float = 0.1,
    random_seed: int = None,
) -> pd.DataFrame:
    """
    Create a (n_cells x n_tfs) matrix initialized to 0.
    Randomly set a subset to +1 (activated) and a disjoint subset to -1 (inactivated).
    No cell–TF pair gets both; conflicts are resolved by random tie-break.
    """
    if n_cells <= 0 or n_tfs <= 0:
        raise ValueError("n_cells and n_tfs must be positive.")
    if not (
        0.0 <= ground_truth_active_inactive_prob <= 1.0
        and 0.0 <= ground_truth_active_inactive_prob <= 1.0
    ):
        raise ValueError("activation_prob and inactivation_prob must be in [0, 1].")

    rng = np.random.default_rng(random_seed)

    gt = np.zeros((n_cells, n_tfs), dtype=np.int8)

    act_mask = rng.random((n_cells, n_tfs)) < ground_truth_active_inactive_prob
    inact_mask = rng.random((n_cells, n_tfs)) < ground_truth_active_inactive_prob

    overlap = act_mask & inact_mask
    if overlap.any():
        keep_act = rng.random(overlap.sum()) < 0.5
        ov_i, ov_j = np.where(overlap)
        act_keep_idx = (ov_i[keep_act], ov_j[keep_act])
        inact_keep_idx = (ov_i[~keep_act], ov_j[~keep_act])
        inact_mask[act_keep_idx] = False
        act_mask[inact_keep_idx] = False

    gt[act_mask] = 1
    gt[inact_mask] = -1

    cell_ids = [f"Cell_{i + 1}" for i in range(n_cells)]
    tf_names = [f"TF_{j + 1}" for j in range(n_tfs)]
    return pd.DataFrame(gt, index=cell_ids, columns=tf_names)


def generate_gene_expression_neg_binomial(
    n_cells: int,
    n_genes: int,
    include_tfs_in_expression: bool,
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
    # Using a heavy tail distribution typical for gene expression
    gene_prop = rng.lognormal(mean=0, sigma=3, size=n_genes)

    # 2) Initialize rate matrix (Gamma) to create cell-to-cell variability
    theta = 5  # dispersion parameter
    gamma_shape = theta
    gamma_scale = gene_prop / theta
    rate_per_gene = rng.gamma(
        shape=gamma_shape, scale=gamma_scale, size=(n_cells, n_genes)
    )

    # 3) Build TF-target mappings (storing both Index and Weight)
    gene_index_map = {g: i for i, g in enumerate(gene_names)}
    tf_name_to_idx = {name: i for i, name in enumerate(tf_names)}

    # Structure: tf_index -> list of target_gene_indices
    tf_to_up_idx = [[] for _ in range(n_tfs)]
    tf_to_down_idx = [[] for _ in range(n_tfs)]

    # Structure: tf_index -> list of weights corresponding to the indices above
    tf_to_up_weights = [[] for _ in range(n_tfs)]
    tf_to_down_weights = [[] for _ in range(n_tfs)]

    prior_tf = prior_dfs["source"].to_numpy()
    prior_target = prior_dfs["target"].to_numpy()
    prior_dir = prior_dfs["interaction"].to_numpy()  # 1 or -1
    prior_w = prior_dfs["weight"].to_numpy()  # 0.0 to 1.0

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
    # We use vectorization. Multiplier = factor ^ weight
    # If weight is 1.0, we multiply by full factor.
    # If weight is 0.0, we multiply by 1 (no change).
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

        # Pre-calculate multipliers for this TF's targets
        if up_genes.size > 0:
            up_multipliers = 1 + (tf_effect_factor * up_w)
        if down_genes.size > 0:
            down_multipliers = 1 + (tf_effect_factor * down_w)

        # -- TF Activated (State = 1) --
        if act_cells.size > 0:
            if up_genes.size > 0:
                rate_per_gene[np.ix_(act_cells, up_genes)] *= up_multipliers
            if down_genes.size > 0:
                rate_per_gene[np.ix_(act_cells, down_genes)] /= down_multipliers

        # -- TF Inactivated (State = -1) --
        if inact_cells.size > 0:
            if up_genes.size > 0:
                rate_per_gene[np.ix_(inact_cells, up_genes)] /= up_multipliers
            if down_genes.size > 0:
                rate_per_gene[np.ix_(inact_cells, down_genes)] *= down_multipliers

    # 5) For each cell, normalize sum to 1
    cell_totals = rate_per_gene.sum(axis=1, keepdims=True)
    cell_totals = np.where(cell_totals == 0, 1, cell_totals)  # Avoid division by zero
    rate_per_gene = rate_per_gene / cell_totals

    # 6) Normalize to library sizes
    target_libsize = 10.0e4
    libsize = rng.lognormal(mean=np.log(target_libsize), sigma=0.35, size=n_cells)
    mean_mat = rate_per_gene * libsize[:, None]

    # 7) Apply Poisson to get final integer counts (Negative Binomial approx via Gamma-Poisson)
    expr_counts = rng.poisson(mean_mat).astype(np.int64)

    # 8) Dropout (zero-inflation)
    n_zeros = np.sum(expr_counts == 0)
    current_missing_ratio = n_zeros / expr_counts.size
    print(
        f"Zero percentage before explicit dropout: {100.0 * current_missing_ratio:.2f}%"
    )

    # Repeat until a desired missing ratio is achieved
    # if missing_ratio > current_missing_ratio:
    drop_mask = rng.random(size=expr_counts.shape) < missing_ratio
    expr_counts = expr_counts * (~drop_mask)
    n_zeros = np.sum(expr_counts == 0)
    current_missing_ratio = n_zeros / expr_counts.size
    print("Zero percentage after dropout:", 100.0 * current_missing_ratio)
    # else:
    #     print("No additional dropout applied.")

    expr = pd.DataFrame(expr_counts, index=cell_ids, columns=gene_names)
    expr.clip(lower=0.0)

    # 9) Include TFs as expression features (optional)
    if include_tfs_in_expression:
        tf_names_arr = np.array(tf_names, dtype=object)
        tf_baseline = rng.lognormal(
            mean=1.0, sigma=0.75, size=(n_cells, len(tf_names_arr))
        )

        # Simple dropout for TFs
        tf_drop = rng.random(size=tf_baseline.shape) < 0.5
        tf_baseline = tf_baseline * (~tf_drop)

        tf_df = pd.DataFrame(
            tf_baseline.astype(np.int64), index=cell_ids, columns=tf_names_arr
        )
        expr = pd.concat([expr, tf_df], axis=1)

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
    prior_df = generate_prior_poisson(
        n_genes=params["n_genes"],
        n_tfs=params["n_tfs"],
        lambda_param=params["average_number_of_targets_per_tf"],
        overlap_alpha_factor=params["overlap_alpha_factor"],
        weighted=params["weighted"],
        random_seed=params["random_seed"],
    )

    # ------------------------------------------
    # STEP 2. Generate Ground Truth
    # ------------------------------------------
    ground_truth_df = generate_ground_truth(
        n_cells=params["n_cells"],
        n_tfs=params["n_tfs"],
        ground_truth_active_inactive_prob=params["ground_truth_active_inactive_prob"],
        random_seed=params["random_seed"],
    )

    # ------------------------------------------
    # STEP 3. Generate Gene Expression Data
    # ------------------------------------------
    gene_exp = generate_gene_expression_neg_binomial(
        n_cells=params["n_cells"],
        n_genes=params["n_genes"],
        include_tfs_in_expression=params["include_tfs_in_expression"],
        tf_effect_factor=params["tf_effect_factor"],
        missing_percentage=params["missing_percentage"],
        random_seed=params["random_seed"],
        prior_dfs=prior_df,
        ground_truth_dfs=ground_truth_df,
    )

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
