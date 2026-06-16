import os
import numpy as np
import pandas as pd
import argparse
import json

def generate_prior(
    n_genes: int,
    n_tfs: int,
    total_mu: float,
    total_sigma: float,
    up_probability: float,
    random_seed: int,
) -> pd.DataFrame:
    if n_genes <= 0 or n_tfs <= 0:
        raise ValueError("n_genes and n_tfs must be positive.")
    if total_sigma <= 0:
        raise ValueError("total_sigma must be > 0.")
    if not 0 <= up_probability <= 1:
        raise ValueError("up_probability must be between 0 and 1.")

    rng = np.random.default_rng(random_seed)

    tfs = np.array([f"TF_{i + 1}" for i in range(n_tfs)], dtype=object)
    genes = np.array([f"G_{i + 1}" for i in range(n_genes)], dtype=object)

    rows = []

    total_targets = np.rint(
        rng.lognormal(mean=total_mu, sigma=total_sigma, size=n_tfs)
    ).astype(int)

    total_targets = np.clip(total_targets, 1, n_genes)

    for tf, n_total in zip(tfs, total_targets):
        target_idx = rng.choice(
            n_genes,
            size=n_total,
            replace=False,
        )

        interactions = rng.choice(
            [1, -1],
            size=n_total,
            p=[up_probability, 1.0 - up_probability],
        )

        for gi, interaction in zip(target_idx, interactions):
            row = {
                "source": tf,
                "target": genes[gi],
                "interaction": int(interaction),
            }
            row["weight"] = 1.0
            rows.append(row)

    return pd.DataFrame(rows)


def generate_noisy_prior(
    prior_df: pd.DataFrame,
    n_genes: int,
    prior_noise_percentage: float,
    random_seed: int | None = None,
) -> pd.DataFrame:
    if prior_df.empty:
        raise ValueError("prior_df is empty.")
    if n_genes <= 0:
        raise ValueError("n_genes must be positive.")
    if not 0 <= prior_noise_percentage <= 100:
        raise ValueError("prior_noise_percentage must be in [0, 100].")

    required_cols = {"source", "target", "interaction", "weight"}
    if not required_cols.issubset(prior_df.columns):
        raise ValueError(f"prior_df must contain columns: {required_cols}")

    rng = np.random.default_rng(random_seed)
    noisy = prior_df.copy().reset_index(drop=True)

    if prior_noise_percentage == 0:
        print("Prior noise: 0.00% | replaced edges: 0")
        return noisy

    all_genes = np.array([f"G_{i + 1}" for i in range(n_genes)], dtype=object)

    n_edges = len(noisy)
    n_to_replace = int(round((prior_noise_percentage / 100.0) * n_edges))

    original_targets_by_tf = (
        prior_df.groupby("source")["target"]
        .apply(lambda x: set(x.astype(str)))
        .to_dict()
    )

    blocked_targets_by_tf = {
        tf: set(targets) for tf, targets in original_targets_by_tf.items()
    }

    changed = 0
    selected_idx = rng.permutation(noisy.index.to_numpy())

    for idx in selected_idx:
        if changed >= n_to_replace:
            break

        tf = str(noisy.at[idx, "source"])
        blocked = blocked_targets_by_tf.get(tf, set())

        available = np.array([g for g in all_genes if g not in blocked], dtype=object)
        if available.size == 0:
            continue

        new_target = str(rng.choice(available))
        noisy.at[idx, "target"] = new_target

        blocked_targets_by_tf.setdefault(tf, set()).add(new_target)
        changed += 1

    if changed < n_to_replace:
        print(
            f"Warning: requested {n_to_replace} noisy edges, but only replaced {changed}. "
            "Some TFs had no available non-target genes left."
        )

    print(
        f"Prior noise: {prior_noise_percentage:.2f}% | "
        f"requested edges: {n_to_replace} | replaced edges: {changed}"
    )

    return noisy


def generate_ground_truth(
    n_cells: int,
    n_tfs: int,
    ground_truth_active_inactive_prob: float,
    random_seed: int,
) -> pd.DataFrame:
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


def generate_gene_expression(
    n_cells: int,
    n_genes: int,
    tf_effect_factor: float,
    missing_percentage: int,
    target_libsize: float,
    random_seed: int,
    gene_propensity_sigma: float,
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
    gene_prop = rng.lognormal(mean=0, sigma=gene_propensity_sigma, size=n_genes)

    # 2) Initialize rate matrix (Gamma) to create cell-to-cell variability
    theta = 5
    gamma_shape = theta
    gamma_scale = gene_prop / theta
    rate_per_gene = rng.gamma(shape=gamma_shape, scale=gamma_scale, size=(n_cells, n_genes))

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

    print(f"Zero percentage before explicit dropout: {100.0 * current_missing_ratio:.2f}%")

    if target_n_zeros > current_n_zeros:
        n_extra_zeros_needed = target_n_zeros - current_n_zeros
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


if __name__ == "__main__":
    # params = {
    #     "output_dir": "simulated_data/",
    #     "output_exp_file": "simulated_scRNASeq.tsv",
    #     "output_prior_file": "simulated_prior_network.tsv",
    #     "output_noisy_prior_file": "simulated_noisy_prior_network.tsv",
    #     "output_ground_truth_file": "simulated_ground_truth.tsv",
    #     "n_cells": 1000,
    #     "n_genes": 1000,
    #     "n_tfs": 100,
    #     "missing_percentage": 80.0,
    #     "prior_noise_percentage": 40.0,
    #     "target_libsize": 1e4,
    #     "ground_truth_active_inactive_prob": 0.1,
    #     "up_probability": 0.90,
    #     "random_seed": 42,
    #     "tf_effect_factor": 2,
    # }

    parser = argparse.ArgumentParser()
    parser.add_argument("--params-file", required=True)
    args = parser.parse_args()

    with open(args.params_file, "r") as f:
        params = json.load(f)

    # ------------------------------------------
    # STEP 1. Generate Prior
    # ------------------------------------------
    prior_df = generate_prior(
        n_genes=params["n_genes"],
        n_tfs=params["n_tfs"],
        total_mu=2.0,
        total_sigma=1.5,
        up_probability=params["up_probability"],
        random_seed=params["random_seed"],
    )

    # ------------------------------------------
    # STEP 2. Generate Noisy Prior
    # ------------------------------------------
    noisy_prior_df = generate_noisy_prior(
        prior_df=prior_df,
        n_genes=params["n_genes"],
        prior_noise_percentage=params["prior_noise_percentage"],
        random_seed=params["random_seed"],
    )

    # ------------------------------------------
    # STEP 3. Generate Ground Truth
    # ------------------------------------------
    ground_truth_df = generate_ground_truth(
        n_cells=params["n_cells"],
        n_tfs=params["n_tfs"],
        ground_truth_active_inactive_prob=params["ground_truth_active_inactive_prob"],
        random_seed=params["random_seed"],
    )

    # ------------------------------------------
    # STEP 4. Generate Gene Expression Data
    # ------------------------------------------
    gene_exp = generate_gene_expression(
        n_cells=params["n_cells"],
        n_genes=params["n_genes"],
        tf_effect_factor=params["tf_effect_factor"],
        missing_percentage=params["missing_percentage"],
        random_seed=params["random_seed"],
        target_libsize=params["target_libsize"],
        gene_propensity_sigma=params["gene_propensity_sigma"],
        prior_dfs=prior_df,
        ground_truth_dfs=ground_truth_df,
    )

    interaction_map = {1: "upregulates-expression", -1: "downregulates-expression"}

    prior_save_df = prior_df.copy()
    prior_save_df["interaction"] = prior_save_df["interaction"].map(interaction_map)

    noisy_prior_save_df = noisy_prior_df.copy()
    noisy_prior_save_df["interaction"] = noisy_prior_save_df["interaction"].map(
        interaction_map
    )

    os.makedirs(params["output_dir"], exist_ok=True)
    prior_save_df.to_csv(
        f"{params['output_dir']}/{params['output_prior_file']}",
        sep="\t",
        index=False,
        header=True,
    )
    noisy_prior_save_df.to_csv(
        f"{params['output_dir']}/{params['output_noisy_prior_file']}",
        sep="\t",
        index=False,
        header=True,
    )
    ground_truth_df.to_csv(
        f"{params['output_dir']}/{params['output_ground_truth_file']}",
        sep="\t",
        index=True,
    )
    gene_exp.to_csv(
        f"{params['output_dir']}/{params['output_exp_file']}", sep="\t", index=True
    )
    print("All files saved to:", params["output_dir"])





