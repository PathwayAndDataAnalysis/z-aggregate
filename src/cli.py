import argparse
import logging
from pathlib import Path
from src.preprocessing import (
    read_adata_file,
    preprocess_adata,
    read_prior_network_file,
    compute_network_weights,
)
from src.WeightType import WeightType
from src.z_aggregate import run_z_aggregate


def main():
    parser = argparse.ArgumentParser(
        description="Z-Aggregate: TF Activity Prediction CLI"
    )
    parser.add_argument(
        "-ds",
        "--dataset",
        required=True,
        help="Path to expression data (h5ad, csv, txt)",
    )
    parser.add_argument(
        "-p",
        "--prior-type",
        required=True,
        help="Type of prior network (e.g., 'causalpath-priors', 'collectri', 'dorothea', 'ensemble-priors', or 'file_path')",
    )
    parser.add_argument("-o", "--output", required=True, help="Output directory")
    parser.add_argument(
        "--min-targets", type=int, default=5, help="Minimum targets per TF"
    )

    parser.add_argument(
        "--preprocess",
        dest="preprocess",
        action="store_true",
        default=True,
        help="Enable preprocessing",
    )
    parser.add_argument(
        "--no-preprocess",
        dest="preprocess",
        action="store_false",
        help="Disable preprocessing",
    )

    parser.add_argument(
        "--min-genes", type=int, default=1000, help="Min genes per cell"
    )
    parser.add_argument("--min-cells", type=int, default=10, help="Min cells per gene")
    parser.add_argument(
        "--max-mt-pct", type=float, default=20.0, help="Max mitochondrial percentage"
    )

    parser.add_argument(
        "--weight-type",
        choices=[w.value for w in WeightType],
        default=WeightType.UNIFORM.value,
        help="Weighting strategy",
    )
    parser.add_argument(
        "--output-format",
        choices=["tsv", "csv", "h5ad", "both"],
        default="both",
        help="Output file format",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Increase output verbosity"
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("h5py").setLevel(logging.WARNING)
    logging.getLogger("numba").setLevel(logging.WARNING)

    # 1. Load Data
    adata = read_adata_file(args.dataset)

    # 2. Preprocess
    if args.preprocess:
        adata = preprocess_adata(
            adata,
            min_genes=args.min_genes,
            min_cells=args.min_cells,
            max_mt_pct=args.max_mt_pct,
        )

    # 3. Load & Weight Network
    prior_df = read_prior_network_file(args.prior_type)
    prior_df = compute_network_weights(
        adata, prior_df, weight_type=WeightType(args.weight_type)
    )

    # 4. Run Algorithm
    scores, pvalues = run_z_aggregate(adata, prior_df, min_targets=args.min_targets)
    scores.sort_index(axis=1, inplace=True)
    pvalues.sort_index(axis=1, inplace=True)

    # 5. Save Results
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    result_prefix = Path(args.dataset).stem
    prior_prefix = Path(args.prior_type).stem if args.prior_type not in ("causalpath-priors", "collectri", "dorothea", "ensemble-priors", 'file_path') else args.prior_type

    if args.output_format in ("tsv", "csv", "both"):
        format_to_write = (
            "tsv" if args.output_format == "both" else [args.output_format]
        )
        sep = "\t" if format_to_write == "tsv" else ","

        scores_file_name = (
            out_dir
            / f"{result_prefix}_{prior_prefix}_z_agg_scores.{format_to_write}"
        )
        pvalues_file_name = (
            out_dir
            / f"{result_prefix}_{prior_prefix}_z_agg_pvalues.{format_to_write}"
        )
        scores.to_csv(scores_file_name, sep=sep)
        pvalues.to_csv(pvalues_file_name, sep=sep)

        logging.info(
            f"Saved z-aggregate scores (cells={scores.shape[0]}, TFs={scores.shape[1]}) "
            f"to {scores_file_name}"
        )
        logging.info(
            f"Saved z-aggregate p-values (cells={pvalues.shape[0]}, TFs={pvalues.shape[1]}) "
            f"to {pvalues_file_name}"
        )

    if args.output_format in ("h5ad", "both"):
        adata_out = adata.copy()

        score_key = "z_agg_scores"
        pval_key = "z_agg_pvalues"

        adata_out.obsm[score_key] = scores
        adata_out.obsm[pval_key] = pvalues

        h5ad_filename = (
            out_dir / f"{result_prefix}_{prior_prefix}_z_agg_results.h5ad"
        )
        adata_out.write_h5ad(h5ad_filename)

        logging.info(
            f"Saved AnnData object to {h5ad_filename}. "
            f"Added .obsm keys: '{score_key}', '{pval_key}'"
        )


if __name__ == "__main__":
    main()
