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
    parser = argparse.ArgumentParser(description="Z-Aggregate: TF Activity Prediction CLI")
    parser.add_argument("-ds", "--dataset", required=True, help="Path to expression data (h5ad, csv, txt)")
    parser.add_argument("-p", "--priors", required=True, help="Path to prior network file")
    parser.add_argument("-o", "--output", required=True, help="Output directory")
    parser.add_argument("--min-targets", type=int, default=5, help="Minimum targets per TF")

    parser.add_argument(
        "--preprocess", dest="preprocess", action="store_true", default=True, help="Enable preprocessing"
    )
    parser.add_argument("--no-preprocess", dest="preprocess", action="store_false", help="Disable preprocessing")

    parser.add_argument("--min-genes", type=int, default=1000, help="Min genes per cell")
    parser.add_argument("--min-cells", type=int, default=10, help="Min cells per gene")
    parser.add_argument("--max-mt-pct", type=float, default=20.0, help="Max mitochondrial percentage")

    parser.add_argument(
        "--weight-type",
        choices=[w.value for w in WeightType],
        default=WeightType.UNIFORM.value,
        help="Weighting strategy",
    )
    parser.add_argument("--output-format", choices=["tsv", "h5ad", "both"], default="both", help="Output file format")
    parser.add_argument("-v", "--verbose", action="store_true", help="Increase output verbosity")

    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s: %(message)s")

    # 1. Load Data
    adata = read_adata_file(args.dataset)

    # 2. Preprocess
    if args.preprocess:
        adata = preprocess_adata(adata, min_genes=args.min_genes, min_cells=args.min_cells, max_mt_pct=args.max_mt_pct)

    # 3. Load & Weight Network
    priors = read_prior_network_file(args.priors)
    priors = compute_network_weights(adata, priors, weight_type=WeightType(args.weight_type))

    # 4. Run Algorithm
    scores, pvalues = run_z_aggregate(adata, priors, min_targets=args.min_targets)

    # 5. Save Results
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    scores.to_csv(out_dir / "z_aggregate_scores.tsv", sep="\t")
    pvalues.to_csv(out_dir / "z_aggregate_pvalues.tsv", sep="\t")

    if args.output_format in ("h5ad", "both"):
        adata_out = adata.copy()
        adata_out.obsm["z_aggregate_scores"] = scores
        adata_out.obsm["z_aggregate_pvalues"] = pvalues
        adata_out.write_h5ad(out_dir / "z_aggregate_results.h5ad")
        logging.info(f"Results saved to {out_dir}")


if __name__ == "__main__":
    main()
