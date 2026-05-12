import argparse
import os

import anndata
import pandas as pd
import scanpy as sc
from sklearn import metrics


def compute_clustering_metrics(adata, label_key, cluster_key):
    labels = adata.obs[label_key].astype(str)
    clusters = adata.obs[cluster_key].astype(str)
    return {
        "ARI": metrics.adjusted_rand_score(labels, clusters),
        "AMI": metrics.adjusted_mutual_info_score(labels, clusters),
        "NMI": metrics.normalized_mutual_info_score(labels, clusters),
        "Homogeneity": metrics.homogeneity_score(labels, clusters),
        "FMI": metrics.fowlkes_mallows_score(labels, clusters),
        "n_clusters": int(adata.obs[cluster_key].nunique()),
    }


def parse_resolutions(value):
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Leiden clustering metrics for QVAE representations."
    )
    parser.add_argument("--h5ad", required=True)
    parser.add_argument("--rep-key", default="X_qvae")
    parser.add_argument("--label-key", default="final_annotation")
    parser.add_argument("--n-neighbors", type=int, default=15)
    parser.add_argument("--resolutions", default="0.2,0.4,0.6,0.8,1.0,1.2,1.5,2.0")
    parser.add_argument("--algorithm", default="leiden", choices=["leiden", "louvain"])
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    print("Loading data...")
    adata = anndata.read_h5ad(args.h5ad)
    print("Data loaded successfully.")
    if args.rep_key not in adata.obsm:
        raise KeyError(f"{args.rep_key!r} not found in adata.obsm")
    if args.label_key not in adata.obs:
        raise KeyError(f"{args.label_key!r} not found in adata.obs")

    neighbors_key = f"{args.rep_key}_neighbors"
    sc.pp.neighbors(
        adata,
        use_rep=args.rep_key,
        n_neighbors=args.n_neighbors,
        key_added=neighbors_key,
    )

    rows = []
    for resolution in parse_resolutions(args.resolutions):
        cluster_key = f"{args.algorithm}_r{resolution:g}"
        if args.algorithm == "leiden":
            sc.tl.leiden(
                adata,
                resolution=resolution,
                key_added=cluster_key,
                neighbors_key=neighbors_key,
                random_state=args.random_state,
            )
        else:
            sc.tl.louvain(
                adata,
                resolution=resolution,
                key_added=cluster_key,
                neighbors_key=neighbors_key,
                random_state=args.random_state,
            )
        row = {
            "rep_key": args.rep_key,
            "label_key": args.label_key,
            "algorithm": args.algorithm,
            "n_neighbors": args.n_neighbors,
            "resolution": resolution,
        }
        row.update(compute_clustering_metrics(adata, args.label_key, cluster_key))
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("ARI", ascending=False)
    print(df.to_string(index=False))

    if args.output_csv is None:
        out_dir = os.path.dirname(args.h5ad) or "."
        args.output_csv = os.path.join(out_dir, f"{args.rep_key}_clustering_metrics.csv")
    df.to_csv(args.output_csv, index=False)
    print(f"Saved metrics to: {args.output_csv}")

    best = df.iloc[0]
    print(
        "Best by ARI: "
        f"resolution={best['resolution']}, "
        f"ARI={best['ARI']:.4f}, "
        f"NMI={best['NMI']:.4f}, "
        f"AMI={best['AMI']:.4f}, "
        f"n_clusters={int(best['n_clusters'])}"
    )


if __name__ == "__main__":
    main()
