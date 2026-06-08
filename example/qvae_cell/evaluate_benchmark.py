"""Optional benchmark metrics for qvae_cell embeddings.

This file is new code for the qvae_cell example. The original example only had
``evaluate_clustering.py`` for Leiden clustering metrics.

Metric groups:
    clustering: Checks whether Leiden clusters match known cell-type labels.
    scIB: Measures biological conservation and batch correction quality.
    scGraph: Measures graph/batch/cell-type structure with the scGraph package.
    classification: Tests whether labels are predictable from the embedding.
    DPT: Computes pseudotime and optional trajectory agreement with PCA.
"""

import argparse
import os
import tempfile

import anndata
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import kendalltau
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

VALID_METRICS = ("clustering", "classification", "scib", "scgraph", "dpt")


def parse_resolutions(value):
    """Parse a comma-separated list of Leiden resolutions.

    Args:
        value: Comma-separated string, for example ``"0.2,0.4,0.8"``.

    Returns:
        A list of floating-point resolution values.
    """
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_metrics(value):
    """Parse benchmark metric groups from a comma-separated option.

    Args:
        value: Comma-separated metric groups. Use ``all`` to run every group.

    Returns:
        A list of metric group names.

    Raises:
        ValueError: If an unknown metric group is requested.
    """
    items = [item.strip().lower() for item in value.split(",") if item.strip()]
    if not items:
        raise ValueError("--metrics cannot be empty")
    if items == ["all"]:
        return list(VALID_METRICS)

    unknown = sorted(set(items) - set(VALID_METRICS))
    if unknown:
        raise ValueError(
            f"Unknown metric group(s): {', '.join(unknown)}. "
            f"Valid values: {', '.join(VALID_METRICS)}, all"
        )
    return items


def require_key(mapping, key, name):
    """Ensure that a dict-like AnnData container contains a key.

    Args:
        mapping: Dict-like container such as ``adata.obs`` or ``adata.obsm``.
        key: Required key name.
        name: Human-readable container name used in the error message.

    Raises:
        KeyError: If ``key`` is missing.
    """
    if key not in mapping:
        raise KeyError(f"{key!r} not found in {name}")


def compute_clustering_metrics(adata, label_key, cluster_key):
    """Compute agreement between known labels and cluster assignments.

    Args:
        adata: AnnData object containing labels and clusters in ``obs``.
        label_key: ``adata.obs`` column with reference cell-type labels.
        cluster_key: ``adata.obs`` column with predicted cluster labels.

    Returns:
        Dictionary with ARI, AMI, NMI, Homogeneity, FMI, and cluster count.
    """
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


def evaluate_clustering(
    adata, rep_key, label_key, resolutions, n_neighbors, random_state
):
    """Run Leiden clustering on an embedding and score each resolution.

    This metric group answers: does the learned embedding separate known cell
    types into coherent unsupervised clusters?

    Args:
        adata: AnnData object with an embedding in ``adata.obsm``.
        rep_key: Embedding key, such as ``X_qvae``.
        label_key: ``adata.obs`` column with reference cell-type labels.
        resolutions: Leiden resolutions to scan.
        n_neighbors: Number of neighbors used for graph construction.
        random_state: Random seed for Leiden clustering.

    Returns:
        DataFrame sorted by ARI in descending order.
    """
    neighbors_key = f"{rep_key}_neighbors"
    sc.pp.neighbors(
        adata,
        use_rep=rep_key,
        n_neighbors=n_neighbors,
        key_added=neighbors_key,
    )

    rows = []
    for resolution in resolutions:
        cluster_key = f"leiden_r{resolution:g}"
        sc.tl.leiden(
            adata,
            resolution=resolution,
            key_added=cluster_key,
            neighbors_key=neighbors_key,
            random_state=random_state,
        )
        row = {
            "metric_group": "clustering",
            "rep_key": rep_key,
            "label_key": label_key,
            "resolution": resolution,
            "n_neighbors": n_neighbors,
        }
        row.update(compute_clustering_metrics(adata, label_key, cluster_key))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("ARI", ascending=False)


def evaluate_scib(adata, rep_key, batch_key, label_key, n_jobs):
    """Run scIB biological conservation and batch correction metrics.

    scIB is a standard single-cell integration benchmark. It reports metrics
    such as isolated labels, KMeans NMI/ARI, silhouette label, cLISI, iLISI,
    KBET, graph connectivity, PCR comparison, biological conservation, batch
    correction, and total score.

    Args:
        adata: AnnData object with the embedding in ``adata.obsm``.
        rep_key: Embedding key to benchmark.
        batch_key: ``adata.obs`` column with batch labels.
        label_key: ``adata.obs`` column with cell-type labels.
        n_jobs: Number of parallel jobs used by scIB.

    Returns:
        One-row DataFrame with scIB scores for ``rep_key``.

    Raises:
        RuntimeError: If the optional ``scib_metrics`` dependency is missing.
    """
    try:
        from scib_metrics.benchmark import BatchCorrection, Benchmarker, BioConservation
    except ImportError as exc:
        raise RuntimeError("scib_metrics is not installed") from exc

    benchmarker = Benchmarker(
        adata,
        batch_key=batch_key,
        label_key=label_key,
        embedding_obsm_keys=[rep_key],
        batch_correction_metrics=BatchCorrection(),
        bio_conservation_metrics=BioConservation(),
        n_jobs=n_jobs,
    )
    benchmarker.benchmark()
    result = benchmarker.get_results(min_max_scale=False)
    result = result.loc[[rep_key]].copy()
    result.insert(0, "metric_group", "scib")
    result.insert(1, "rep_key", rep_key)
    return result.reset_index(drop=True)


def evaluate_scgraph(
    adata,
    rep_key,
    batch_key,
    label_key,
    output_dir,
    trim_rate,
    thres_batch,
    thres_celltype,
    only_umap,
):
    """Run scGraph graph-structure evaluation.

    scGraph evaluates neighborhood graph structure with respect to batches and
    cell-type labels. The package expects an h5ad path, so this function writes
    a temporary h5ad and deletes it after evaluation.

    Args:
        adata: AnnData object to evaluate.
        rep_key: Embedding key used for labeling the result rows.
        batch_key: ``adata.obs`` column with batch labels.
        label_key: ``adata.obs`` column with cell-type labels.
        output_dir: Directory for the temporary h5ad file.
        trim_rate: Robust trimming rate passed to scGraph.
        thres_batch: Minimum cell count per batch passed to scGraph.
        thres_celltype: Minimum cell count per cell type passed to scGraph.
        only_umap: Whether scGraph should evaluate only UMAP embeddings.

    Returns:
        DataFrame returned by scGraph with metadata columns added.

    Raises:
        RuntimeError: If the optional ``scgraph`` dependency is missing.
    """
    try:
        from scgraph import scGraph
    except ImportError as exc:
        raise RuntimeError("scgraph is not installed") from exc

    tmp_path = None
    try:
        fd, tmp_path = tempfile.mkstemp(
            suffix=f"_{rep_key}_scgraph.h5ad",
            dir=output_dir,
        )
        os.close(fd)
        adata.write_h5ad(tmp_path)
        analyzer = scGraph(
            adata_path=tmp_path,
            batch_key=batch_key,
            label_key=label_key,
            trim_rate=trim_rate,
            thres_batch=thres_batch,
            thres_celltype=thres_celltype,
            only_umap=only_umap,
        )
        result = analyzer.main()
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

    result = result.copy()
    result.insert(0, "metric_group", "scgraph")
    result.insert(1, "rep_key", rep_key)
    return result.reset_index(drop=True)


def evaluate_classifier(adata, rep_key, label_key, test_size, random_state, max_iter):
    """Train a downstream classifier on the embedding.

    This metric group answers: are known cell-type labels linearly recoverable
    from the learned representation? A stronger representation should usually
    yield higher held-out classification scores.

    Args:
        adata: AnnData object with an embedding in ``adata.obsm``.
        rep_key: Embedding key, such as ``X_qvae``.
        label_key: ``adata.obs`` column with cell-type labels.
        test_size: Fraction of cells held out for testing.
        random_state: Random seed for splitting and classifier fitting.
        max_iter: Maximum iterations for logistic regression.

    Returns:
        One-row DataFrame with accuracy, weighted precision, weighted recall,
        weighted F1, and class count.
    """
    x = adata.obsm[rep_key]
    labels = adata.obs[label_key].astype(str).to_numpy()
    encoder = LabelEncoder()
    y = encoder.fit_transform(labels)

    stratify = y if np.min(np.bincount(y)) >= 2 else None
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    classifier = LogisticRegression(
        max_iter=max_iter,
        n_jobs=-1,
        random_state=random_state,
        class_weight="balanced",
    )
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)

    return pd.DataFrame(
        [
            {
                "metric_group": "classification",
                "rep_key": rep_key,
                "label_key": label_key,
                "classifier": "logistic_regression",
                "test_size": test_size,
                "accuracy": metrics.accuracy_score(y_test, y_pred),
                "precision_weighted": metrics.precision_score(
                    y_test, y_pred, average="weighted", zero_division=0
                ),
                "recall_weighted": metrics.recall_score(
                    y_test, y_pred, average="weighted", zero_division=0
                ),
                "f1_weighted": metrics.f1_score(
                    y_test, y_pred, average="weighted", zero_division=0
                ),
                "n_classes": len(encoder.classes_),
            }
        ]
    )


def run_dpt(adata, rep_key, root_cell_type, label_key, n_neighbors, out_key):
    """Run diffusion pseudotime on a representation.

    Args:
        adata: AnnData object modified in place.
        rep_key: Representation used to build the neighbor graph.
        root_cell_type: Cell type used as the DPT root.
        label_key: ``adata.obs`` column with cell-type labels.
        n_neighbors: Number of neighbors used for graph construction.
        out_key: ``adata.obs`` column where pseudotime is stored.

    Returns:
        Pseudotime values as a pandas Series.

    Raises:
        ValueError: If ``root_cell_type`` is not present in ``label_key``.
    """
    sc.pp.neighbors(adata, use_rep=rep_key, n_neighbors=n_neighbors)
    if root_cell_type not in set(adata.obs[label_key].astype(str)):
        raise ValueError(f"{root_cell_type!r} not found in adata.obs[{label_key!r}]")

    root_cells = adata.obs_names[adata.obs[label_key].astype(str) == root_cell_type]
    adata.uns["iroot"] = int(np.flatnonzero(adata.obs_names == root_cells[0])[0])
    sc.tl.diffmap(adata)
    sc.tl.dpt(adata)
    adata.obs[out_key] = adata.obs["dpt_pseudotime"].to_numpy()
    return adata.obs[out_key]


def evaluate_dpt(adata, rep_key, label_key, root_cell_type, n_neighbors):
    """Evaluate DPT pseudotime and optional trajectory agreement.

    This metric group answers: does the embedding produce a pseudotime ordering
    that is consistent with a PCA-based baseline? If ``X_pca`` exists, Kendall's
    tau is computed between QVAE and PCA pseudotime, then rescaled to [0, 1].

    Args:
        adata: AnnData object with an embedding in ``adata.obsm``.
        rep_key: Embedding key, such as ``X_qvae``.
        label_key: ``adata.obs`` column with cell-type labels.
        root_cell_type: Cell type used as the DPT root.
        n_neighbors: Number of neighbors used for graph construction.

    Returns:
        One-row DataFrame with DPT cell count and optional trajectory scores.
    """
    integrated = adata.copy()
    integrated_pt = run_dpt(
        integrated,
        rep_key,
        root_cell_type,
        label_key,
        n_neighbors,
        f"{rep_key}_dpt_pseudotime",
    )

    row = {
        "metric_group": "dpt",
        "rep_key": rep_key,
        "label_key": label_key,
        "root_cell_type": root_cell_type,
        "n_cells_with_dpt": int(integrated_pt.notna().sum()),
    }

    if "X_pca" in adata.obsm:
        baseline = adata.copy()
        baseline_pt = run_dpt(
            baseline,
            "X_pca",
            root_cell_type,
            label_key,
            n_neighbors,
            "X_pca_dpt_pseudotime",
        )
        common = integrated.obs_names.intersection(baseline.obs_names)
        left = integrated_pt.loc[common]
        right = baseline_pt.loc[common]
        valid = left.notna() & right.notna()
        if valid.sum() >= 2:
            tau, _ = kendalltau(left[valid], right[valid])
            row["kendall_tau_vs_X_pca"] = tau
            row["trajectory_conservation_vs_X_pca"] = (tau + 1) / 2

    return pd.DataFrame([row])


def save_result(df, path, label):
    """Save an evaluation table to CSV.

    Args:
        df: Evaluation results.
        path: Destination CSV path.
        label: Human-readable result name printed to stdout.
    """
    df.to_csv(path, index=False)
    print(f"Saved {label}: {path}")


def make_long_summary(results, label_key):
    """Convert benchmark outputs into a compact metric/value summary table.

    Args:
        results: Per-task benchmark DataFrames.
        label_key: Label column used for the run, used when a task output does
            not include it explicitly.

    Returns:
        Long-form DataFrame with one metric per row.
    """
    summary = pd.concat(results, ignore_index=True, sort=False)
    if "label_key" not in summary:
        summary["label_key"] = label_key
    else:
        summary["label_key"] = summary["label_key"].fillna(label_key)

    summary = summary.rename(columns={"metric_group": "task"})
    id_columns = ["task", "rep_key", "label_key", "resolution", "classifier"]
    for column in id_columns:
        if column not in summary:
            summary[column] = np.nan

    parameter_columns = {"n_neighbors", "test_size", "root_cell_type"}
    metric_columns = [
        column
        for column in summary.columns
        if column not in id_columns and column not in parameter_columns
    ]

    long_summary = summary.melt(
        id_vars=id_columns,
        value_vars=metric_columns,
        var_name="metric",
        value_name="value",
    )
    long_summary = long_summary.dropna(subset=["value"]).reset_index(drop=True)
    return long_summary[id_columns + ["metric", "value"]]


def maybe_add_pca(adata):
    """Create ``X_pca`` only when DPT needs a baseline.

    Args:
        adata: AnnData object modified in place when PCA is missing.
    """
    if "X_pca" in adata.obsm:
        return
    n_comps = min(50, adata.n_obs - 1, adata.n_vars - 1)
    if n_comps > 1:
        sc.pp.pca(adata, n_comps=n_comps)


def main():
    """Parse command-line arguments and run selected benchmark groups."""
    parser = argparse.ArgumentParser(
        description="Run optional benchmark metrics for qvae_cell embeddings."
    )
    parser.add_argument("--h5ad", required=True)
    parser.add_argument("--rep-key", default="X_qvae")
    parser.add_argument("--label-key", default="final_annotation")
    parser.add_argument("--batch-key", default="batch")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--n-neighbors", type=int, default=15)
    parser.add_argument("--resolutions", default="0.2,0.4,0.6,0.8,1.0,1.2,1.5,2.0")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--metrics",
        default="clustering",
        help=(
            "Comma-separated metric groups to run. Valid values: "
            "clustering, classification, scib, scgraph, dpt, all. "
            "Default: clustering."
        ),
    )
    parser.add_argument("--root-cell-type", default=None)
    parser.add_argument("--classifier-test-size", type=float, default=0.2)
    parser.add_argument("--classifier-max-iter", type=int, default=1000)
    parser.add_argument("--scib-n-jobs", type=int, default=6)
    parser.add_argument("--scgraph-trim-rate", type=float, default=0.05)
    parser.add_argument("--scgraph-thres-batch", type=int, default=100)
    parser.add_argument("--scgraph-thres-celltype", type=int, default=10)
    parser.add_argument("--scgraph-only-umap", action="store_true")
    args = parser.parse_args()
    selected_metrics = parse_metrics(args.metrics)

    output_dir = args.output_dir or os.path.dirname(args.h5ad) or "."
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading data: {args.h5ad}")
    adata = anndata.read_h5ad(args.h5ad)
    require_key(adata.obsm, args.rep_key, "adata.obsm")
    require_key(adata.obs, args.label_key, "adata.obs")

    if "dpt" in selected_metrics:
        maybe_add_pca(adata)

    all_results = []

    if "clustering" in selected_metrics:
        clustering_df = evaluate_clustering(
            adata,
            args.rep_key,
            args.label_key,
            parse_resolutions(args.resolutions),
            args.n_neighbors,
            args.random_state,
        )
        save_result(
            clustering_df,
            os.path.join(output_dir, f"{args.rep_key}_clustering_metrics.csv"),
            "clustering metrics",
        )
        all_results.append(clustering_df)

    if "classification" in selected_metrics:
        classifier_df = evaluate_classifier(
            adata,
            args.rep_key,
            args.label_key,
            args.classifier_test_size,
            args.random_state,
            args.classifier_max_iter,
        )
        save_result(
            classifier_df,
            os.path.join(output_dir, f"{args.rep_key}_classification_metrics.csv"),
            "classification metrics",
        )
        all_results.append(classifier_df)

    if "scib" in selected_metrics:
        require_key(adata.obs, args.batch_key, "adata.obs")
        try:
            scib_df = evaluate_scib(
                adata,
                args.rep_key,
                args.batch_key,
                args.label_key,
                args.scib_n_jobs,
            )
            save_result(
                scib_df,
                os.path.join(output_dir, f"{args.rep_key}_scib_metrics.csv"),
                "scIB metrics",
            )
            all_results.append(scib_df)
        except RuntimeError as exc:
            print(f"Skipping scIB: {exc}")

    if "scgraph" in selected_metrics:
        require_key(adata.obs, args.batch_key, "adata.obs")
        try:
            scgraph_df = evaluate_scgraph(
                adata,
                args.rep_key,
                args.batch_key,
                args.label_key,
                output_dir,
                args.scgraph_trim_rate,
                args.scgraph_thres_batch,
                args.scgraph_thres_celltype,
                args.scgraph_only_umap,
            )
            save_result(
                scgraph_df,
                os.path.join(output_dir, f"{args.rep_key}_scgraph_metrics.csv"),
                "scGraph metrics",
            )
            all_results.append(scgraph_df)
        except RuntimeError as exc:
            print(f"Skipping scGraph: {exc}")

    if "dpt" in selected_metrics:
        if args.root_cell_type is None:
            raise ValueError("--root-cell-type is required when --metrics includes dpt")
        dpt_df = evaluate_dpt(
            adata,
            args.rep_key,
            args.label_key,
            args.root_cell_type,
            args.n_neighbors,
        )
        save_result(
            dpt_df,
            os.path.join(output_dir, f"{args.rep_key}_dpt_metrics.csv"),
            "DPT metrics",
        )
        all_results.append(dpt_df)

    if not all_results:
        print(
            "No benchmark results were produced. Check optional dependencies and --metrics."
        )
        return

    summary = make_long_summary(all_results, args.label_key)
    save_result(
        summary,
        os.path.join(output_dir, f"{args.rep_key}_benchmark_summary.csv"),
        "benchmark summary",
    )


if __name__ == "__main__":
    main()
