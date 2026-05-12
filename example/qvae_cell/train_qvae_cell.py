import argparse
import os

import anndata
import numpy as np
import pandas as pd
import scanpy as sc
import torch

from trainer import Trainer, set_seed
from visualization import (
    plot_training_history,
    save_energy_plots,
    save_umap_plots,
    setup_plotting,
)


DATASET_PARAMS = {
    "immune": {
        "dataset_name": "immune",
        "batch_key": "batch",
        "labels_key": "final_annotation",
        "file_path": "../immune_processed.h5ad",
    }
}


def parse_args():
    parser = argparse.ArgumentParser(description="Train KPP QVAE for single-cell representations.")
    parser.add_argument("--dataset", default="immune", choices=sorted(DATASET_PARAMS))
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--output-dir", default="./outputs_immune_sa")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--val-percentage", type=float, default=0.1)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--latent-dim", type=int, default=256)
    parser.add_argument("--num-visible", type=int, default=128)
    parser.add_argument("--num-hidden", type=int, default=128)
    parser.add_argument("--normalization-method", default="layer", choices=["layer", "batch"])
    parser.add_argument("--dist-beta", type=float, default=10.0)
    parser.add_argument("--kl-beta", type=float, default=1e-5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--rbm-lr", type=float, default=1e-3)
    parser.add_argument("--bm-weight-decay", type=float, default=0.0)
    parser.add_argument("--loss-type", default="mse", choices=["mse", "bernoulli"])
    parser.add_argument("--representation", default="q", choices=["zeta", "q"])
    parser.add_argument("--load-weights", action="store_true")
    parser.add_argument("--sampler-type", default="sa", choices=["sa", "cim"])
    parser.add_argument("--sa-initial-temperature", type=float, default=1000)
    parser.add_argument("--sa-alpha", type=float, default=0.5)
    parser.add_argument("--sa-cutoff-temperature", type=float, default=0.001)
    parser.add_argument("--sa-iterations-per-t", type=int, default=10)
    parser.add_argument("--sa-size-limit", type=int, default=10)
    parser.add_argument("--sa-rand-seed", type=int, default=512)
    parser.add_argument("--project-no", default="26035324")
    parser.add_argument("--task-name", default="demo2_qvae")
    parser.add_argument("--task-mode", default="sample", choices=["sample", "quota"])
    parser.add_argument("--sample-number", type=int, default=16)
    parser.add_argument("--precision", type=int, default=8)
    parser.add_argument("--truncated-precision", type=int, default=10)
    parser.add_argument("--target-bits", type=int, default=550)
    parser.add_argument("--tmp-dir", default="./tmp")
    return parser.parse_args()


def main():
    args = parse_args()
    args.device_obj = torch.device(args.device if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    setup_plotting()
    os.makedirs(args.output_dir, exist_ok=True)
    trainer = Trainer(args, args.device_obj)

    params = DATASET_PARAMS[args.dataset].copy()
    if args.data_path is not None:
        params["file_path"] = args.data_path

    print(f"Using device: {args.device_obj}")
    print(f"Loading data from: {params['file_path']}")
    adata = anndata.read_h5ad(params["file_path"])
    adata.obs_names_make_unique()
    print(f"Dataset '{args.dataset}': {adata.n_obs} cells, {adata.n_vars} features")

    x = trainer.adata_to_array(adata)
    batch_indices = trainer.batch_indices(adata, params["batch_key"])
    train_loader, val_loader, eval_loader = trainer.make_loaders(x, batch_indices)
    model, vae_optimizer, bm_optimizer = trainer.create_model(adata.n_vars, train_loader)

    weights_path = os.path.join(args.output_dir, f"{args.dataset}_kpp_qvae_best.pth")
    if args.load_weights:
        model.load_state_dict(torch.load(weights_path, map_location=args.device_obj))
        print(f"Loaded weights from {weights_path}")
    else:
        history = trainer.train(
            model, train_loader, val_loader, vae_optimizer, bm_optimizer, weights_path
        )
        pd.DataFrame(history).to_csv(
            os.path.join(args.output_dir, "training_history.csv"), index=False
        )
        plot_training_history(history, os.path.join(args.output_dir, "training_curves.png"))
        print(f"Best weights saved to {weights_path}")

    reps = trainer.extract_representation(model, eval_loader)
    adata.obsm["X_qvae"] = reps
    np.save(os.path.join(args.output_dir, "X_qvae.npy"), reps)
    print(f"Representation shape: {reps.shape}")

    sc.pp.neighbors(adata, use_rep="X_qvae", n_neighbors=15)
    sc.tl.umap(adata, random_state=args.seed)
    save_umap_plots(adata, params["labels_key"], params["batch_key"], args.output_dir)

    adata.obs["QVAE_Energy"] = trainer.compute_energy(model, eval_loader)
    save_energy_plots(adata, params["labels_key"], args.output_dir)

    adata.write_h5ad(os.path.join(args.output_dir, f"{args.dataset}_kpp_qvae.h5ad"))
    print(f"Outputs saved under: {args.output_dir}")


if __name__ == "__main__":
    main()
