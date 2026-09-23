# -*- coding: utf-8 -*-
"""Unified dataset loader for QVAE-Anomaly.

Returns dict:
    {
      "name": str,
      "dim": int,
      "x_train", "y_train", "x_val", "y_val", "x_test", "y_test": torch.Tensor,
      ("kept_cols": list[bool], for structure only)
    }
y is int64 with 1 = anomaly, 0 = normal.
"""
from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import RobustScaler
from .exception import ValueError

ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = Path(os.environ.get(
    "QVAE_DATA_DIR",
    str(ROOT / "data")))


def _split_1to1(x, y, seed=42, n_val=500, n_test=500):
    """Stratified split: train keeps imbalance; val/test are 1:1."""
    rng = np.random.RandomState(seed)
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    rng.shuffle(pos); rng.shuffle(neg)
    nv = min(n_val // 2, len(pos) // 2, len(neg) // 2)
    nt = min(n_test // 2, (len(pos) - nv) // 2, (len(neg) - nv) // 2)
    vp, vn = pos[:nv], neg[:nv]
    tp, tn = pos[nv:nv + nt], neg[nv:nv + nt]
    tr_p, tr_n = pos[nv + nt:], neg[nv + nt:]
    val_idx = np.concatenate([vp, vn]); rng.shuffle(val_idx)
    test_idx = np.concatenate([tp, tn]); rng.shuffle(test_idx)
    train_idx = np.concatenate([tr_p, tr_n]); rng.shuffle(train_idx)
    return (x[train_idx], y[train_idx],
            x[val_idx], y[val_idx],
            x[test_idx], y[test_idx])


def _clean_scale_structure(x_train, x_val, x_test):
    """Drop constant columns (fit on train), median-impute, RobustScaler."""
    keep = np.asarray([len(np.unique(col[np.isfinite(col)])) > 1
                       for col in x_train.T], dtype=bool)
    x_train = x_train[:, keep]; x_val = x_val[:, keep]; x_test = x_test[:, keep]
    medians = np.nanmedian(x_train, axis=0).astype(np.float32)
    for blk in (x_train, x_val, x_test):
        rows, cols = np.where(~np.isfinite(blk))
        blk[rows, cols] = medians[cols]
    sc = RobustScaler().fit(x_train)
    return (sc.transform(x_train).astype(np.float32),
            sc.transform(x_val).astype(np.float32),
            sc.transform(x_test).astype(np.float32), keep, sc)


def load_esm():
    d = torch.load(ROOT / "esm2_mlp_dataset.pt", map_location="cpu", weights_only=False)
    xtr = d["train_x"].float().numpy()
    xv = d["validation_x"].float().numpy()
    xte = d["test_x"].float().numpy()
    sc = RobustScaler().fit(xtr)
    return {"name": "esm", "dim": int(xtr.shape[1]),
            "x_train": torch.tensor(sc.transform(xtr).astype(np.float32)),
            "y_train": d["train_y"].long(),
            "x_val": torch.tensor(sc.transform(xv).astype(np.float32)),
            "y_val": d["validation_y"].long(),
            "x_test": torch.tensor(sc.transform(xte).astype(np.float32)),
            "y_test": d["test_y"].long(),
            "scaler": sc}


def load_structure():
    d = torch.load(ROOT / "structure_mlp_dataset.pt", map_location="cpu", weights_only=False)
    xtr = d["train_x"].numpy(); xv = d["validation_x"].numpy(); xte = d["test_x"].numpy()
    xtr, xv, xte, keep, sc = _clean_scale_structure(xtr, xv, xte)
    return {"name": "structure", "dim": int(xtr.shape[1]),
            "x_train": torch.tensor(xtr), "y_train": d["train_y"].long(),
            "x_val": torch.tensor(xv), "y_val": d["validation_y"].long(),
            "x_test": torch.tensor(xte), "y_test": d["test_y"].long(),
            "kept_cols": keep, "scaler": sc}


def load_creditcard():
    df = pd.read_csv(DATA_ROOT / "creditcard.csv")
    df["Time"] = df["Time"] / 3600 % 24
    df["Amount"] = np.log(df["Amount"] + 1)
    y = df["Class"].values
    x = df.drop(columns=["Class"]).values.astype(np.float32)
    xtr, ytr, xv, yv, xte, yte = _split_1to1(x, y)
    sc = RobustScaler().fit(xtr)
    return {"name": "creditcard", "dim": int(xtr.shape[1]),
            "x_train": torch.tensor(sc.transform(xtr).astype(np.float32)),
            "y_train": torch.tensor(ytr, dtype=torch.int64),
            "x_val": torch.tensor(sc.transform(xv).astype(np.float32)),
            "y_val": torch.tensor(yv, dtype=torch.int64),
            "x_test": torch.tensor(sc.transform(xte).astype(np.float32)),
            "y_test": torch.tensor(yte, dtype=torch.int64),
            "scaler": sc}


def load_thyroid():
    z = np.load(DATA_ROOT / "38_thyroid.npz")
    x = z["X"].astype(np.float32)
    y = z["y"].astype(int).reshape(-1)
    xtr, ytr, xv, yv, xte, yte = _split_1to1(x, y)
    sc = RobustScaler().fit(xtr)
    return {"name": "thyroid", "dim": int(xtr.shape[1]),
            "x_train": torch.tensor(sc.transform(xtr).astype(np.float32)),
            "y_train": torch.tensor(ytr, dtype=torch.int64),
            "x_val": torch.tensor(sc.transform(xv).astype(np.float32)),
            "y_val": torch.tensor(yv, dtype=torch.int64),
            "x_test": torch.tensor(sc.transform(xte).astype(np.float32)),
            "y_test": torch.tensor(yte, dtype=torch.int64),
            "scaler": sc}


def load_kddcup99():
    """KDD Cup 99 (corrected 10% train + corrected test)."""
    NAMES = (
        "duration,protocol_type,service,flag,src_bytes,dst_bytes,land,"
        "wrong_fragment,urgent,hot,num_failed_logins,logged_in,num_compromised,"
        "root_shell,su_attempted,num_root,num_file_creations,num_shells,"
        "num_access_files,num_outbound_cmds,is_host_login,is_guest_login,"
        "count,srv_count,serror_rate,srv_serror_rate,rerror_rate,srv_rerror_rate,"
        "same_srv_rate,diff_srv_rate,srv_diff_host_rate,dst_host_count,"
        "dst_host_srv_count,dst_host_same_srv_rate,dst_host_diff_srv_rate,"
        "dst_host_same_src_port_rate,dst_host_srv_diff_host_rate,"
        "dst_host_serror_rate,dst_host_srv_serror_rate,dst_host_rerror_rate,"
        "dst_host_srv_rerror_rate,label"
    ).split(",")
    NUMERIC = [
        "duration","src_bytes","dst_bytes","wrong_fragment","urgent","hot",
        "num_failed_logins","num_compromised","root_shell","su_attempted",
        "num_root","num_file_creations","num_shells","num_access_files",
        "num_outbound_cmds","count","srv_count","serror_rate","srv_serror_rate",
        "rerror_rate","srv_rerror_rate","same_srv_rate","diff_srv_rate",
        "srv_diff_host_rate","dst_host_count","dst_host_srv_count",
        "dst_host_same_srv_rate","dst_host_diff_srv_rate",
        "dst_host_same_src_port_rate","dst_host_srv_diff_host_rate",
        "dst_host_serror_rate","dst_host_srv_serror_rate",
        "dst_host_rerror_rate","dst_host_srv_rerror_rate",
    ]

    train = pd.read_csv(PUBLIC_ROOT / "kdd_99/kddcup.data_10_percent_corrected.gz",
                        header=None, names=NAMES, compression=None, encoding="latin-1",
                        engine="python", on_bad_lines="skip")
    test = pd.read_csv(PUBLIC_ROOT / "kdd_99/corrected_dir/corrected",
                       header=None, names=NAMES, compression=None, encoding="latin-1")

    # numeric coercion
    for col in NUMERIC:
        train[col] = pd.to_numeric(train[col], errors="coerce")
        test[col] = pd.to_numeric(test[col], errors="coerce")

    # log1p for heavy-tailed byte features (reduce recon MSE scale)
    for col in ["src_bytes", "dst_bytes", "duration"]:
        train[col] = np.log1p(train[col].clip(lower=0))
        test[col] = np.log1p(test[col].clip(lower=0))

    # label: normal=0, everything else=1
    train["y"] = (train["label"].str.strip() != "normal.").astype(int)
    test["y"] = (test["label"].str.strip() != "normal.").astype(int)

    # attack type mapping
    ATTACK_MAP = {"back.": "dos", "buffer_overflow.": "u2r", "ftp_write.": "r2l",
                  "guess_passwd.": "r2l", "imap.": "r2l", "ipsweep.": "probe",
                  "land.": "dos", "loadmodule.": "u2r", "multihop.": "r2l",
                  "neptune.": "dos", "nmap.": "probe", "perl.": "u2r",
                  "phf.": "r2l", "pod.": "dos", "portsweep.": "probe",
                  "rootkit.": "u2r", "satan.": "probe", "smurf.": "dos",
                  "spy.": "r2l", "teardrop.": "dos", "warezclient.": "r2l",
                  "warezmaster.": "r2l", "normal.": "normal"}
    train["attack_type"] = train["label"].str.strip().map(ATTACK_MAP).fillna("unknown")
    test["attack_type"] = test["label"].str.strip().map(ATTACK_MAP).fillna("unknown")

    # one-hot categorical
    X_train = pd.get_dummies(train.drop(columns=["label", "y", "attack_type"]),
                              columns=["protocol_type", "service", "flag"])
    X_test = pd.get_dummies(test.drop(columns=["label", "y", "attack_type"]),
                             columns=["protocol_type", "service", "flag"])
    X_train, X_test = X_train.align(X_test, join="outer", axis=1, fill_value=0)

    xtr = X_train.values.astype(np.float32)
    xte = X_test.values.astype(np.float32)
    ytr = train["y"].values.astype(np.int64)
    yte = test["y"].values.astype(np.int64)

    # split train → train + val (val 1:1, n_val=500)
    xtr, ytr, xv, yv, _, _ = _split_1to1(xtr, ytr, n_val=500, n_test=0)

    sc = RobustScaler().fit(xtr)
    return {"name": "kddcup99", "dim": int(xtr.shape[1]),
            "x_train": torch.tensor(sc.transform(xtr).astype(np.float32)),
            "y_train": torch.tensor(ytr, dtype=torch.int64),
            "x_val": torch.tensor(sc.transform(xv).astype(np.float32)),
            "y_val": torch.tensor(yv, dtype=torch.int64),
            "x_test": torch.tensor(sc.transform(xte).astype(np.float32)),
            "y_test": torch.tensor(yte, dtype=torch.int64),
            "attack_type_test": test["attack_type"].values,
            "scaler": sc}


LOADERS = {
    "creditcard": load_creditcard,
    "thyroid": load_thyroid,
    "kddcup99": load_kddcup99,
}


def load(name: str) -> dict:
    if name not in LOADERS:
        raise ValueError(f"unknown dataset '{name}'; choose {list(LOADERS)}")
    return LOADERS[name]()


if __name__ == "__main__":
    for n in LOADERS:
        r = load(n)
        print(f"{n:10s} dim={r['dim']:4d} train={tuple(r['x_train'].shape)} "
              f"pos={r['y_train'].sum().item()}/{len(r['y_train'])} "
              f"val_pos={r['y_val'].sum().item()} test_pos={r['y_test'].sum().item()}")
