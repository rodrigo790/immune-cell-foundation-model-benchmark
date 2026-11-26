"""Matched CPU pilot of the frozen scGPT whole-human encoder on five lineages."""

import hashlib
import json
import time
import anndata as ad
import gdown
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from inputs import CACHE, OUT, source
from analysis import URL, PANELS, folds

IDS = {
    "vocab.json": "1H3E_MJ-Dl36AQV6jLbna2EdvgPaqvqcC",
    "args.json": "1hh2zGKyWAx3DyovD30GStZ3QlzmSqdk1",
    "best_model.pt": "14AebJfGOUF047Eg40hk57HCtrb0fyDTm",
}


def checkpoint():
    folder = CACHE / "scgpt"
    folder.mkdir(parents=True, exist_ok=True)
    meta = {}
    for name, ident in IDS.items():
        path = folder / name
        if not path.exists():
            gdown.download(id=ident, output=str(path), quiet=True)
        with path.open("rb") as handle:
            meta[name] = dict(
                file_id=ident, sha256=hashlib.file_digest(handle, "sha256").hexdigest()
            )
    from pathlib import Path

    record = Path("data/checkpoint.json")
    if record.exists() and json.loads(record.read_text()) != meta:
        raise ValueError("Changed checkpoint")
    record.write_text(json.dumps(meta, indent=2))
    return folder


def embeddings(X, genes, folder):
    args = json.loads((folder / "args.json").read_text())
    vocab = json.loads((folder / "vocab.json").read_text())
    state = torch.load(folder / "best_model.pt", map_location="cpu", weights_only=True)
    width = args["embsize"]
    layer = nn.TransformerEncoderLayer(
        width, args["nheads"], args["d_hid"], dropout=0, batch_first=True
    )
    encoder = nn.TransformerEncoder(
        layer, args["nlayers"], enable_nested_tensor=False
    ).eval()
    prefix = "transformer_encoder."
    encoder.load_state_dict(
        {
            k[len(prefix) :].replace("Wqkv.", "in_proj_"): v
            for k, v in state.items()
            if k.startswith(prefix)
        }
    )

    def norm(x, key):
        return F.layer_norm(x, (width,), state[key + ".weight"], state[key + ".bias"])

    def linear(x, key):
        return F.linear(x, state[key + ".weight"], state[key + ".bias"])

    rng = np.random.default_rng(431)
    ids = np.full((X.shape[0], 128), vocab["<pad>"], dtype=int)
    values = np.zeros(ids.shape, dtype=np.float32)
    ids[:, 0] = vocab["<cls>"]
    mapping = np.array([vocab.get(g, -1) for g in genes])
    for i in range(X.shape[0]):
        row = X.getrow(i)
        match = mapping[row.indices] >= 0
        counts = row.data[match]
        tokens = mapping[row.indices[match]]
        if not len(tokens):
            continue
        edges = np.quantile(counts, np.linspace(0, 1, 50))
        binned = np.maximum(1, np.digitize(counts, edges))
        take = rng.choice(len(tokens), min(len(tokens), 127), replace=False)
        ids[i, 1 : 1 + len(take)] = tokens[take]
        values[i, 1 : 1 + len(take)] = binned[take]
    chunks = []
    with torch.inference_mode():
        for first in range(0, len(ids), 8):
            tok = torch.tensor(ids[first : first + 8])
            v = torch.tensor(values[first : first + 8]).unsqueeze(-1)
            ge = norm(
                F.embedding(tok, state["encoder.embedding.weight"]), "encoder.enc_norm"
            )
            ve = norm(
                linear(
                    F.relu(linear(v, "value_encoder.linear1")), "value_encoder.linear2"
                ),
                "value_encoder.norm",
            )
            embedding = encoder(ge + ve, src_key_padding_mask=tok.eq(vocab["<pad>"]))[
                :, 0
            ]
            chunks.append(F.normalize(embedding).numpy())
    return np.vstack(chunks)


def main():
    torch.set_num_threads(4)
    rng = np.random.default_rng(431)
    a = ad.read_h5ad(source(URL), backed="r")
    obs = a.obs.reset_index(drop=True)
    chosen = []
    for _, g in obs[obs.cell_type.isin(PANELS)].groupby(
        ["donor_id", "cell_type"], observed=True
    ):
        chosen.extend(rng.choice(g.index, min(12, len(g)), replace=False))
    ix = np.sort(chosen)
    X = a.raw.X[ix, :].tocsr()
    symbols = a.raw.var.feature_name.to_numpy()
    donors = obs.donor_id.iloc[ix].astype(str).to_numpy()
    labels = obs.cell_type.iloc[ix].astype(str).to_numpy()
    markers = [
        np.flatnonzero(symbols == g)[0] for group in PANELS.values() for g in group
    ]
    small = np.log1p(
        1e4 * X[:, markers].toarray() / np.maximum(np.asarray(X.sum(1)), 1)
    )
    start = time.perf_counter()
    features = embeddings(X, symbols, checkpoint())
    seconds = time.perf_counter() - start
    rows = []
    for donor, tr, te in folds(donors):
        for method in ["markers", "PCA-logistic", "scGPT-linear"]:
            if method == "markers":
                pred = np.array(list(PANELS))[
                    small[te].reshape(-1, 5, 3).mean(2).argmax(1)
                ]
            else:
                z = features if method == "scGPT-linear" else small
                model = (
                    make_pipeline(
                        StandardScaler(),
                        LogisticRegression(class_weight="balanced", max_iter=1000),
                    )
                    if method == "scGPT-linear"
                    else make_pipeline(
                        StandardScaler(),
                        PCA(10),
                        LogisticRegression(class_weight="balanced", max_iter=1000),
                    )
                )
                pred = model.fit(z[tr], labels[tr]).predict(z[te])
            rows.append(
                [
                    donor,
                    method,
                    len(te),
                    f1_score(labels[te], pred, average="macro"),
                    seconds if method == "scGPT-linear" else np.nan,
                ]
            )
    pd.DataFrame(
        rows,
        columns=[
            "donor",
            "method",
            "cells",
            "macro_f1",
            "whole_pilot_embedding_seconds",
        ],
    ).to_csv(OUT / "foundation_pilot.csv", index=False)


if __name__ == "__main__":
    main()
