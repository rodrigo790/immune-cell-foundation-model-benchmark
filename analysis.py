"""Leave-one-donor-out annotation of five broad, author-labelled lymph node lineages."""

import time
import anndata as ad
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay
from inputs import OUT, source

URL = "https://datasets.cellxgene.cziscience.com/9050e178-26b3-40d7-aff0-2611693a8012.h5ad"
PANELS = {
    "B cell": ["MS4A1", "CD79A", "CD79B"],
    "T cell": ["CD3D", "CD3E", "TRAC"],
    "natural killer cell": ["NKG7", "KLRD1", "GNLY"],
    "endothelial cell": ["PECAM1", "VWF", "CDH5"],
    "mast cell": ["TPSAB1", "TPSB2", "CPA3"],
}


def folds(donors):
    for donor in sorted(set(donors)):
        yield donor, np.flatnonzero(donors != donor), np.flatnonzero(donors == donor)


def main():
    a = ad.read_h5ad(source(URL), backed="r")
    genes = sum(PANELS.values(), [])
    symbols = a.raw.var["feature_name"].to_numpy()
    if any(np.sum(symbols == g) != 1 for g in genes):
        raise ValueError("Ambiguous panel gene")
    idx = [np.flatnonzero(symbols == g)[0] for g in genes]
    mask = a.obs.cell_type.isin(PANELS).to_numpy()
    obs = a.obs.loc[mask]
    # Slice rows before columns to support backed CSR.
    X = a.raw.X[np.flatnonzero(mask), :].tocsr()
    depth = np.asarray(X.sum(axis=1)).ravel()
    x = np.log1p(X[:, idx].toarray() * 1e4 / np.maximum(depth[:, None], 1))
    labels = obs.cell_type.astype(str).to_numpy()
    donors = obs.donor_id.astype(str).to_numpy()
    classes = list(PANELS)
    rows = []
    times = []
    strata = []
    condition = obs.sample_type.astype(str).to_numpy()
    conf = np.zeros((len(classes), len(classes)), int)
    pd.DataFrame(
        [(k, v) for k, vs in PANELS.items() for v in vs], columns=["label", "marker"]
    ).to_csv(OUT / "label_map.csv", index=False)
    for donor, train, test in folds(donors):
        for method in ["marker", "PCA-logistic"]:
            start = time.perf_counter()
            if method == "marker":
                predicted = np.asarray(classes)[
                    x[test].reshape(-1, len(classes), 3).mean(axis=2).argmax(axis=1)
                ]
            else:
                model = make_pipeline(
                    StandardScaler(),
                    PCA(n_components=10),
                    LogisticRegression(class_weight="balanced", max_iter=500),
                )
                model.fit(x[train], labels[train])
                predicted = model.predict(x[test])
                conf += confusion_matrix(labels[test], predicted, labels=classes)
            times.append([donor, method, time.perf_counter() - start])
            for group in sorted(set(condition[test])):
                for label in classes:
                    keep = (condition[test] == group) & (labels[test] == label)
                    if keep.any():
                        strata.append(
                            [
                                donor,
                                method,
                                group,
                                label,
                                int(keep.sum()),
                                np.mean(predicted[keep] == label),
                            ]
                        )
            present = np.unique(labels[test])
            rows.append(
                [
                    donor,
                    method,
                    len(test),
                    len(present),
                    f1_score(
                        labels[test],
                        predicted,
                        labels=present,
                        average="macro",
                        zero_division=0,
                    ),
                ]
            )
    pd.DataFrame(
        rows, columns=["donor", "method", "cells", "observed_classes", "macro_f1"]
    ).to_csv(OUT / "donor_metrics.csv", index=False)
    pd.DataFrame(
        strata, columns=["donor", "method", "condition", "cell_type", "cells", "recall"]
    ).to_csv(OUT / "condition_recall.csv", index=False)
    pd.DataFrame(times, columns=["donor", "method", "seconds"]).to_csv(
        OUT / "runtime.csv", index=False
    )
    ConfusionMatrixDisplay(conf, display_labels=classes).plot(
        xticks_rotation=40, colorbar=False, values_format="d"
    )
    plt.title("Five lineages, held-out donor predictions")
    plt.tight_layout()
    plt.savefig(OUT / "confusion_matrix.png", dpi=160)


if __name__ == "__main__":
    main()
