import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

N, L, D = 80000, 36, 1152
N_COMPONENTS = 50
OUTLIER_RATIO = 20.0  # dim considered outlier if var > OUTLIER_RATIO * median layer var

IN_DIR = Path("stageA_mean_pooled")
OUT_DIR = Path("stageA_pca")
OUT_DIR.mkdir(exist_ok=True)

CONDITIONS = {
    "binder_alone": "binder_alone_mean_hidden.f32",
    "joint": "binder_target_joint_mean_hidden.f32",
}


def fit_pca(X: np.ndarray, n_components: int) -> tuple[np.ndarray, np.ndarray, PCA]:
    k = min(n_components, X.shape[1])
    pca = PCA(n_components=k, svd_solver="randomized", random_state=0)
    scores = pca.fit_transform(X)
    return scores.astype(np.float32), pca.explained_variance_ratio_.astype(np.float32), pca


for cond_name, fname in CONDITIONS.items():
    print(f"\n### condition: {cond_name} ###", flush=True)
    t_load = time.time()
    arr = np.memmap(IN_DIR / fname, dtype="float32", mode="r", shape=(N, L, D))
    arr = np.asarray(arr)  # full read into RAM (~13.3GB)
    print(f"loaded into RAM in {time.time() - t_load:.1f}s", flush=True)

    cond_out = OUT_DIR / cond_name
    cond_out.mkdir(exist_ok=True)

    for layer in range(L):
        t0 = time.time()
        X = arr[:, layer, :]  # (N, D) view, no copy

        var = X.var(axis=0)
        median_var = np.median(var)
        outlier_dims = np.where(var > OUTLIER_RATIO * median_var)[0]
        keep_dims = np.setdiff1d(np.arange(D), outlier_dims)

        # Diagnostic: pairwise correlation among this layer's outlier dims.
        outlier_corr_note = "n/a"
        if len(outlier_dims) >= 2:
            sub = X[:, outlier_dims]
            corr = np.corrcoef(sub, rowvar=False)
            iu = np.triu_indices(len(outlier_dims), k=1)
            outlier_corr_note = ", ".join(
                f"({outlier_dims[i]},{outlier_dims[j]})={corr[i, j]:.2f}"
                for i, j in zip(*iu)
            )

        results = {}

        # Variant 1: correlation-matrix PCA (standardize all D dims, then PCA)
        scaler = StandardScaler().fit(X)
        Xs = scaler.transform(X)
        scores, evr, pca = fit_pca(Xs, N_COMPONENTS)
        results["correlation_all_dims"] = (scores, evr, pca, scaler)

        # Variant 2: covariance-matrix PCA, all dims, raw scale (generation-relevant)
        scores, evr, pca = fit_pca(X, N_COMPONENTS)
        results["covariance_all_dims"] = (scores, evr, pca, None)

        # Variant 3: covariance-matrix PCA, outliers excluded (interpretability)
        X_no_outliers = X[:, keep_dims]
        scores, evr, pca = fit_pca(X_no_outliers, N_COMPONENTS)
        results["covariance_outliers_excluded"] = (scores, evr, pca, None)

        layer_dir = cond_out / f"layer_{layer:02d}"
        layer_dir.mkdir(exist_ok=True)
        for variant_name, (scores, evr, pca, scaler) in results.items():
            np.save(layer_dir / f"{variant_name}_scores.npy", scores)
            np.save(layer_dir / f"{variant_name}_explained_var_ratio.npy", evr)
            joblib.dump(pca, layer_dir / f"{variant_name}_pca.joblib")
            if scaler is not None:
                joblib.dump(scaler, layer_dir / f"{variant_name}_scaler.joblib")
        np.save(layer_dir / "outlier_dims.npy", outlier_dims)
        np.save(layer_dir / "keep_dims.npy", keep_dims)

        top1 = {k: v[1][0] for k, v in results.items()}
        top10sum = {k: v[1][:10].sum() for k, v in results.items()}
        print(
            f"layer {layer:2d} ({time.time() - t0:.1f}s): "
            f"n_outliers={len(outlier_dims)} outlier_dims={list(outlier_dims)} "
            f"pairwise_corr=[{outlier_corr_note}] | "
            f"PC1_var_explained: corr={top1['correlation_all_dims']:.1%} "
            f"cov_all={top1['covariance_all_dims']:.1%} "
            f"cov_excl={top1['covariance_outliers_excluded']:.1%} | "
            f"top10_var_explained: corr={top10sum['correlation_all_dims']:.1%} "
            f"cov_all={top10sum['covariance_all_dims']:.1%} "
            f"cov_excl={top10sum['covariance_outliers_excluded']:.1%}",
            flush=True,
        )

    del arr
    print(f"### done with {cond_name} ###", flush=True)

print("\nall done")
