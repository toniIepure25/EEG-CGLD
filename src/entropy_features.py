import numpy as np
import antropy as ant
from joblib import Parallel, delayed
import time
from numba import njit

@njit(fastmath=True)
def fast_entropy(signal, edges, bins, renyi_alpha):
    var = np.var(signal)
    diff_e = 0.5 * np.log(2 * np.pi * np.e * var + 1e-12)

    inds = np.digitize(signal, edges) - 1
    hist = np.zeros(bins, dtype=np.float32)
    for i in range(len(inds)):
        if 0 <= inds[i] < bins:
            hist[inds[i]] += 1
    p = hist / (hist.sum() + 1e-12)
    renyi_e = (1.0 / (1.0 - renyi_alpha)) * np.log2(np.sum(p ** renyi_alpha) + 1e-12)

    return renyi_e, diff_e


def extract_features_ultrafast(
    X_ssa: np.ndarray,
    sfreq: float = 128.0,
    renyi_alpha: float = 2.0,
    bins: int = 32,
    ds: int = 12,
    nperseg: int = 64,
    n_jobs: int = 6,
    use_threads: bool = True,
    batch_size: int = 20,
    dtype=np.float32
) -> np.ndarray:
    t0 = time.time()
    X = X_ssa.astype(dtype, copy=False)[..., ::ds]
    sf = sfreq / ds
    n_epochs, n_comps, n_samples = X.shape

    print(f"[INFO] Downsampled to {n_samples} samples. Components: {n_comps}. Epochs: {n_epochs}")

    # Precompute histogram edges
    global_min, global_max = X.min(), X.max()
    edges = np.linspace(global_min, global_max, bins + 1, dtype=dtype)

    def process_signal(sig):
        renyi_e, diff_e = fast_entropy(sig, edges, bins, renyi_alpha)
        samp_e = ant.sample_entropy(sig)
        # app_e removed for speed
        perm_e = ant.perm_entropy(sig, order=3, delay=1, normalize=True)
        spec_e = ant.spectral_entropy(sig, sf, method='welch', nperseg=nperseg, normalize=True)
        return np.array([samp_e, perm_e, spec_e, renyi_e, diff_e], dtype=dtype)

    backend = 'threading' if use_threads else 'loky'
    n_features = 5  # app_entropy removed

    feats = np.zeros((n_epochs, n_comps * n_features), dtype=dtype)

    for start in range(0, n_epochs, batch_size):
        end = min(start + batch_size, n_epochs)
        print(f"[BATCH] Epochs {start}–{end - 1}...", end=' ')
        bt0 = time.time()

        # process each signal independently
        results = Parallel(n_jobs=n_jobs, backend=backend)(
            delayed(process_signal)(X[i, c])
            for i in range(start, end)
            for c in range(n_comps)
        )

        idx = 0
        for i in range(start, end):
            for c in range(n_comps):
                feats[i, c * n_features:(c + 1) * n_features] = results[idx]
                idx += 1

        print(f"done in {time.time() - bt0:.2f}s")

    print(f"[TOTAL] Finished in {time.time() - t0:.2f}s")
    return feats
