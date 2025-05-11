import logging
import numpy as np
from scipy.stats import skew, kurtosis
from scipy.signal import welch
import antropy as ant
from joblib import Parallel, delayed

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

def extract_features_cissa_light_v2(
    X_cissa: np.ndarray,
    sfreq: float = 128.0,
    bands=None,
    n_jobs: int = -1,
    ds: int = 8,                # ↑ more aggressive downsampling
    nperseg: int = 64,
    chunk_size: int = 5000,     # ↑ larger chunks
    log_every: int = 5,         # only log every 5 chunks
    max_comps: int = None       # e.g. 2 to keep just first 2 components
) -> np.ndarray:
    """
    Ultra‐light feature extractor, v2:
      • (Optional) slice to `max_comps`
      • downsample each signal by ds (now default=8)
      • clamp nperseg≤n_s so Welch never upsamples
      • drop fractal dims, keep only 3 entropies + bandpowers
      • process in big chunks of chunk_size (default=5k)
      • only log every `log_every` chunks
    """
    # 1) Optionally reduce number of components
    if max_comps is not None:
        X_cissa = X_cissa[:, :max_comps, :]

    # 2) Prep and flatten
    n_epochs, n_comps, n_samples = X_cissa.shape
    Xds   = X_cissa[..., ::ds]
    n_s   = Xds.shape[-1]
    flat  = Xds.reshape(-1, n_s)
    total = flat.shape[0]

    # 3) Build band masks once
    if bands is None:
        bands = [(1,4),(4,8),(8,13),(13,30),(30,40)]
    clamp = min(nperseg, n_s)
    f0, _ = welch(np.zeros(n_s), fs=sfreq, nperseg=clamp)
    band_masks = [ (f0>=lo)&(f0<hi) for lo,hi in bands ]

    # 4) Analytical feature‐length: 4 stats + 3 Hjorth + 3 entropies + len(bands)
    n_feats = 4 + 3 + 3 + len(band_masks)
    feats_flat = np.zeros((total, n_feats), dtype=np.float64)

    # 5) Per‐signal compute
    def _compute(sig):
        out = [
            sig.mean(), sig.std(),
            skew(sig), kurtosis(sig),
        ]
        # Hjorth
        d1, d2 = np.diff(sig), np.diff(sig, 2)
        v0, v1, v2 = np.var(sig), np.var(d1), np.var(d2)
        mob  = np.sqrt(v1/v0) if v0>0 else 0.0
        comp = (np.sqrt(v2/v1)/mob) if (v1>0 and mob>0) else 0.0
        out += [v0, mob, comp]

        # entropies
        out += [
            ant.sample_entropy(sig, metric="chebyshev"),
            ant.perm_entropy(sig, order=3, delay=1, normalize=True),
            ant.spectral_entropy(sig, sfreq, method='welch',
                                 nperseg=clamp, normalize=True),
        ]

        # band‐powers
        f, Pxx = welch(sig, fs=sfreq, nperseg=clamp)
        for mask in band_masks:
            out.append(np.trapz(Pxx[mask], f[mask]))

        return np.array(out, dtype=np.float64)

    # 6) Process in big chunks, logging sparsely
    chunk_idx = 0
    for start in range(0, total, chunk_size):
        end   = min(start + chunk_size, total)
        block = flat[start:end]
        results = Parallel(n_jobs=n_jobs)(
            delayed(_compute)(sig) for sig in block
        )
        feats_flat[start:end] = np.vstack(results)

        if (chunk_idx % log_every) == 0:
            logger.info(f"Processed signals {start:,d}–{end:,d} / {total:,d}")
        chunk_idx += 1

    # 7) Reshape back to (n_epochs, n_comps * n_feats)
    return feats_flat.reshape(n_epochs, n_comps * n_feats)




"""
X_mat_fast  = extract_features_cissa_light_v2(
    X_mat_ssa,  sfreq=128, n_jobs=-1, max_comps=2
)
X_stew_fast = extract_features_cissa_light_v2(
    X_stew_ssa, sfreq=256, n_jobs=-1, max_comps=2
)
"""