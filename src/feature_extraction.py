import numpy as np
from scipy.stats import skew, kurtosis
from scipy.signal import welch
import antropy as ant
from joblib import Parallel, delayed

def extract_features_cissa(
    X_cissa: np.ndarray,
    sfreq: float = 128.0,
    bands=None,
    n_jobs: int = -1,
    ds: int = 2,
    nperseg: int = 128
) -> np.ndarray:
    """
    Much faster CiSSA‐feature extraction by:
      1) downsampling each signal by ds
      2) flattening all component‐signals into one big 2D array
      3) one single Parallel call over ~n_epochs*n_comps workers
      4) precomputing frequency/band masks
    
    Params
    ------
    X_cissa : (n_epochs, n_comps, n_samples)
    ds       : down‐sample factor (keep every ds-th point)
    nperseg  : for welch – smaller blocks are faster
    """
    n_epochs, n_comps, n_samples = X_cissa.shape
    # 1) downsample
    Xds = X_cissa[..., ::ds]
    n_samples_ds = Xds.shape[-1]

    # 2) precompute band masks on dummy
    if bands is None:
        bands = [(1,4),(4,8),(8,13),(13,30),(30,40)]
    f_dummy, _ = welch(np.zeros(n_samples_ds), fs=sfreq, nperseg=nperseg)
    band_masks = [ (f_dummy>=lo)&(f_dummy<hi) for lo,hi in bands ]

    # 3) flatten to (n_epochs*n_comps, n_samples_ds)
    flat = Xds.reshape(-1, n_samples_ds)

    def _compute(sig):
        feats = [
            sig.mean(),
            sig.std(),
            skew(sig),
            kurtosis(sig),
        ]
        # Hjorth
        d1 = np.diff(sig); d2 = np.diff(sig, n=2)
        v0, v1, v2 = np.var(sig), np.var(d1), np.var(d2)
        mob = np.sqrt(v1/v0) if v0>0 else 0.0
        comp = (np.sqrt(v2/v1)/mob) if (v1>0 and mob>0) else 0.0
        feats += [v0, mob, comp]

        # nonlinear
        feats += [
            ant.sample_entropy(sig),
            ant.perm_entropy(sig, order=3, delay=1, normalize=True),
            ant.spectral_entropy(sig, sfreq, method='welch',
                                 nperseg=nperseg, normalize=True),
            ant.higuchi_fd(sig, kmax=10),
            ant.petrosian_fd(sig),
        ]

        # band‐powers
        f, Pxx = welch(sig, fs=sfreq, nperseg=nperseg)
        for mask in band_masks:
            feats.append(np.trapz(Pxx[mask], f[mask]))

        return np.array(feats, dtype=np.float64)

    # 4) single big parallel sweep
    all_feats = Parallel(n_jobs=n_jobs)(
        delayed(_compute)(sig) for sig in flat
    )
    all_feats = np.vstack(all_feats)    # shape: (n_epochs*n_comps, n_feats)
    
    # 5) reshape back to (n_epochs, n_comps * n_feats)
    n_feats = all_feats.shape[1]
    return all_feats.reshape(n_epochs, n_comps * n_feats)
