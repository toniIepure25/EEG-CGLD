# import numpy as np
# from numpy.lib.stride_tricks import sliding_window_view
# from sklearn.utils.extmath import randomized_svd
# from numba import njit, prange
# from joblib import Parallel, delayed

# @njit
# def hankelize_numba(X_elem, N, L, K):
#     """
#     Hankelization (diagonal averaging) of X_elem to reconstruct a time series.
#     Optimized with Numba.

#     X_elem: (L, K) matrix
#     N: length of original signal
#     L: window size
#     K: N - L + 1
#     Returns: comp of length N
#     """
#     comp = np.empty(N, dtype=np.float64)
#     for k in range(N):
#         total = 0.0
#         count = 0
#         # diagonal index offset
#         offset = k - (L - 1)
#         for i in range(L):
#             j = offset + i
#             if 0 <= j < K:
#                 total += X_elem[L - 1 - i, j]
#                 count += 1
#         comp[k] = total / count if count > 0 else 0.0
#     return comp


# def ssa_1d_fast(signal: np.ndarray, window_size: int = 30, n_components: int = 3, n_iter: int = 5, random_state: int = 0):
#     """
#     Fast SSA for 1D signal using randomized SVD and Numba Hankelization.

#     Parameters
#     ----------
#     signal : 1D array (n_samples,)
#     window_size : L, must be < n_samples/2
#     n_components : number of SSA components to extract
#     n_iter : iterations for randomized SVD
#     random_state : seed for reproducibility

#     Returns
#     -------
#     components : array (n_components, n_samples)
#     """
#     N = signal.shape[0]
#     L = window_size
#     K = N - L + 1
#     # 1) Embedding
#     X = sliding_window_view(signal, window_shape=L).T  # shape (L, K)
#     # 2) Truncated randomized SVD
#     U, S, Vt = randomized_svd(X, n_components=n_components, n_iter=n_iter, random_state=random_state)
#     # 3) Reconstruct and hankelize each component
#     comps = np.empty((n_components, N), dtype=np.float64)
#     for idx in range(n_components):
#         Xc = S[idx] * np.outer(U[:, idx], Vt[idx])
#         comps[idx, :] = hankelize_numba(Xc, N, L, K)
#     return comps


# def decompose_epoch_fast(epoch: np.ndarray, window_size: int = 30, n_imfs: int = 3, n_jobs: int = -1):
#     """
#     Decompose multichannel EEG epoch into SSA components in parallel across channels.

#     Parameters
#     ----------
#     epoch : (n_channels, n_samples)
#     window_size : SSA window size L
#     n_imfs : number of components per channel
#     n_jobs : number of parallel jobs (use -1 for all cores)

#     Returns
#     -------
#     decomposed : (n_channels * n_imfs, n_samples)
#     """
#     # parallelize SSA across channels
#     results = Parallel(n_jobs=n_jobs)(
#         delayed(ssa_1d_fast)(epoch[ch], window_size, n_imfs)
#         for ch in range(epoch.shape[0])
#     )
#     # stack each channel's components
#     return np.vstack(results)


# def batch_decompose(X: np.ndarray, window_size: int = 30, n_imfs: int = 3, n_jobs: int = -1):
#     """
#     Apply SSA decomposition to all epochs in X efficiently.

#     Parameters
#     ----------
#     X : (n_epochs, n_channels, n_samples)
#     window_size, n_imfs, n_jobs : as above

#     Returns
#     -------
#     X_ssa : (n_epochs, n_channels * n_imfs, n_samples)
#     """
#     # parallelize across epochs
#     decomposed = Parallel(n_jobs=n_jobs)(
#         delayed(decompose_epoch_fast)(epoch, window_size, n_imfs)
#         for epoch in X
#     )
#     return np.array(decomposed)
import numpy as np

def cissa_1d(signal: np.ndarray, n_imfs: int = 3) -> np.ndarray:
    """
    Circulant SSA (CiSSA) decomposition: approximate Hankel matrix by a circulant matrix
    whose eigenvectors are the DFT basis. Extract the top n_imfs narrow-band components.

    Parameters
    ----------
    signal : 1D numpy array of length N
    n_imfs : number of components to extract (dominant frequencies)

    Returns
    -------
    comps : array, shape (n_imfs, N)
        Reconstructed time-series components.
    """
    N = signal.shape[0]
    # 1) Compute FFT of the signal
    X = np.fft.fft(signal)
    # 2) Identify n_imfs largest-magnitude frequency bins (excluding DC)
    magnitudes = np.abs(X)
    # ignore negative-frequency mirror for selection
    half = N // 2 + 1
    idxs = np.argsort(magnitudes[:half])[::-1]
    selected = idxs[:n_imfs]

    comps = np.zeros((n_imfs, N), dtype=np.float64)
    for i, k in enumerate(selected):
        # build mask for frequency k (and its mirror at N-k if k>0)
        mask = np.zeros(N, dtype=bool)
        mask[k] = True
        if k > 0 and k < half:
            mask[N - k] = True
        # apply mask and reconstruct component
        Xc = np.zeros_like(X)
        Xc[mask] = X[mask]
        comp = np.fft.ifft(Xc)
        comps[i] = np.real(comp)
    return comps


def decompose_epoch_cissa(epoch: np.ndarray, n_imfs: int = 3) -> np.ndarray:
    """
    Decompose a multi-channel epoch into CiSSA components across channels.

    Parameters
    ----------
    epoch : array, shape (n_channels, n_samples)
    n_imfs : number of components per channel

    Returns
    -------
    stacked : array, shape (n_channels * n_imfs, n_samples)
    """
    n_ch, _ = epoch.shape
    comps = [cissa_1d(epoch[ch], n_imfs) for ch in range(n_ch)]
    # stack channels' components vertically
    return np.vstack(comps)


def batch_decompose_cissa(
    X: np.ndarray,
    n_imfs: int = 3,
    n_jobs: int = 1
) -> np.ndarray:
    """
    Apply CiSSA decomposition to all epochs in X efficiently.

    Parameters
    ----------
    X : array, shape (n_epochs, n_channels, n_samples)
    n_imfs : components per channel
    n_jobs : number of parallel jobs (joblib) or 1

    Returns
    -------
    X_cissa : array, shape (n_epochs, n_channels * n_imfs, n_samples)
    """
    from joblib import Parallel, delayed
    decomposed = Parallel(n_jobs=n_jobs)(
        delayed(decompose_epoch_cissa)(epoch, n_imfs)
        for epoch in X
    )
    return np.array(decomposed)
