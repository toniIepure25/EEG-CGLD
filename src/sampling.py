import numpy as np
from sklearn.utils import resample

def undersample(X: np.ndarray, y: np.ndarray, random_state: int = 42):
    """
    Undersample the majority class so that both classes have equal counts.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        Feature matrix.
    y : np.ndarray, shape (n_samples,)
        Binary labels (0 or 1).
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    X_bal : np.ndarray, shape (2 * n_minority, n_features)
        Balanced feature matrix.
    y_bal : np.ndarray, shape (2 * n_minority,)
        Balanced label array.
    """
    # indices of each class
    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]

    # determine minority count
    n_min = min(len(idx0), len(idx1))

    # downsample the larger class
    if len(idx0) > len(idx1):
        idx0_down = resample(idx0, replace=False, n_samples=n_min, random_state=random_state)
        idx1_down = idx1
    else:
        idx1_down = resample(idx1, replace=False, n_samples=n_min, random_state=random_state)
        idx0_down = idx0

    # combine and shuffle
    idx_bal = np.concatenate([idx0_down, idx1_down])
    rng = np.random.RandomState(random_state)
    rng.shuffle(idx_bal)

    X_bal = X[idx_bal]
    y_bal = y[idx_bal]
    return X_bal, y_bal
