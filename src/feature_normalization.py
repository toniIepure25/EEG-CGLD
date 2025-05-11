import numpy as np

class FeatureNormalizer:
    """
    Z-score feature normalizer for datasets.
    Computes per-feature mean and std on fit, and applies transformation.
    """
    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X: np.ndarray):
        """
        Compute mean and std per feature.

        Parameters
        ----------
        X : array, shape (n_samples, n_features)
        """
        # ensure float64 for precision
        X = X.astype(np.float64, copy=False)
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        # avoid division by zero
        self.std_[self.std_ < 1e-8] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply z-score normalization using stored parameters.

        Parameters
        ----------
        X : array, shape (n_samples, n_features)

        Returns
        -------
        X_norm : array, shape (n_samples, n_features)
        """
        if self.mean_ is None or self.std_ is None:
            raise ValueError("Normalizer must be fit before calling transform.")
        # ensure float64 for precision
        X = X.astype(np.float64, copy=False)
        return (X - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Convenience: fit and transform in one step.
        """
        self.fit(X)
        return self.transform(X)
