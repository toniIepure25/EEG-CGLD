# src/eeg_knn_bhho/feature_normalization.py
"""
Feature normalization utility for EEG KNN+BHHO pipeline.
Provides a scikit-learn compatible transformer that standardizes features.
"""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureNormalizer(BaseEstimator, TransformerMixin):
    """
    Standardize features by removing the mean and scaling to unit variance.

    This wrapper conforms to the scikit-learn Transformer API so it can be
    included in Pipelines or applied independently.
    """
    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, X: np.ndarray, y=None):
        """
        Compute the mean and std to be used for later scaling.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
        """
        X = np.asarray(X, dtype=np.float64)
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)
        # Avoid division by zero
        self.scale_[self.scale_ == 0] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Perform standardization by centering and scaling.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        X_tr : np.ndarray, shape (n_samples, n_features)
        """
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("FeatureNormalizer not fitted yet. Call `fit` first.")
        X = np.asarray(X, dtype=np.float64)
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X: np.ndarray, y=None) -> np.ndarray:
        """
        Fit to data, then transform it.
        """
        return self.fit(X, y).transform(X)
