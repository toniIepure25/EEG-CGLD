# src/eeg_knn_bhho/classification.py
"""
Classification utilities for EEG KNN+BHHO pipeline.
Includes training final KNN and plotting/saving confusion matrices.
"""
import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from typing import Tuple, List


def train_final_knn(
    X: np.ndarray,
    y: np.ndarray,
    n_neighbors: int = 5
) -> KNeighborsClassifier:
    """
    Train a KNN classifier on the full dataset.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_selected_features)
    y : np.ndarray, shape (n_samples,)
    n_neighbors : int
        Number of neighbors.

    Returns
    -------
    clf : trained KNeighborsClassifier
    """
    clf = KNeighborsClassifier(n_neighbors=n_neighbors, weights="distance")
    clf.fit(X, y)
    return clf


def plot_and_save_confusion(
    clf: KNeighborsClassifier,
    X: np.ndarray,
    y: np.ndarray,
    labels: List[str] = None,
    out_path: str = None
) -> np.ndarray:
    """
    Compute, plot, and save a confusion matrix.

    Parameters
    ----------
    clf : classifier with a predict() method
    X : np.ndarray, shape (n_samples, n_features)
    y : np.ndarray, shape (n_samples,)
    labels : list of str, default=None
        Class labels to display on axes.
    out_path : str, optional
        File path (including filename) to save the figure as PNG.

    Returns
    -------
    cm : np.ndarray
        The confusion matrix array.
    """
    y_pred = clf.predict(X)
    cm = confusion_matrix(y, y_pred)

    disp = ConfusionMatrixDisplay(cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap='Blues')
    plt.title('Confusion Matrix')
    plt.tight_layout()

    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fig.savefig(out_path, dpi=300)
    plt.close(fig)

    return cm
