import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def evaluate_selected_features(X, y, feature_mask, k=5, cv=10):
    """
    Evaluate classifier on selected features using KNN and report metrics.

    Parameters
    ----------
    X : np.ndarray
        Full feature matrix (n_samples, n_features)
    y : np.ndarray
        Labels
    feature_mask : np.ndarray
        Binary mask for selected features
    k : int
        Number of neighbors for KNN
    cv : int
        Cross-validation folds

    Returns
    -------
    report : str
        Text summary of classification report
    cm : np.ndarray
        Confusion matrix
    """
    X_selected = X[:, feature_mask == 1]
    clf = KNeighborsClassifier(n_neighbors=k)
    y_pred = cross_val_predict(clf, X_selected, y, cv=cv)

    report = classification_report(y, y_pred, target_names=["Rest", "Task"])
    cm = confusion_matrix(y, y_pred)
    return report, cm


def plot_confusion_matrix(cm, labels=["Rest", "Task"]):
    """
    Plot confusion matrix using seaborn heatmap.

    Parameters
    ----------
    cm : np.ndarray
        Confusion matrix
    labels : list
        List of class names
    """
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()
