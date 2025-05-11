import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from src.bhho2 import BinaryHHO

def evaluate_feature_subset(X: np.ndarray, y: np.ndarray, feature_mask: np.ndarray, k: int = 5, cv: int = 10) -> float:
    """
    Evaluate feature subset using KNN and cross-validation.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        Feature matrix.
    y : np.ndarray, shape (n_samples,)
        Labels.
    feature_mask : np.ndarray, shape (n_features,)
        Binary vector indicating selected features (1 = keep).
    k : int
        Number of neighbors in KNN.
    cv : int
        Number of folds in cross-validation.

    Returns
    -------
    accuracy : float
    """
    if feature_mask.sum() == 0:
        return 0.0  # avoid empty subset
    X_selected = X[:, feature_mask == 1]
    clf = KNeighborsClassifier(n_neighbors=k)
    return cross_val_score(clf, X_selected, y, cv=cv, n_jobs=-1).mean()


def run_bhho_feature_selection(
    X: np.ndarray,
    y: np.ndarray,
    pop_size: int = 20,
    max_iter: int = 30,
    transfer: str = "S",
    k: int = 5,
    cv: int = 10
) -> (np.ndarray, float):
    """
    Wrap Binary Harris Hawks Optimization (BHHO) + KNN for feature selection
    with configurable transfer function, neighbor count, and CV folds.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        Feature matrix.
    y : np.ndarray, shape (n_samples,)
        Labels.
    pop_size : int
        Number of hawks in the BHHO population.
    max_iter : int
        Number of optimization iterations.
    transfer : {'S','V'}
        Transfer function type used inside BHHO.
    k : int
        Number of neighbors for KNN.
    cv : int
        Number of folds for cross-validation.

    Returns
    -------
    best_mask : np.ndarray, shape (n_features,)
        Binary vector indicating which features were selected.
    best_score : float
        Best cross-validated accuracy achieved.
    """
    def fitness(mask: np.ndarray) -> float:
        # avoid empty feature‐sets
        if mask.sum() == 0:
            return 0.0
        X_sel = X[:, mask.astype(bool)]
        clf = KNeighborsClassifier(n_neighbors=k)
        # error_score="raise" ensures you see exceptions immediately
        return cross_val_score(
            clf, X_sel, y,
            cv=cv,
            n_jobs=-1,
            error_score="raise"
        ).mean()

    # instantiate BHHO with the given transfer type
    bhho = BinaryHHO(
        fitness_func=fitness,
        n_features=X.shape[1],
        pop_size=pop_size,
        max_iter=max_iter,
        transfer=transfer
    )

    best_mask, best_score = bhho.run()
    return best_mask, best_score
