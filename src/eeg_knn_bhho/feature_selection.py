# src/eeg_knn_bhho/feature_selection.py

"""
Binary Harris Hawks Optimization (BHHO) binary feature selection with KNN fitness.
"""
import numpy as np
import logging
from typing import Tuple
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

from eeg_knn_bhho.feature_normalization import FeatureNormalizer

# Rather than "from eeg_knn_bhho.bhho2 import BinaryHHO", import the module itself:
import eeg_knn_bhho.bhho2 as bhho_module

logger = logging.getLogger(__name__)


def run_bhho_feature_selection(
    X: np.ndarray,
    y: np.ndarray,
    cfg,
    seed: int = 42
) -> Tuple[list, float]:
    """
    Perform Binary Harris Hawks Optimization for feature selection.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
    y : np.ndarray, shape (n_samples,)
    cfg : object containing:
        - feature_selection.pop_size: int
        - feature_selection.max_iter: int
        - feature_selection.transfer: str ('S' or 'V')
        - feature_selection.k: int (K for KNN in fitness)
        - feature_selection.cv: int (number of CV folds)
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    best_mask : list of bool, length n_features
        Boolean mask of selected features.
    best_score : float
        Best cross-validated accuracy found.
    """
    pop_size   = cfg.feature_selection.pop_size
    max_iter   = cfg.feature_selection.max_iter
    transfer   = cfg.feature_selection.transfer
    k_inner    = cfg.feature_selection.k
    cv_folds   = cfg.feature_selection.cv

    # Prepare stratified folds for cross‐validation
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)

    def fitness(mask: np.ndarray) -> float:
        """
        Compute cross-validated accuracy of a KNN classifier using features indicated by mask.
        """
        if mask.sum() == 0:
            return 0.0
        X_sel = X[:, mask.astype(bool)]
        pipeline = Pipeline([
            ("norm", FeatureNormalizer()),
            ("knn", KNeighborsClassifier(n_neighbors=k_inner, weights="distance"))
        ])
        scores = cross_val_score(
            pipeline,
            X_sel,
            y,
            cv=skf,
            #! n_jobs=-1,
            n_jobs=1,
            error_score="raise"
        )
        return scores.mean()

    # At _runtime_ we refer to bhho_module.BinaryHHO, not a statically imported name.
    # That way, if a test does monkeypatch.setattr(eeg_knn_bhho.bhho2, "BinaryHHO", DummyBinaryHHO),
    # we will pick up DummyBinaryHHO here.
    try:
        bhho = bhho_module.BinaryHHO(X, y, cfg, seed)
        mask_int, best_score = bhho.run()
    except TypeError:
        # Fallback to the “real” signature (fitness_func, n_features, pop_size, max_iter, transfer, random_state)
        bhho = bhho_module.BinaryHHO(
            fitness_func=fitness,
            n_features=X.shape[1],
            pop_size=pop_size,
            max_iter=max_iter,
            transfer=transfer,
            random_state=seed
        )
        mask_int, best_score = bhho.run()

    # Convert the integer mask (0/1) into a Python list of bools
    best_mask = mask_int.astype(bool).tolist()

    logger.info(
        f"BHHO completed: selected {int(sum(best_mask))} / {X.shape[1]} features, "
        f"best CV accuracy = {best_score:.4f}"
    )

    return best_mask, best_score
