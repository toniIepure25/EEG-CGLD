# src/eeg_knn_bhho/utils.py
"""
Utility functions for EEG KNN+BHHO pipeline:
- Logging setup
- Data balancing (undersample or SMOTE)
- Train/Test splitting (LOSO or random stratified)
"""
import os
import logging
import numpy as np
from typing import Tuple, List
from sklearn.model_selection import StratifiedShuffleSplit
from imblearn.over_sampling import SMOTE

from eeg_knn_bhho.preprocessing import normalize_epoch


def setup_logging(output_dir: str, experiment_name: str) -> logging.Logger:
    """
    Configure Python logging to write to both console and a log file.

    Parameters
    ----------
    output_dir : str
        Base directory where logs are saved.
    experiment_name : str
        Name of the current experiment; used to name the log file.

    Returns
    -------
    logger : logging.Logger
    """
    logs_dir = os.path.join(output_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    log_file = os.path.join(logs_dir, f"{experiment_name}.log")

    logger = logging.getLogger(experiment_name)
    logger.setLevel(logging.INFO)

    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    fh_formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh.setFormatter(fh_formatter)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch_formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    ch.setFormatter(ch_formatter)

    # Avoid adding multiple handlers if logger already configured
    if not logger.handlers:
        logger.addHandler(fh)
        logger.addHandler(ch)

    return logger


def balance_data(
    X: np.ndarray,
    y: np.ndarray,
    sampling_cfg
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Balance a binary dataset (X, y) using undersampling or SMOTE.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
    y : np.ndarray, shape (n_samples,)
    sampling_cfg : object with attributes:
        - method: 'undersample' or 'smote'
        - random_state: int

    Returns
    -------
    X_bal : np.ndarray, shape (n_balanced, n_features)
    y_bal : np.ndarray, shape (n_balanced,)
    """
    method = sampling_cfg.method.lower()
    rs = sampling_cfg.random_state

    if method == "undersample":
        idx0 = np.where(y == 0)[0]
        idx1 = np.where(y == 1)[0]
        n_min = min(len(idx0), len(idx1))

        rng = np.random.RandomState(rs)
        if len(idx0) > len(idx1):
            idx0_down = rng.choice(idx0, size=n_min, replace=False)
            idx1_down = idx1
        else:
            idx1_down = rng.choice(idx1, size=n_min, replace=False)
            idx0_down = idx0

        idx_bal = np.concatenate([idx0_down, idx1_down])
        rng.shuffle(idx_bal)
        X_bal = X[idx_bal]
        y_bal = y[idx_bal]
        return X_bal, y_bal

    elif method == "smote":
        sm = SMOTE(random_state=rs)
        X_res, y_res = sm.fit_resample(X, y)
        return X_res, y_res

    else:
        raise ValueError(f"Unknown sampling method: {sampling_cfg.method}")


def train_test_split_subjectwise(
    X: np.ndarray,
    y: np.ndarray,
    subject_ids: np.ndarray,
    eval_cfg
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Create train/test splits either using LOSO (Leave-One-Subject-Out) or random stratified split.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
    y : np.ndarray, shape (n_samples,)
    subject_ids : np.ndarray, shape (n_samples,)
        Array containing subject identifier (string or int) for each sample/epoch.
    eval_cfg : object with attributes:
        - strategy: 'LOSO' or 'train_test'
        - train_ratio: float (only if strategy='train_test')
        - seed: int

    Returns
    -------
    splits : List of tuples
        Each tuple is (train_indices, test_indices).
    """
    splits: List[Tuple[np.ndarray, np.ndarray]] = []
    strategy = eval_cfg.strategy.lower()

    if strategy == "loso":
        unique_subjects = np.unique(subject_ids)
        for subj in unique_subjects:
            test_idx = np.where(subject_ids == subj)[0]
            train_idx = np.where(subject_ids != subj)[0]
            splits.append((train_idx, test_idx))

    elif strategy == "train_test":
        sss = StratifiedShuffleSplit(
            n_splits=1,
            test_size=1.0 - eval_cfg.train_ratio,
            random_state=eval_cfg.seed
        )
        for train_idx, test_idx in sss.split(X, y):
            splits.append((train_idx, test_idx))

    else:
        raise ValueError(f"Unknown evaluation strategy: {eval_cfg.strategy}")

    return splits
