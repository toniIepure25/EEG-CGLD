# tests/test_classification_and_utils.py
import numpy as np
from eeg_knn_bhho.classification import train_final_knn, plot_and_save_confusion
from eeg_knn_bhho.utils import balance_data, train_test_split_subjectwise, setup_logging
import os

def test_train_knn_and_confusion(tmp_path):
    # Create a trivial dataset: two classes perfectly separable
    X = np.vstack([
        np.zeros((10, 2)),     # class 0
        np.ones((10, 2)) * 5   # class 1
    ])
    y = np.array([0]*10 + [1]*10)
    clf = train_final_knn(X, y, n_neighbors=1)
    y_pred = clf.predict(X)
    assert (y_pred == y).all()

    # Confusion matrix plot save
    out_png = tmp_path / "cm.png"
    cm = plot_and_save_confusion(clf, X, y, labels=["A", "B"], out_path=str(out_png))
    assert cm.shape == (2, 2)
    assert out_png.exists()

def test_balance_split_and_logging(tmp_path):
    # Create imbalanced data: 3 samples of class 0, 10 samples of class 1
    X = np.arange(13).reshape(13,1)
    y = np.array([0,0,0] + [1]*10)
    class DummyCfg:
        method = "undersample"
        random_state = 0

    X_bal, y_bal = balance_data(X, y, DummyCfg)
    # After undersample, should have 3 samples of each class => total 6
    assert len(y_bal) == 6
    assert (y_bal == 0).sum() == 3 and (y_bal == 1).sum() == 3

    # train_test_split_subjectwise
    # Simulate subject IDs: [0,0,0,1,1,1,...]
    subject_ids = np.array([0,0,0] + [1,1,1] + [2,2,2])
    X2 = np.arange(9).reshape(9,1)
    y2 = np.array([0,1,0] * 3)
    class EvalCfg:
        strategy = "LOSO"
        train_ratio = 0.8
        seed = 0

    splits = train_test_split_subjectwise(X2, y2, subject_ids, EvalCfg)
    # For LOSO with 3 unique subjects, expect 3 splits
    assert len(splits) == 3

    # Logging
    logger = setup_logging(str(tmp_path), "test_exp")
    logger.info("This is a test")
    # Check that log file was created
    log_file = tmp_path / "logs" / "test_exp.log"
    assert log_file.exists()
