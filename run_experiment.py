import os
import joblib
import numpy as np
import mlflow
import hydra
from omegaconf import DictConfig, OmegaConf
from sklearn.neighbors import KNeighborsClassifier

from eeg_knn_bhho.data_loading import load_mat_epochs, load_stew_epochs
from eeg_knn_bhho.preprocessing import BandpassNotchFilter, EpochNormalizer
from eeg_knn_bhho.ica_preprocessing import ICACleaner
from eeg_knn_bhho.decomposition import CISSADecomposer
from eeg_knn_bhho.features import CiSSALightFeatureExtractor
from eeg_knn_bhho.feature_selection import run_bhho_feature_selection
from eeg_knn_bhho.classification import train_final_knn, plot_and_save_confusion
from eeg_knn_bhho.utils import setup_logging, balance_data, train_test_split_subjectwise

@hydra.main(config_path="configs", config_name="default")
def main(cfg: DictConfig):
    # 1. Setup logging & MLflow
    logger = setup_logging(cfg.logging.output_dir, cfg.logging.experiment_name)
    logger.info("Starting experiment")
    mlflow.set_tracking_uri(cfg.logging.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.logging.experiment_name)

    # 2. Load datasets
    logger.info("Loading MAT dataset")
    X_mat, y_mat = load_mat_epochs(
        cfg.data.mat_dir,
        epoch_length=cfg.preprocess.epoch_length,
        resample_sfreq=cfg.preprocess.sfreq_mat,
        overlap=cfg.preprocess.overlap
    )
    logger.info(f"MAT loaded: {X_mat.shape[0]} epochs, {X_mat.shape[1]} channels")

    logger.info("Loading STEW dataset")
    X_stew, y_stew = load_stew_epochs(
        cfg.data.stew_dir,
        epoch_length=cfg.preprocess.epoch_length,
        sfreq=cfg.preprocess.sfreq_stew,
        overlap=cfg.preprocess.overlap
    )
    logger.info(f"STEW loaded: {X_stew.shape[0]} epochs, {X_stew.shape[1]} channels")

    # 3. Preprocessing pipeline
    logger.info("Configuring preprocessing transformers")

    ica_cleaner_mat = ICACleaner(
        sfreq=cfg.preprocess.sfreq_mat,
        n_components=cfg.preprocess.ica_n_components,
        kurtosis_thresh=cfg.preprocess.ica_kurtosis_thresh,
        random_state=cfg.seed
    )
    ica_cleaner_stew = ICACleaner(
        sfreq=cfg.preprocess.sfreq_stew,
        n_components=cfg.preprocess.ica_n_components,
        kurtosis_thresh=cfg.preprocess.ica_kurtosis_thresh,
        random_state=cfg.seed
    )
    bp_filter_mat = BandpassNotchFilter(
        band_low=cfg.preprocess.band_low,
        band_high=cfg.preprocess.band_high,
        notch_freq=cfg.preprocess.notch_freq,
        notch_Q=cfg.preprocess.notch_Q,
        sfreq=cfg.preprocess.sfreq_mat,
        order=cfg.preprocess.filter_order
    )
    bp_filter_stew = BandpassNotchFilter(
        band_low=cfg.preprocess.band_low,
        band_high=cfg.preprocess.band_high,
        notch_freq=cfg.preprocess.notch_freq,
        notch_Q=cfg.preprocess.notch_Q,
        sfreq=cfg.preprocess.sfreq_stew,
        order=cfg.preprocess.filter_order
    )
    normalizer = EpochNormalizer()

    # 4. Decomposer
    logger.info("Initializing CiSSA decomposer")
    decomposer = CISSADecomposer(n_imfs=cfg.ssa.n_imfs, n_jobs=cfg.ssa.n_jobs)

    # 5. Feature extractors
    logger.info("Initializing CiSSA feature extractors")
    extractor_mat = CiSSALightFeatureExtractor(
        sfreq=cfg.preprocess.sfreq_mat,
        bands=cfg.feature_extraction.bands,
        ds=cfg.feature_extraction.ds,
        nperseg=cfg.feature_extraction.nperseg,
        n_jobs=cfg.feature_extraction.n_jobs
    )
    extractor_stew = CiSSALightFeatureExtractor(
        sfreq=cfg.preprocess.sfreq_stew,
        bands=cfg.feature_extraction.bands,
        ds=cfg.feature_extraction.ds,
        nperseg=cfg.feature_extraction.nperseg,
        n_jobs=cfg.feature_extraction.n_jobs
    )

    # Process each dataset separately
    for dataset in ["mat", "stew"]:
        if dataset == "mat":
            X_raw, y_raw = X_mat, y_mat
            ica_cleaner = ica_cleaner_mat
            bp_filter = bp_filter_mat
            extractor = extractor_mat
            sfreq = cfg.preprocess.sfreq_mat
        else:
            X_raw, y_raw = X_stew, y_stew
            ica_cleaner = ica_cleaner_stew
            bp_filter = bp_filter_stew
            extractor = extractor_stew
            sfreq = cfg.preprocess.sfreq_stew

        logger.info(f"Preprocessing {dataset.upper()} dataset")
        # 6. ICA artifact removal
        X_ica = ica_cleaner.fit(X_raw).transform(X_raw)
        logger.info(f"ICA-cleaned: {X_ica.shape}")

        # 7. Notch + bandpass filtering
        X_filt = bp_filter.transform(X_ica)
        logger.info(f"Filtered: {X_filt.shape}")

        # 8. Per-epoch normalization
        X_clean = normalizer.transform(X_filt)
        logger.info(f"Normalized: {X_clean.shape}")

        # 9. CiSSA decomposition
        logger.info("Performing CiSSA decomposition")
        X_decomp = decomposer.transform(X_clean)
        n_epochs, n_comps, n_samples = X_decomp.shape
        logger.info(f"CiSSA: {n_comps} components per epoch, {n_samples} samples each")

        # 10. Feature extraction
        logger.info("Extracting features from CiSSA outputs")
        X_feats = extractor.transform(X_decomp)
        logger.info(f"Extracted features: {X_feats.shape[1]} features per epoch")

        # 11. Remove NaNs and balance classes
        X_feats = np.nan_to_num(X_feats)
        X_bal, y_bal = balance_data(X_feats, y_raw, cfg.sampling)
        logger.info(f"Balanced data: {X_bal.shape[0]} samples")

        # 12. Train/Test split (subject-wise or random)
        logger.info("Creating train/test splits")
        subject_ids = None
        if hasattr(cfg.data, "subject_ids") and cfg.data.subject_ids is not None:
            subject_ids = np.array(cfg.data.subject_ids)
        else:
            subject_ids = np.array([-1] * X_bal.shape[0])

        splits = train_test_split_subjectwise(X_bal, y_bal, subject_ids, cfg.evaluation)
        logger.info(f"Number of splits: {len(splits)}")

        # 13. Feature selection + classification per split
        fold_accuracies = []
        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            X_train, y_train = X_bal[train_idx], y_bal[train_idx]
            X_test, y_test = X_bal[test_idx], y_bal[test_idx]

            logger.info(f"Fold {fold_idx}: running BHHO feature selection")
            mask, cv_score = run_bhho_feature_selection(X_train, y_train, cfg, seed=cfg.seed + fold_idx)
            logger.info(f"Fold {fold_idx}: BHHO CV acc = {cv_score:.4f}, selected {mask.sum()} features")

            # Train final KNN on selected features
            X_tr_sel = X_train[:, mask.astype(bool)]
            knn = train_final_knn(X_tr_sel, y_train, n_neighbors=cfg.classifier.k)

            # Save model + mask
            model_dir = os.path.join(cfg.logging.output_dir, "models", dataset, f"fold_{fold_idx}")
            os.makedirs(model_dir, exist_ok=True)
            model_path = os.path.join(model_dir, "knn_model.joblib")
            joblib.dump({"model": knn, "mask": mask}, model_path)
            logger.info(f"Saved model to {model_path}")

            # Evaluate on test set
            X_te_sel = X_test[:, mask.astype(bool)]
            y_pred = knn.predict(X_te_sel)
            acc = (y_pred == y_test).mean()
            fold_accuracies.append(acc)
            logger.info(f"Fold {fold_idx}: Test accuracy = {acc:.4f}")

            # Plot and save confusion matrix
            cm_dir = os.path.join(cfg.logging.output_dir, "figures", dataset, f"fold_{fold_idx}")
            os.makedirs(cm_dir, exist_ok=True)
            cm_path = os.path.join(cm_dir, "confusion_matrix.png")
            plot_and_save_confusion(knn, X_te_sel, y_test, labels=["Rest", "Task"], out_path=cm_path)
            logger.info(f"Saved confusion matrix to {cm_path}")

        avg_acc = np.mean(fold_accuracies)
        logger.info(f"{dataset.upper()} dataset: Average test accuracy across folds = {avg_acc:.4f}")

        # Log to MLflow
        with mlflow.start_run(run_name=f"feature_selection_{dataset}"):
            mlflow.log_params(OmegaConf.to_container(cfg.feature_selection, resolve=True))
            mlflow.log_metric(f"avg_test_accuracy_{dataset}", float(avg_acc))
            mlflow.log_artifact(model_path, artifact_path=f"models/{dataset}")
            mlflow.log_artifact(cm_path, artifact_path=f"figures/{dataset}")

    logger.info("Experiment complete.")

if __name__ == "__main__":
    main()
