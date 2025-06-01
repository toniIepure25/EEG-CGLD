import os
import joblib
import numpy as np
import mlflow
import hydra
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier

from eeg_knn_bhho.data_loading import load_mat_epochs, load_stew_epochs
from eeg_knn_bhho.preprocessing import BandpassNotchFilter, EpochNormalizer
from eeg_knn_bhho.ica_preprocessing import ICACleaner
from eeg_knn_bhho.decomposition import CISSADecomposer
from eeg_knn_bhho.features import CiSSALightFeatureExtractor
from eeg_knn_bhho.connectivity import ConnectivityExtractor
from eeg_knn_bhho.cnn_encoder import CNNEncoderTransformer
from eeg_knn_bhho.feature_selection import run_bhho_feature_selection
from eeg_knn_bhho.classification import train_final_knn, plot_and_save_confusion
from eeg_knn_bhho.utils import setup_logging, balance_data

@hydra.main(config_path="configs", config_name="default")
def main(cfg: DictConfig):
    """
    Main entrypoint for the EEG‐KNN experiment. 
    For each dataset ("mat" and "stew"), this script:
      1. Loads raw epochs (MAT/STEW).
      2. Applies ICA artifact removal.
      3. Applies notch + bandpass filtering.
      4. Performs per‐epoch normalization.
      5. Runs CiSSA decomposition.
      6. Extracts hand‐crafted (CiSSA) features.
      7. Extracts connectivity features.
      8. Computes CNN embeddings.
      9. Concatenates all feature types.
     10. Balances the classes.
     11. Runs nested cross‐validation (outer/inner) with BHHO + KNN.
     12. Saves train/test splits (as .npy), models, and confusion matrices.
     13. Logs parameters and metrics to MLflow.
    """
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

    # 3. Configure preprocessing transformers
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

    # 4. Initialize CiSSA decomposer
    logger.info("Initializing CiSSA decomposer")
    decomposer = CISSADecomposer(n_imfs=cfg.ssa.n_imfs, n_jobs=cfg.ssa.n_jobs)

    # 5. Initialize hand‐crafted feature extractors
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

    # 6. Process each dataset ("mat", "stew") separately
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

        logger.info(f"--- Processing {dataset.upper()} dataset ---")

        # 6.1 ICA artifact removal
        X_ica = ica_cleaner.fit(X_raw).transform(X_raw)
        logger.info(f"ICA-cleaned: {X_ica.shape}")

        # 6.2 Notch + bandpass filtering
        X_filt = bp_filter.transform(X_ica)
        logger.info(f"Filtered: {X_filt.shape}")

        # 6.3 Per‐epoch normalization
        X_clean = normalizer.transform(X_filt)
        logger.info(f"Normalized: {X_clean.shape}")

        # 6.4 CiSSA decomposition
        logger.info("Performing CiSSA decomposition")
        X_decomp = decomposer.transform(X_clean)
        n_epochs, n_comps, n_samples = X_decomp.shape
        logger.info(f"CiSSA: {n_comps} components per epoch, {n_samples} samples each")

        # 7. Hand‐crafted feature extraction (CiSSA outputs)
        logger.info("Extracting hand‐crafted features from CiSSA outputs")
        X_feats = extractor.transform(X_decomp)
        logger.info(f"Hand-crafted features: {X_feats.shape[1]} features per epoch")

        # 8. Connectivity feature extraction (on normalized epochs)
        logger.info("Extracting connectivity features")
        conn_extractor = ConnectivityExtractor(
            sfreq=sfreq,
            nperseg=cfg.feature_extraction.nperseg,
            n_jobs=cfg.feature_extraction.n_jobs
        )
        X_conn = conn_extractor.transform(X_clean)
        logger.info(f"Connectivity features: {X_conn.shape[1]} features per epoch")

        # 9. Concatenate hand‐crafted + connectivity features
        X_combined = np.hstack([X_feats, X_conn])
        logger.info(f"Combined features shape (CiSSA + connectivity): {X_combined.shape}")

        # 10. CNN embedding extraction (on normalized epochs)
        logger.info("Computing CNN embeddings")
        cnn_cfg = cfg.cnn  # expects keys: hidden_channels, kernel_sizes, pool_sizes, embedding_dim, device, batch_size
        cnn_transformer = CNNEncoderTransformer(
            input_channels=X_clean.shape[1],    # # of EEG channels
            input_samples=X_clean.shape[2],     # # of time samples per epoch
            hidden_channels=list(cnn_cfg.hidden_channels),
            kernel_sizes=list(cnn_cfg.kernel_sizes),
            pool_sizes=list(cnn_cfg.pool_sizes),
            embedding_dim=cnn_cfg.embedding_dim,
            device=cnn_cfg.device,
            batch_size=cnn_cfg.batch_size
        )
        X_emb = cnn_transformer.transform(X_clean)
        logger.info(f"CNN embeddings shape: {X_emb.shape}")

        # 11. Concatenate all features: CiSSA + connectivity + CNN embeddings
        X_all = np.hstack([X_combined, X_emb])
        logger.info(f"All features combined shape: {X_all.shape}")

        # 12. Remove NaNs and balance classes
        X_all = np.nan_to_num(X_all)
        X_bal, y_bal = balance_data(X_all, y_raw, cfg.sampling)
        logger.info(f"Balanced data: {X_bal.shape[0]} samples")

        # 13. Nested cross‐validation (outer/inner) for BHHO + KNN
        logger.info("Starting nested cross‐validation")
        outer_cv = StratifiedKFold(
            n_splits=cfg.evaluation.outer_folds,
            shuffle=True,
            random_state=cfg.seed
        )
        outer_fold_accuracies = []

        for outer_idx, (train_outer_idx, test_outer_idx) in enumerate(
            outer_cv.split(X_bal, y_bal)
        ):
            # 13.1 Split outer train/test
            X_train_outer = X_bal[train_outer_idx]
            y_train_outer = y_bal[train_outer_idx]
            X_test_outer  = X_bal[test_outer_idx]
            y_test_outer  = y_bal[test_outer_idx]

            logger.info(
                f"Outer fold {outer_idx} – training BHHO on {len(train_outer_idx)} samples"
            )

            # 13.2 Create & save a folder for this outer fold
            fold_dir = os.path.join(
                cfg.logging.output_dir,
                "splits", 
                dataset, 
                f"outer_fold_{outer_idx}"
            )
            os.makedirs(fold_dir, exist_ok=True)

            # 13.3 Save train/test splits as .npy
            np.save(os.path.join(fold_dir, "X_train.npy"), X_train_outer)
            np.save(os.path.join(fold_dir, "y_train.npy"), y_train_outer)
            np.save(os.path.join(fold_dir, "X_test.npy"),  X_test_outer)
            np.save(os.path.join(fold_dir, "y_test.npy"),  y_test_outer)
            logger.info(f"    Saved train/test splits to {fold_dir}")

            # 13.4 Inner BHHO feature selection (CV inside run_bhho_feature_selection)
            mask, inner_cv_score = run_bhho_feature_selection(
                X_train_outer, y_train_outer, cfg, seed=cfg.seed + outer_idx
            )
            logger.info(
                f"    Inner BHHO CV accuracy = {inner_cv_score:.4f}, selected {mask.sum()} features"
            )

            # 13.5 Train final KNN on the full outer train set with selected features
            X_tr_sel = X_train_outer[:, mask.astype(bool)]
            knn = train_final_knn(X_tr_sel, y_train_outer, n_neighbors=cfg.classifier.k)

            # 13.6 Evaluate on the outer test set
            X_te_sel = X_test_outer[:, mask.astype(bool)]
            y_pred_outer = knn.predict(X_te_sel)
            outer_acc = (y_pred_outer == y_test_outer).mean()
            outer_fold_accuracies.append(outer_acc)
            logger.info(f"    Outer fold {outer_idx} test accuracy = {outer_acc:.4f}")

            # 13.7 Save confusion matrix for this outer fold
            cm_dir = os.path.join(
                cfg.logging.output_dir,
                "figures",
                dataset,
                f"outer_fold_{outer_idx}"
            )
            os.makedirs(cm_dir, exist_ok=True)
            cm_path = os.path.join(cm_dir, "confusion_matrix.png")
            plot_and_save_confusion(
                knn,
                X_te_sel,
                y_test_outer,
                labels=["Rest", "Task"],
                out_path=cm_path
            )
            logger.info(f"    Saved confusion matrix to {cm_path}")

            # 13.8 (Optional) Save the trained model & mask
            model_dir = os.path.join(
                cfg.logging.output_dir,
                "models",
                dataset,
                f"outer_fold_{outer_idx}"
            )
            os.makedirs(model_dir, exist_ok=True)
            model_path = os.path.join(model_dir, "knn_model.joblib")
            joblib.dump({"model": knn, "mask": mask}, model_path)
            logger.info(f"    Saved model & mask to {model_path}")

        # 14. Compute & log nested‐CV average accuracy
        avg_outer_acc = np.mean(outer_fold_accuracies)
        logger.info(f"{dataset.upper()}: Nested‐CV average test accuracy = {avg_outer_acc:.4f}")

        # 15. Log to MLflow
        with mlflow.start_run(run_name=f"nested_cv_{dataset}"):
            # Log BHHO (feature_selection) hyperparameters
            mlflow.log_params(OmegaConf.to_container(cfg.feature_selection, resolve=True))
            # Log CNN hyperparameters
            mlflow.log_params({
                "cnn_hidden_channels": tuple(cfg.cnn.hidden_channels),
                "cnn_kernel_sizes": tuple(cfg.cnn.kernel_sizes),
                "cnn_pool_sizes": tuple(cfg.cnn.pool_sizes),
                "cnn_embedding_dim": cfg.cnn.embedding_dim,
                "cnn_batch_size": cfg.cnn.batch_size,
                "cnn_device": cfg.cnn.device
            })
            # Log nested‐CV metric
            mlflow.log_metric(f"nested_cv_avg_accuracy_{dataset}", float(avg_outer_acc))

            # (Optional) Log last‐fold model & confusion matrix as artifacts:
            # mlflow.log_artifact(model_path, artifact_path=f"models/{dataset}")
            # mlflow.log_artifact(cm_path,    artifact_path=f"figures/{dataset}")

    logger.info("Experiment complete.")

if __name__ == "__main__":
    main()
