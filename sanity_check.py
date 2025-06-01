# sanity_check_phase3.py
import os
import numpy as np
from eeg_knn_bhho.data_loading import load_mat_epochs
from eeg_knn_bhho.ica_preprocessing import ICACleaner
from eeg_knn_bhho.preprocessing import BandpassNotchFilter, EpochNormalizer
from eeg_knn_bhho.decomposition import CISSADecomposer
from eeg_knn_bhho.features import CiSSALightFeatureExtractor
from eeg_knn_bhho.connectivity import ConnectivityExtractor
from eeg_knn_bhho.cnn_encoder import CNNEncoderTransformer

def main():
    project_dir = os.environ.get("PROJECT_DIR", ".")
    mat_dir = os.path.join(project_dir, "data", "raw", "MAT")

    X_mat, y_mat = load_mat_epochs(mat_dir, epoch_length=1.0, resample_sfreq=128.0, overlap=0.5)
    print("Loaded:", X_mat.shape)

    # ICA
    cleaner = ICACleaner(sfreq=128.0, n_components=3, kurtosis_thresh=5.0, random_state=0)
    X_ica = cleaner.fit(X_mat).transform(X_mat)
    print("ICA:", X_ica.shape)

    # Filter + Normalize
    bp = BandpassNotchFilter(1.0, 40.0, 50.0, 30.0, 128.0, 5)
    X_filt = bp.transform(X_ica)
    norm = EpochNormalizer()
    X_norm = norm.transform(X_filt)
    print("Norm:", X_norm.shape)

    # CiSSA + Features
    decomposer = CISSADecomposer(n_imfs=3, n_jobs=1)
    X_decomp = decomposer.transform(X_norm)
    feat_ext = CiSSALightFeatureExtractor(sfreq=128.0, bands=[(1,4),(4,8),(8,13),(13,30),(30,40)], ds=4, nperseg=64, n_jobs=1)
    X_feats = feat_ext.transform(X_decomp)
    print("CiSSA feats:", X_feats.shape)

    # Connectivity
    conn_ext = ConnectivityExtractor(sfreq=128.0, nperseg=64, n_jobs=1)
    X_conn = conn_ext.transform(X_norm)
    print("Connectivity:", X_conn.shape)

    # Combine
    X_combined = np.hstack([X_feats, X_conn])
    print("Combined hand-crafted + connectivity:", X_combined.shape)

    # CNN embeddings
    n_ch, n_samp = X_norm.shape[1], X_norm.shape[2]
    cnn_trans = CNNEncoderTransformer(
        input_channels=n_ch,
        input_samples=n_samp,
        hidden_channels=[16, 32],
        kernel_sizes=[7, 5],
        pool_sizes=[2, 2],
        embedding_dim=64,
        device="cpu",
        batch_size=4
    )
    X_emb = cnn_trans.transform(X_norm)
    print("CNN embeddings:", X_emb.shape)

    # Final combined
    X_all = np.hstack([X_combined, X_emb])
    print("All features:", X_all.shape)

if __name__ == "__main__":
    main()
