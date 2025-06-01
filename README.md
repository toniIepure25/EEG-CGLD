# EEG KNN + Binary Harris Hawks Optimization (BHHO) Pipeline

> **Note:** This repository contains the codebase for a complete EEG classification pipeline that combines CiSSA‐based feature extraction with KNN classification and Binary Harris Hawks Optimization (BHHO) for feature selection. Experimental results (accuracy, runtime, confusion matrices, MLflow artifacts, etc.) are still being generated—placeholders are marked below where results will be inserted once available.

---

## 📖 Project Overview

This project implements a fully automated EEG processing and classification workflow, including:

1. **Data Loading & Epoching**

   * Support for two open EEG datasets:

     * **MAT Dataset:** EDF‐based rest vs. task recordings
     * **STEW Dataset:** Text‐based high vs. low workload recordings
   * Overlapping epoch extraction (configurable length & overlap)

2. **Preprocessing**

   * **Notch Filter** (50 Hz) → remove power‐line noise
   * **Bandpass Filter** (1–40 Hz) → isolate EEG bands
   * **Epoch Normalization** (z‐score per channel)
   * **ICA‐based Artifact Removal** (optional; kurtosis‐based automatic component rejection)

3. **CiSSA Decomposition**

   * Circulant Singular Spectrum Analysis (CiSSA) extracts narrow‐band Intrinsic Mode Functions (IMFs) per channel
   * Configurable number of IMFs (e.g., 3 per channel)

4. **Feature Extraction**

   * For each IMF:

     * **Band‐Power** (Welch’s method) across defined EEG bands (e.g., δ, θ, α, β, γ)
     * **Downsampling** (configurable factor)
     * **Time‐Domain Statistics:** mean, standard deviation, skewness, kurtosis
     * **Hjorth Parameters:** activity, mobility, complexity
     * **Entropy Measures:** sample entropy, permutation entropy, spectral entropy

5. **Class Balancing**

   * **Undersampling** or **SMOTE** (configurable) to balance classes

6. **Feature Selection via BHHO**

   * **Binary Harris Hawks Optimization** (BHHO) for selecting a subset of features
   * Fitness = cross‐validated accuracy of a KNN classifier on the selected features
   * Configurable parameters: population size, number of iterations, transfer function (S‐shaped or V‐shaped), CV folds, K (neighbors)

7. **Classification & Evaluation**

   * **Final KNN** trained on selected features
   * **Accuracy** computed on test folds (train/test split or Leave‐One‐Subject‐Out (LOSO))
   * **Confusion Matrix** generation and saving as figure
   * **MLflow Logging:** parameters, metrics, model artifacts, confusion matrices

---

## 📂 Repository Structure

```
EEG-CGLD/
├── configs/
│   └── default.yaml             # Hydra configuration (paths, parameters, hyperparameters)
├── data/
│   └── raw/
│       ├── MAT Dataset/         # Place raw MAT EDF files here
│       └── STEW Dataset/        # Place raw STEW .txt files here
├── results/                     # (generated) MLflow runs, models, figures, logs
├── src/
│   └── eeg_knn_bhho/
│       ├── __init__.py
│       ├── data_loading.py      # load_mat_epochs(), load_stew_epochs(), epoch_data_overlap()
│       ├── preprocessing.py     # bandpass_filter(), notch_filter(), normalize_epoch(), ICA (ICACleaner)
│       ├── decomposition.py     # cissa_1d(), CISSADecomposer
│       ├── features.py          # CiSSALightFeatureExtractor
│       ├── feature_selection.py # run_bhho_feature_selection()
│       ├── bhho2.py             # BinaryHHO implementation
│       ├── feature_normalization.py  # FeatureNormalizer (standard scaler)
│       ├── classification.py    # train_final_knn(), plot_and_save_confusion()
│       └── utils.py             # setup_logging(), balance_data(), train_test_split_subjectwise()
├── tests/                       # pytest unit tests for each module
│   ├── test_data_loading.py
│   ├── test_preprocessing.py
│   ├── test_ica_preprocessing.py
│   ├── test_decomposition.py
│   ├── test_feature_extractor.py
│   ├── test_feature_selection_toy.py
│   ├── test_cissa_decomposer.py
│   ├── test_classification_and_utils.py
│   └── test_epoch_data_overlap.py
├── .gitignore
├── README.md
├── requirements.txt             # pip‐installable dependencies
├── setup.py or pyproject.toml   # for editable installation
└── run_experiment.py            # Main entry point (Hydra‐driven)
```

---

## ⚙️ Installation & Setup

1. **Clone this repository**

   ```
   git clone https://github.com/toniIepure25/EEG-CGLD.git
   cd EEG-CGLD
   ```

2. **Set up a Python virtual environment** (recommended)

   ```
   python -m venv eeg_env
   source eeg_env/bin/activate    # Linux/MacOS
   eeg_env\Scripts\activate.bat   # Windows
   ```

3. **Install dependencies**

   ```
   pip install --upgrade pip setuptools
   pip install -r requirements.txt
   # Or, if you support editable install:
   pip install -e .
   ```

4. **Organize raw data**

   * Place your MAT EDF files under `data/raw/MAT Dataset/`
   * Place your STEW `.txt` files under `data/raw/STEW Dataset/`
   * Ensure the folder names match exactly (spaces included)

---

## 🚀 Running the Pipeline Locally

By default, Hydra looks for `configs/default.yaml`. That file already points to:

```
data.mat_dir: ${hydra:runtime.cwd}/data/raw/MAT Dataset
data.stew_dir: ${hydra:runtime.cwd}/data/raw/STEW Dataset
```

So if your folder structure matches, simply:

```
python run_experiment.py
```

### Command‐Line Overrides

You can override any parameter in `default.yaml` by appending `<key>=<value>` flags. For example:

* **Change preprocessing parameters:**

  ```
  python run_experiment.py preprocess.band_low=0.5 preprocess.band_high=30.0
  ```
* **Use LOSO (Leave‐One‐Subject‐Out) instead of train/test:**

  ```
  python run_experiment.py evaluation.strategy=LOSO
  ```
* **Increase BHHO iterations to 100:**

  ```
  python run_experiment.py feature_selection.max_iter=100
  ```
* **Specify subject IDs for LOSO (if available):**

  ```
  python run_experiment.py data.subject_ids="[1,2,3,4,5]"
  ```

---

## 🚀 Running in Google Colab

1. **Mount Google Drive** (with raw EEG folders uploaded there)

   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

2. **Clone this repo and install dependencies**

   ```python
   !git clone https://github.com/toniIepure25/EEG-CGLD.git
   %cd EEG-CGLD
   !pip install -r requirements.txt
   ```

3. **Set Drive paths for `MAT Dataset` and `STEW Dataset`**

   > Adjust `MY_DRIVE_PATH` to where you uploaded the folders in your Drive.

   ```python
   MAT_PATH = "/content/drive/MyDrive/EEG-CGLD-Data/MAT Dataset"
   STEW_PATH = "/content/drive/MyDrive/EEG-CGLD-Data/STEW Dataset"
   ```

4. **Run the experiment with overrides**

   ```python
   !python run_experiment.py \
     data.mat_dir="$MAT_PATH" \
     data.stew_dir="$STEW_PATH"
   ```

No changes to `default.yaml` are needed—just override those two keys at runtime.

---

## 🔧 Configuration (`configs/default.yaml`)

The main config file (`default.yaml`) sets all hyperparameters and file paths. Important sections:

```
data:
  mat_dir: "${hydra:runtime.cwd}/data/raw/MAT Dataset"
  stew_dir: "${hydra:runtime.cwd}/data/raw/STEW Dataset"
  subject_ids: null

preprocess:
  sfreq_mat: 128
  sfreq_stew: 256
  epoch_length: 1.0
  overlap: 0.5
  band_low: 1.0
  band_high: 40.0
  notch_freq: 50.0
  notch_Q: 30.0
  filter_order: 5
  ica_n_components: null         # None = use all channels
  ica_kurtosis_thresh: 5.0

ssa:
  n_imfs: 3
  n_jobs: 6
  ds: 2                          # Downsampling factor

feature_extraction:
  bands:
    - [1, 4]
    - [4, 8]
    - [8, 13]
    - [13, 30]
    - [30, 40]
  nperseg: 64
  n_jobs: 6

sampling:
  method: undersample           # "undersample" or "smote"
  random_state: 42

feature_selection:
  pop_size: 50
  max_iter: 60
  transfer: "S"
  k: 7                          # K for inner KNN
  cv: 10

classifier:
  k: 7                          # Final KNN neighbors

evaluation:
  strategy: train_test          # "train_test" or "LOSO"
  train_ratio: 0.8
  seed: 42

logging:
  mlflow:
    tracking_uri: "file:${hydra:runtime.cwd}/results/mlruns"
  experiment_name: "eeg_knn_bhho_experiments"
  output_dir: "${hydra:runtime.cwd}/results"
```

Feel free to tweak any of these via CLI overrides.

---

## 🧪 Unit Tests

We include `pytest`-based unit tests for each module under `tests/`. To run them locally:

```
pytest --maxfail=1 --disable-warnings -q
```

Expected status: **All tests pass** (except any currently computing new functionality).

---

## 📊 Placeholder Results

Once you run the full pipeline, the results will be logged under `results/mlruns/`. You can display summary metrics here. Below are placeholders:

* **MAT Dataset (train\_test)**

  * Number of epochs: 17 280
  * Number of channels: 21
  * Average test accuracy: **`<MAT_AVG_ACC_PLACEHOLDER>%`**
  * Selected features (per fold average): **`<MAT_MEAN_FEATURES_SELECTED>`**

* **STEW Dataset (train\_test)**

  * Number of epochs: 14 304
  * Number of channels: 14
  * Average test accuracy: **`<STEW_AVG_ACC_PLACEHOLDER>%`**
  * Selected features (per fold average): **`<STEW_MEAN_FEATURES_SELECTED>`**

* **Runtime (CPU‐only, Ryzen 5 5600H)**

  * Total pipeline time (MAT+STEW, train\_test): **`<RUNTIME_LOCAL_MM_SS>`**
  * Total pipeline time (MAT+STEW, LOSO‐5): **`<RUNTIME_LOCAL_LOSO_MM_SS>`**

* **Runtime (Colab CPU)**

  * Total pipeline time (MAT+STEW, train\_test): **`<RUNTIME_COLAB_CPU_MM_SS>`**
  * Total pipeline time (MAT+STEW, LOSO‐5): **`<RUNTIME_COLAB_CPU_LOSO_MM_SS>`**

* **Runtime (Colab GPU)** *(if GPU‐porting is implemented)*

  * Total pipeline time (MAT+STEW, train\_test): **`<RUNTIME_COLAB_GPU_MM_SS>`**

* **Confusion Matrices**

  * Example MAT confusion matrix: `results/figures/mat/fold_0/confusion_matrix.png`
  * Example STEW confusion matrix: `results/figures/stew/fold_0/confusion_matrix.png`

> Replace all `<...>` placeholders with actual numbers/images once experiments complete.

---

## 🔬 Results Folder Structure

After running `run_experiment.py`, you’ll see:

```
results/
├── mlruns/                   # MLflow experiment logs
│   └── <run_id>/             # Experiment runs (parameters, metrics, artifacts)
├── models/
│   ├── mat/
│   │   ├── fold_0/knn_model.joblib
│   │   ├── fold_1/knn_model.joblib
│   │   └── ...
│   └── stew/
│       ├── fold_0/knn_model.joblib
│       └── ...
└── figures/
    ├── mat/
    │   ├── fold_0/confusion_matrix.png
    │   ├── fold_1/confusion_matrix.png
    │   └── ...
    └── stew/
        ├── fold_0/confusion_matrix.png
        └── ...
```

---

## 📈 Expected Outcomes & Future Work

* **Performance Goal:** ≥ 85 % average test accuracy on both MAT and STEW in a train\_test split.
* **Next Steps:**

  1. Integrate GPU‐accelerated FFT and PSD computations (CuPy / PyTorch) → reduce preprocessing time
  2. Port KNN evaluation to FAISS on GPU for faster BHHO fitness calls
  3. Evaluate LOSO strategy across all subjects and report per‐subject accuracy
  4. Compare BHHO feature selection with alternative methods (e.g., ReliefF, Recursive Feature Elimination)
  5. Hyperparameter sweep using Hydra multirun (e.g., vary number of IMFs, BHHO pop\_size/max\_iter)

---

## 👥 Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork this repository.
2. Create a new branch for your feature/bugfix:

   ```
   git checkout -b feature/awesome-improvement
   ```
3. Make your changes, add tests if needed, and ensure all tests pass.
4. Push to your fork and submit a Pull Request.

---

## 📝 License

This project is released under the [MIT License](LICENSE).

---

## 📬 Contact

For questions or issues, please open an issue on GitHub or contact:

* **Toni Iepure**
* GitHub: [toniIepure25](https://github.com/toniIepure25)
* Email: `<YOUR_EMAIL_PLACEHOLDER>`

Thank you for using the EEG KNN + BHHO pipeline!
