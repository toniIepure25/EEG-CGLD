# shap_analysis.py
import joblib
import numpy as np
import shap
import matplotlib.pyplot as plt

# 1. Load one trained model and its mask
#    Assume you just finished fold 0 on "mat" and saved in:
model_artifact = "results/models/mat/fold_0/knn_model.joblib"
data_artifact = "results/models/mat/fold_0"  # same folder also contains 'mask'

obj = joblib.load(model_artifact)
knn = obj["model"]            # KNeighborsClassifier
mask = obj["mask"]            # boolean mask of selected features

# 2. Load the corresponding X_test and y_test used in that fold
#    For simplicity, store X_test and y_test in joblib at runtime. 
#    If you didn't save them, reload and re-split exactly as in run_experiment.py:
#    Here we assume you have X_bal, y_bal and the same split indices.
#    Otherwise, retrain BHHO and split to recover X_test.
X_test = np.load("results/models/mat/fold_0/X_test.npy")   # shape: (n_samples, n_feats_total)
y_test = np.load("results/models/mat/fold_0/y_test.npy")   # shape: (n_samples,)

# 3. Extract the features the model actually uses
X_test_sel = X_test[:, mask]  # shape: (n_samples, n_sel_feats)

# 4. Create a small background dataset for SHAP (e.g., 100 random training points)
#    Again, load X_train from fold_0 or remap:
X_train = np.load("results/models/mat/fold_0/X_train.npy")
X_train_sel = X_train[:, mask]

#    Take 100 random points
bg_idx = np.random.choice(X_train_sel.shape[0], size=min(100, X_train_sel.shape[0]), replace=False)
background = X_train_sel[bg_idx]

# 5. KernelExplainer (takes ~ O(n_background * n_test * model_prediction_time))
explainer = shap.KernelExplainer(knn.predict_proba, background)

# 6. Compute SHAP values for a subset of test (say first 50 samples)
X_subset = X_test_sel[:50]
shap_values = explainer.shap_values(X_subset, nsamples=200)

# 7. Plot summary (for class 1 probability)
plt.figure(figsize=(8, 6))
shap.summary_plot(shap_values[1], X_subset, show=False)
plt.tight_layout()
plt.savefig("results/figures/shap_summary_mat_fold0.png", dpi=300)
print("Saved SHAP summary plot for class 1 at 'results/figures/shap_summary_mat_fold0.png'")
