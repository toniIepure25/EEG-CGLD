# tests/test_feature_selection_toy.py
import numpy as np
from eeg_knn_bhho.feature_selection import run_bhho_feature_selection
from typeguard import typechecked

# A minimal “dummy” BinaryHHO for this test (monkey‐patched):
import eeg_knn_bhho.bhho2 as bhho_module

class DummyBinaryHHO:
    def __init__(self, fitness_func, n_features, pop_size, max_iter, transfer, random_state):
        self.fitness_func = fitness_func
        self.n_features = n_features
    def run(self):
        # Always select the first feature (mask = [True, False, False...])
        mask = np.zeros(self.n_features, dtype=bool)
        mask[0] = True
        return mask, self.fitness_func(mask)

def test_bhho_on_toy_dataset(monkeypatch):
    # Create a simple 2D dataset where only feature 0 perfectly separates classes
    rng = np.random.RandomState(0)
    n_samples = 50
    # y = 0 or 1; X[:,0] = y + small noise; X[:,1] = random noise
    y = rng.randint(0, 2, size=n_samples)
    X = np.zeros((n_samples, 2))
    X[:,0] = y + 0.01 * rng.randn(n_samples)
    X[:,1] = rng.randn(n_samples)
    # Monkey‐patch BinaryHHO to our Dummy
    monkeypatch.setattr(bhho_module, "BinaryHHO", DummyBinaryHHO)

    class Cfg:
        class feature_selection:
            pop_size = 10
            max_iter = 5
            transfer = "S"
            k = 3
            cv = 3

    mask, score = run_bhho_feature_selection(X, y, Cfg(), seed=0)
    # The dummy always picks feature 0; check mask[0] == True, mask[1] == False
    assert mask[0] is True and mask[1] is False
    # Score should be near 1.0, since using only feature 0 perfectly separates
    assert score > 0.9
