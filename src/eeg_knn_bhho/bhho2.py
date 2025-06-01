# src/eeg_knn_bhho/bhho2.py

"""
Binary Harris Hawks Optimization (B-HHO) for feature selection.

This implementation follows the standard B-HHO algorithm (Heidari et al., 2019),
adapted to binary search spaces via transfer functions (Mirjalili & Lewis, 2013).

Given a fitness function that accepts a binary mask and returns a scalar score
(maximization), this optimizer evolves a population of binary feature-selection masks.

References
----------
- Heidari, A. A., Mirjalili, S., Faris, H., Aljarah, I., Mafarja, M., & Chen, H.
  (2019). Harris Hawks Optimization: Algorithm and Applications.
  Future Generation Computer Systems, 97, 849–872.
- Mirjalili, S., & Lewis, A. (2013). S-shaped versus V-shaped transfer
  functions for binary particle swarm optimization. Swarm and Evolutionary
  Computation, 9, 1–14.
"""

import math
from typing import Callable, Optional, Tuple

import numpy as np


class BinaryHHO:
    """
    Binary Harris Hawks Optimization (B-HHO) for feature selection.

    Parameters
    ----------
    fitness_func : Callable[[np.ndarray], float]
        Function that takes a binary mask (1D array of 0/1) and returns a scalar fitness.
        Higher fitness is better (maximization problem).
    n_features : int
        Total number of features (length of the binary mask).
    pop_size : int, default=20
        Number of Harris hawks (population size).
    max_iter : int, default=30
        Maximum number of iterations to run.
    transfer : str, default='S'
        Transfer function type: 'S' (sigmoid) or 'V' (V-shaped).
    random_state : Optional[int], default=None
        Seed for reproducibility.

    Attributes
    ----------
    best_mask_ : np.ndarray, shape (n_features,)
        Best binary mask found.
    best_score_ : float
        Fitness of the best_mask_.
    history_ : list of float
        Best fitness value at each iteration.
    """

    def __init__(
        self,
        fitness_func: Callable[[np.ndarray], float],
        n_features: int,
        pop_size: int = 20,
        max_iter: int = 30,
        transfer: str = 'S',
        random_state: Optional[int] = None
    ):
        self.fitness_func = fitness_func
        self.n_features = n_features
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.transfer = transfer.upper()
        if self.transfer not in ('S', 'V'):
            raise ValueError("`transfer` must be 'S' or 'V'")

        self.random_state = random_state
        self.rng = np.random.RandomState(self.random_state)

        # To be set during optimization
        self.best_mask_: Optional[np.ndarray] = None
        self.best_score_: Optional[float] = None
        self.history_: list[float] = []

    def _levy_flight(self, dim: int, beta: float = 1.5) -> np.ndarray:
        """
        Generate a Levy flight step vector of length `dim` using Mantegna’s algorithm.

        Parameters
        ----------
        dim : int
            Number of dimensions (features).
        beta : float, default=1.5
            Levy distribution parameter (1 < beta <= 2).

        Returns
        -------
        step : np.ndarray, shape (dim,)
            Levy flight perturbation vector.
        """
        # Mantegna’s algorithm constants
        numerator = math.gamma(1 + beta) * math.sin(math.pi * beta / 2)
        denominator = math.gamma((1 + beta) / 2) * beta * (2 ** ((beta - 1) / 2))
        sigma_u = (numerator / denominator) ** (1 / beta)
        sigma_v = 1.0

        u = self.rng.randn(dim) * sigma_u
        v = self.rng.randn(dim) * sigma_v
        step = u / (np.abs(v) ** (1 / beta))
        return step

    def _transfer_S(self, x: np.ndarray) -> np.ndarray:
        """
        S-shaped transfer function (sigmoid-based).

        T(x) = 1 / (1 + exp(-10 * (x - 0.5)))

        Parameters
        ----------
        x : np.ndarray
            Continuous values in [0, 1].

        Returns
        -------
        T : np.ndarray
            Transfer probabilities in (0, 1).
        """
        return 1.0 / (1.0 + np.exp(-10.0 * (x - 0.5)))

    def _transfer_V(self, x: np.ndarray) -> np.ndarray:
        """
        V-shaped transfer function.

        T(x) = |erf((sqrt(pi) * (x - 0.5)) / 2)|

        Parameters
        ----------
        x : np.ndarray
            Continuous values in [0, 1].

        Returns
        -------
        T : np.ndarray
            Transfer probabilities in (0, 1).
        """
        return np.abs(np.erf((np.sqrt(math.pi) * (x - 0.5)) / 2))

    def _initialize_population(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Initialize continuous and binary populations.

        Returns
        -------
        X_cont : np.ndarray, shape (pop_size, n_features)
            Continuous position values in [0, 1].
        X_bin : np.ndarray, shape (pop_size, n_features)
            Initial binary masks (0/1) derived from X_cont.
        """
        X_cont = self.rng.rand(self.pop_size, self.n_features)
        X_bin = (X_cont > 0.5).astype(int)
        return X_cont, X_bin

    def _binarize(self, X_cont: np.ndarray) -> np.ndarray:
        """
        Convert continuous positions to binary masks via the chosen transfer function.

        Parameters
        ----------
        X_cont : np.ndarray, shape (..., n_features)
            Continuous values in [0, 1].

        Returns
        -------
        X_bin : np.ndarray of int (0 or 1), same shape as X_cont
        """
        if self.transfer == 'S':
            T = self._transfer_S(X_cont)
        else:
            T = self._transfer_V(X_cont)

        rand_matrix = self.rng.rand(*X_cont.shape)
        return (rand_matrix < T).astype(int)

    def run(self) -> Tuple[np.ndarray, float]:
        """
        Execute the B-HHO algorithm.

        Returns
        -------
        best_mask : np.ndarray, shape (n_features,)
            Best binary mask found.
        best_score : float
            Fitness of the best mask.
        """
        # 1. Initialize population
        X_cont, X_bin = self._initialize_population()
        fitness = np.zeros(self.pop_size)

        # 2. Evaluate initial fitness
        for i in range(self.pop_size):
            fitness[i] = self.fitness_func(X_bin[i])

        # 3. Identify the initial best (prey)
        best_idx = np.argmax(fitness)
        self.best_mask_ = X_bin[best_idx].copy()
        self.best_score_ = fitness[best_idx]
        self.history_ = [self.best_score_]

        # 4. Iterative optimization
        for t in range(1, self.max_iter + 1):
            E0 = 2 * self.rng.rand() - 1  # in [-1, 1]
            E = 2 * E0 * (1 - t / self.max_iter)  # escaping energy

            for i in range(self.pop_size):
                Xi_cont = X_cont[i].copy()
                Xi_bin = X_bin[i].copy()

                if abs(E) >= 1:
                    # --------------------------
                    # Exploration phase
                    # --------------------------
                    q = self.rng.rand()
                    rand_idx = self.rng.randint(self.pop_size)
                    X_rand_cont = X_cont[rand_idx]

                    if q < 0.5:
                        # (i) Random tall jumps between hawks
                        r1 = self.rng.rand(self.n_features)
                        r2 = self.rng.rand(self.n_features)
                        X_new_cont = X_rand_cont - r1 * np.abs(
                            X_rand_cont - 2 * r2 * Xi_cont
                        )
                    else:
                        # (ii) Perch randomly based on average position (leapfrog)
                        avg_cont = np.mean(X_cont, axis=0)
                        r3 = self.rng.rand(self.n_features)
                        X_new_cont = (self.best_mask_.astype(float) - avg_cont) - \
                                     r3 * (self.best_mask_.astype(float) - avg_cont)
                else:
                    # --------------------------
                    # Exploitation phase (besiege)
                    # --------------------------
                    r2 = self.rng.rand()
                    if r2 >= 0.5 and abs(E) >= 0.5:
                        # Soft besiege
                        X_new_cont = self.best_mask_.astype(float) - E * \
                                     np.abs(self.best_mask_.astype(float) - Xi_cont)
                    elif r2 >= 0.5 and abs(E) < 0.5:
                        # Hard besiege
                        X_new_cont = self.best_mask_.astype(float) - E * \
                                     np.abs(self.best_mask_.astype(float) - Xi_cont)
                    elif r2 < 0.5 and abs(E) >= 0.5:
                        # Soft besiege with progressive rapid dive
                        LF = self._levy_flight(self.n_features)
                        X_temp = self.best_mask_.astype(float) - E * \
                                 np.abs(self.best_mask_.astype(float) - Xi_cont)
                        r4 = self.rng.rand(self.n_features)
                        X_new_cont = X_temp + r4 * LF
                    else:
                        # Hard besiege with progressive rapid dive
                        LF = self._levy_flight(self.n_features)
                        avg_cont = np.mean(X_cont, axis=0)
                        X_temp = self.best_mask_.astype(float) - E * \
                                 np.abs(self.best_mask_.astype(float) - avg_cont)
                        r5 = self.rng.rand(self.n_features)
                        X_new_cont = X_temp + r5 * LF

                # 5. Clip continuous values to [0, 1]
                X_new_cont = np.clip(X_new_cont, 0.0, 1.0)

                # 6. Binarize
                X_new_bin = self._binarize(X_new_cont)

                # 7. Evaluate new fitness
                f_new = self.fitness_func(X_new_bin)

                # 8. Greedy selection: if new solution is better, replace
                if f_new > fitness[i]:
                    X_cont[i] = X_new_cont
                    X_bin[i] = X_new_bin
                    fitness[i] = f_new

                    # Update global best if improved
                    if f_new > self.best_score_:
                        self.best_score_ = f_new
                        self.best_mask_ = X_new_bin.copy()

            # Record best fitness for this iteration
            self.history_.append(self.best_score_)

        return self.best_mask_, self.best_score_
