# import numpy as np
# import math
# import logging

# logger = logging.getLogger(__name__)   # __name__ → 'src.bhho2'

# def levy_flight(beta: float, size: tuple) -> np.ndarray:
#     """
#     Generate Lévy‐flight steps using Mantegna's algorithm.
    
#     Parameters
#     ----------
#     beta : float
#         Stability parameter (1 < beta ≤ 2), often 1.5.
#     size : tuple
#         Output shape, e.g. (n_features,)
    
#     Returns
#     -------
#     step : np.ndarray
#         Lévy‐flight step array of shape `size`.
#     """
#     # Mantegna’s sigma for u
#     num = math.gamma(1 + beta) * math.sin(math.pi * beta / 2)
#     den = math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2)
#     sigma_u = (num / den) ** (1 / beta)
#     u = np.random.randn(*size) * sigma_u
#     v = np.random.randn(*size)
#     return u / (np.abs(v) ** (1 / beta))


# class BinaryHHO:
#     def __init__(
#         self,
#         fitness_func,
#         n_features: int,
#         pop_size: int = 20,
#         max_iter: int = 30,
#         levy_beta: float = 1.5
#     ):
#         """
#         Bit‐for‐bit BHHO as in the paper.
        
#         Parameters
#         ----------
#         fitness_func : Callable[[np.ndarray], float]
#             Takes a binary mask (1D array) → fitness score.
#         n_features : int
#             Dimensionality of the binary mask.
#         pop_size : int
#             Number of hawks.
#         max_iter : int
#             Number of iterations.
#         levy_beta : float
#             β for Lévy flights (default 1.5).
#         """
#         self.fitness_func = fitness_func
#         self.D = n_features
#         self.N = pop_size
#         self.T = max_iter
#         self.beta = levy_beta
        
#         # Internal state
#         self.population = None     # shape (N, D)
#         self.fitness = None        # shape (N,)
#         self.best_pos = None       # shape (D,)
#         self.best_score = -np.inf

#     def _initialize(self):
#         self.population = np.random.randint(0, 2, size=(self.N, self.D))
#         self.fitness = np.zeros(self.N)
#         # Evaluate
#         for i in range(self.N):
#             self.fitness[i] = self.fitness_func(self.population[i])
#         idx = np.argmax(self.fitness)
#         self.best_pos = self.population[idx].copy()
#         self.best_score = self.fitness[idx]
#         logger.info(f"Initial best score = {self.best_score:.4f} (hawk {idx})")

#     def _transfer(self, x: np.ndarray) -> np.ndarray:
#         """
#         Numerically stable sigmoid: 
#         for x>=0: 1/(1+e^{-2x}) 
#         for x< 0: e^{2x}/(1+e^{2x})
#         """
#         x = x.astype(np.float64)
#         out = np.empty_like(x)
#         pos = x >= 0
#         # positive branch avoids huge positive exponent
#         out[pos] = 1.0 / (1.0 + np.exp(-2.0 * x[pos]))
#         # negative branch avoids huge negative exponent
#         exp2x = np.exp(2.0 * x[~pos])
#         out[~pos] = exp2x / (1.0 + exp2x)
#         return out


#     def run(self):
#         # 1) Initialization
#         self._initialize()

#         # Pre-allocate mean placeholder
#         for t in range(self.T):
#             logger.info(f"--- Iteration {t}/{self.T} ---")

#             E1 = 2 * (1 - t / self.T)          # Energy decay factor
#             X_mean = self.population.mean(axis=0)

#             for i in range(self.N):
#                 E0 = np.random.uniform(-1, 1)  # initial random energy
#                 E  = E1 * E0                  # escape energy
#                 r  = np.random.rand()

#                 X_i     = self.population[i].astype(float)
#                 X_best  = self.best_pos.astype(float)

#                 # --- EXPLORATION PHASE ---
#                 if abs(E) >= 1:
#                     # random hawk
#                     rand_idx = np.random.randint(0, self.N)
#                     X_rand   = self.population[rand_idx].astype(float)
#                     X_new_c  = X_rand - np.random.rand() * np.abs(
#                         X_rand - 2 * np.random.rand() * X_i
#                     )

#                 # --- EXPLOITATION PHASE ---
#                 else:
#                     # 1) Soft besiege: r ≥ 0.5 & |E| ≥ 0.5
#                     if r >= 0.5 and abs(E) >= 0.5:
#                         X_new_c = X_best - E * np.abs(X_best - X_i)

#                     # 2) Hard besiege: r ≥ 0.5 & |E| < 0.5
#                     elif r >= 0.5 and abs(E) < 0.5:
#                         X_new_c = X_best - E * np.abs(X_best - X_mean)

#                     # 3) Soft besiege + Lévy: r < 0.5 & |E| ≥ 0.5
#                     elif r < 0.5 and abs(E) >= 0.5:
#                         X1     = X_best - E * np.abs(X_best - X_i)
#                         LF     = levy_flight(self.beta, (self.D,))
#                         X_new_c = X1 + LF

#                     # 4) Hard besiege + Lévy: r < 0.5 & |E| < 0.5
#                     else:
#                         X1      = X_best - E * np.abs(X_best - X_mean)
#                         LF      = levy_flight(self.beta, (self.D,))
#                         X_new_c = X1 + LF

#                 # Binary conversion via S-transfer
#                 probs  = self._transfer(X_new_c)
#                 X_new  = (probs > np.random.rand(self.D)).astype(int)

#                 # Evaluate fitness
#                 f_new = self.fitness_func(X_new)
#                 self.population[i] = X_new
#                 self.fitness[i]   = f_new

#                 # Update global best
#                 if f_new > self.best_score:
#                     self.best_score = f_new
#                     self.best_pos   = X_new.copy()
#                     # logger.info(f"New best! Iter {t}, hawk[{i}] → score = {f_new:.4f}")


#         return self.best_pos, self.best_score
import numpy as np
import math
import logging
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

def levy_flight(beta: float, size: tuple) -> np.ndarray:
    """
    Generate Lévy‐flight steps using Mantegna's algorithm.
    """
    num = math.gamma(1 + beta) * math.sin(math.pi * beta / 2)
    den = math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2)
    sigma_u = (num / den) ** (1 / beta)
    u = np.random.randn(*size) * sigma_u
    v = np.random.randn(*size)
    return u / (np.abs(v) ** (1 / beta))

class BinaryHHO:
    """
    Binary Harris Hawks Optimization with choice of S- or V-shaped transfer.
    """
    def __init__(
        self,
        fitness_func,
        n_features: int,
        pop_size: int = 20,
        max_iter: int = 30,
        levy_beta: float = 1.5,
        transfer: str = 'S'
    ):
        self.fitness_func = fitness_func
        self.D = n_features
        self.N = pop_size
        self.T = max_iter
        self.beta = levy_beta
        self.transfer = transfer.upper()

        self.population = None
        self.fitness = None
        self.best_pos = None
        self.best_score = -np.inf

    def _initialize(self):
        self.population = np.random.randint(0, 2, size=(self.N, self.D))
        self.fitness = np.zeros(self.N)
        for i in range(self.N):
            self.fitness[i] = self.fitness_func(self.population[i])
        idx = np.argmax(self.fitness)
        self.best_pos = self.population[idx].copy()
        self.best_score = self.fitness[idx]
        logger.info(f"Initial best score = {self.best_score:.4f} (hawk {idx})")

    def _transfer_S(self, x: np.ndarray) -> np.ndarray:
        out = np.empty_like(x, dtype=np.float64)
        pos = x >= 0
        out[pos] = 1.0 / (1.0 + np.exp(-2.0 * x[pos]))
        exp2x = np.exp(2.0 * x[~pos])
        out[~pos] = exp2x / (1.0 + exp2x)
        return out

    def _transfer_V(self, x: np.ndarray) -> np.ndarray:
        # V1 transfer: |x| / sqrt(1 + x^2)
        return np.abs(x / np.sqrt(1.0 + x**2))

    def _transfer(self, x: np.ndarray) -> np.ndarray:
        if self.transfer == 'S':
            return self._transfer_S(x)
        elif self.transfer == 'V':
            return self._transfer_V(x)
        else:
            raise ValueError(f"Unknown transfer type: {self.transfer}")

    def run(self):
        self._initialize()
        for t in range(self.T):
            logger.info(f"--- Iteration {t+1}/{self.T} ---")
            E1 = 2 * (1 - t / self.T)
            X_mean = self.population.mean(axis=0).astype(float)

            for i in range(self.N):
                E0 = np.random.uniform(-1, 1)
                E = E1 * E0
                r = np.random.rand()
                X_i = self.population[i].astype(float)
                X_best = self.best_pos.astype(float)

                if abs(E) >= 1:
                    rand_idx = np.random.randint(0, self.N)
                    X_rand = self.population[rand_idx].astype(float)
                    X_new_c = X_rand - np.random.rand() * np.abs(X_rand - 2 * np.random.rand() * X_i)
                else:
                    if r >= 0.5 and abs(E) >= 0.5:
                        X_new_c = X_best - E * np.abs(X_best - X_i)
                    elif r >= 0.5 and abs(E) < 0.5:
                        X_new_c = X_best - E * np.abs(X_best - X_mean)
                    elif r < 0.5 and abs(E) >= 0.5:
                        X1 = X_best - E * np.abs(X_best - X_i)
                        LF = levy_flight(self.beta, (self.D,))
                        X_new_c = X1 + LF
                    else:
                        X1 = X_best - E * np.abs(X_best - X_mean)
                        LF = levy_flight(self.beta, (self.D,))
                        X_new_c = X1 + LF

                probs = self._transfer(X_new_c)
                X_new = (probs > np.random.rand(self.D)).astype(int)

                f_new = self.fitness_func(X_new)
                self.population[i] = X_new
                self.fitness[i] = f_new

                if f_new > self.best_score:
                    self.best_score = f_new
                    self.best_pos = X_new.copy()
                    logger.info(f"New best! Iter {t+1}, hawk[{i}] -> score = {f_new:.4f}")

        return self.best_pos, self.best_score


def run_bhho_feature_selection(
    X: np.ndarray,
    y: np.ndarray,
    pop_size: int = 20,
    max_iter: int = 30,
    transfer: str = 'S',
    k: int = 5,
    cv: int = 10
) -> (np.ndarray, float):
    """
    Wrap BHHO + KNN for feature selection with configurable transfer function.
    """
    def fitness(mask: np.ndarray) -> float:
        if np.count_nonzero(mask) == 0:
            return 0.0
        X_sel = X[:, mask == 1]
        return cross_val_score(
            KNeighborsClassifier(n_neighbors=k),
            X_sel,
            y,
            cv=cv,
            n_jobs=-1
        ).mean()

    bhho = BinaryHHO(
        fitness_func=fitness,
        n_features=X.shape[1],
        pop_size=pop_size,
        max_iter=max_iter,
        transfer=transfer
    )
    best_mask, best_score = bhho.run()
    return best_mask, best_score
