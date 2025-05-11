import numpy as np
from sklearn.base import clone
from sklearn.model_selection import cross_val_score


class BinaryHHO:
    def __init__(self, fitness_func, n_features, pop_size=20, max_iter=30, classifier=None, cv=10):
        self.fitness_func = fitness_func  # evaluates binary mask
        self.n_features = n_features
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.classifier = classifier
        self.cv = cv

        # internal state
        self.population = None
        self.fitness_scores = None
        self.best_pos = None
        self.best_score = -np.inf

    def _initialize_population(self):
        self.population = np.random.randint(0, 2, size=(self.pop_size, self.n_features))
        self.fitness_scores = np.zeros(self.pop_size)

    def _transfer_function(self, x):
        return 1 / (1 + np.exp(-x))

    def run(self):
        self._initialize_population()

        # Evaluate initial fitness
        for i in range(self.pop_size):
            self.fitness_scores[i] = self.fitness_func(self.population[i])
        best_idx = np.argmax(self.fitness_scores)
        self.best_pos = self.population[best_idx].copy()
        self.best_score = self.fitness_scores[best_idx]

        for t in range(self.max_iter):
            E0 = 2 * np.random.rand() - 1  # Initial energy

            for i in range(self.pop_size):
                E = 2 * E0 * (1 - t / self.max_iter)  # Update escape energy
                q = np.random.rand()
                r = np.random.rand()

                # Candidate position update (continuous version)
                X = self.population[i].astype(float)
                X_rabbit = self.best_pos.astype(float)

                if abs(E) >= 1:
                    rand_idx = np.random.randint(0, self.pop_size)
                    X_rand = self.population[rand_idx].astype(float)
                    X_new = X_rand - np.random.rand() * abs(X_rand - 2 * np.random.rand() * X)
                else:
                    if r >= 0.5 and abs(E) >= 0.5:
                        X_new = X_rabbit - E * abs(X_rabbit - X)
                    elif r >= 0.5 and abs(E) < 0.5:
                        X_new = X_rabbit - E * abs(X_rabbit - np.mean(self.population, axis=0))
                    else:
                        Y = X_rabbit - E * abs(2 * (1 - np.random.rand()) * X_rabbit - X)
                        Z = Y + np.random.randn(self.n_features) * 0.01  # Simulated dive
                        if self.fitness_func(self._binary(Z)) > self.fitness_func(self._binary(Y)):
                            X_new = Z
                        else:
                            X_new = Y

                # Binary conversion
                binary = self._binary(X_new)
                fitness = self.fitness_func(binary)

                self.population[i] = binary
                self.fitness_scores[i] = fitness

                if fitness > self.best_score:
                    self.best_score = fitness
                    self.best_pos = binary

        return self.best_pos, self.best_score

    def _binary(self, X):
        prob = self._transfer_function(X)
        return (prob > np.random.rand(len(X))).astype(int)
