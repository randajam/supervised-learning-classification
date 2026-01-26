import numpy as np

class CustomGaussianNB:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.mean = {}
        self.var = {}
        self.prior = {}

        for c in self.classes:
            X_c = X[y == c]
            self.mean[c] = X_c.mean(axis=0)
            self.var[c] = X_c.var(axis=0) + 1e-9
            self.prior[c] = X_c.shape[0] / X.shape[0]

    def _log_gaussian_density(self, x, mean, var):
        return -0.5 * (np.log(2 * np.pi * var) + ((x - mean) ** 2) / var)

    def predict_proba(self, X):
        log_probs = []

        for x in X:
            class_log_probs = []
            for c in self.classes:
                log_prior = np.log(self.prior[c])
                log_likelihood = self._log_gaussian_density(
                    x, self.mean[c], self.var[c]
                ).sum()
                class_log_probs.append(log_prior + log_likelihood)

            log_probs.append(class_log_probs)

        log_probs = np.array(log_probs)

        # стабилизация: вычитаем максимум
        log_probs -= log_probs.max(axis=1, keepdims=True)
        probs = np.exp(log_probs)
        probs /= probs.sum(axis=1, keepdims=True)
        return probs
