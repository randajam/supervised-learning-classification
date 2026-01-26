import numpy as np

class CustomKNN:
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict_proba(self, X):
        proba = []
        for x in X:
            distances = np.linalg.norm(self.X_train - x, axis=1)
            idx = np.argsort(distances)[:self.n_neighbors]
            votes = self.y_train[idx]
            proba.append(np.mean(votes))
        return np.array(proba)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)
