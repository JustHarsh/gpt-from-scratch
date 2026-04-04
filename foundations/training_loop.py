import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def train(self, X: NDArray[np.float64], y: NDArray[np.float64], epochs: int, lr: float) -> Tuple[NDArray[np.float64], float]:
        # X: (n_samples, n_features)
        # y: (n_samples,) targets
        # epochs: number of training iterations
        # lr: learning rate
        #
        # Model: y_hat = X @ w + b
        # Loss: MSE = (1/n) * sum((y_hat - y)^2)
        # Initialize w = zeros, b = 0
        # return (np.round(w, 5), round(b, 5))
        m, n = X.shape
        theta = np.zeros(n, dtype=np.float64)
        b = 0.0

        for epoch in range(epochs):
            y_pred = X @ theta + b

            error = y_pred - y

            dw = (2.0 / m) * (X.T @ error)
            db = (2.0 / m) * np.sum(error)

            theta -= lr * dw
            b -= lr * db

        return (np.round(theta, 5), round(float(b), 5))
