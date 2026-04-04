import numpy as np
from numpy.typing import NDArray


class Solution:
    def forward(self, X: NDArray[np.float64], gamma: NDArray[np.float64], beta: NDArray[np.float64]) -> NDArray[np.float64]:
        # x: 1D feature vector
        # gamma: 1D scale parameter (same length as x)
        # beta: 1D shift parameter (same length as x)
        eps = 1e-5
        # Normalize: x_hat = (x - mean) / sqrt(var + eps)
        # Scale and shift: out = gamma * x_hat + beta
        # return np.round(your_answer, 5)
        mean = X.mean(axis=-1, keepdims=True)
        var = X.var(axis=-1, keepdims=True)

        X_norm = (X - mean) / np.sqrt(var + eps)

        scale_and_shift = gamma * X_norm + beta

        return np.round(scale_and_shift, 5)
