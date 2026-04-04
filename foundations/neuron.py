import numpy as np
from numpy.typing import NDArray


class Solution:
    def sigmoid(self, Z, axis=0):
        return 1.0 / (1.0 + np.exp(-Z))

    def relu(self, Z):
        return np.maximum(0.0, Z)

    def forward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, activation: str) -> float:
        # x: 1D input array
        # w: 1D weight array (same length as x)
        # b: scalar bias
        # activation: "sigmoid" or "relu"
        #
        # Pre-activation: z = dot(x, w) + b
        # Sigmoid: σ(z) = 1 / (1 + exp(-z))
        # ReLU: max(0, z)
        # return round(your_answer, 5)
        pre_activation = x @ w + b
        if (activation == "sigmoid"):
            return np.round(self.sigmoid(pre_activation), 5)

        return np.round(self.relu(pre_activation), 5)
