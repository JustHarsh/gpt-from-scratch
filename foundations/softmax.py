import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)
        Z_stable = z - np.max(z, axis=0)
        soft = np.exp(Z_stable) / np.sum(np.exp(Z_stable))
        return np.round(soft, 4)
