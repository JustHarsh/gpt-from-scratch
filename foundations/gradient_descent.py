class Solution:
    
    def f_der(self, x):
        return 2*x

    def get_minimizer(self, iterations: int, learning_rate: float, init: int) -> float:
        # Objective function: f(x) = x^2
        # Derivative:         f'(x) = 2x
        # Update rule:        x = x - learning_rate * f'(x)
        # Round final answer to 5 decimal places
        x_old = init

        for it in range(iterations):
            x_new = x_old - (learning_rate * self.f_der(x_old))
            x_old = x_new
        
        return round(x_old, 5)

