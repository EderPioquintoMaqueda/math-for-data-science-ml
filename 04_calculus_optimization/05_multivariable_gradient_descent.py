
"""
Lesson 5: Multivariable gradient descent.

Optimize the slope and intercept of a linear regression model
using vectorized NumPy operations.
"""

import numpy as np

# Training data
x = np.array([1, 2, 3, 4, 5], dtype=float)
y = np.array([2, 3, 5, 4, 6], dtype=float)

# Design matrix: first column for slope, second for intercept
X = np.column_stack((x, np.ones(len(x))))

# Initialize parameters: [slope, intercept]
beta = np.zeros(2)

learning_rate = 0.03
iterations = 1000
m = len(y)


def cost_function(X, y, beta):
    """Calculate half mean squared error."""
    errors = X @ beta - y
    return np.sum(errors ** 2) / (2 * len(y))


print("Initial parameters:", beta)
print(f"Initial cost: {cost_function(X, y, beta):.4f}")

for step in range(1, iterations + 1):
    # Predictions and errors
    predictions = X @ beta
    errors = predictions - y

    # Gradient with respect to every parameter
    gradient = (X.T @ errors) / m

    # Update all parameters simultaneously
    beta = beta - learning_rate * gradient

    if step in [1, 10, 100, 500, 1000]:
        current_cost = cost_function(X, y, beta)
        print(
            f"Step {step:4d} | "
            f"Slope: {beta[0]:.4f} | "
            f"Intercept: {beta[1]:.4f} | "
            f"Cost: {current_cost:.4f}"
        )

print("\nFinal model:")
print(f"y = {beta[0]:.4f}x + {beta[1]:.4f}")

# Compare against the least-squares solution
least_squares_beta = np.linalg.lstsq(X, y, rcond=None)[0]

print("\nLeast-squares reference:")
print(f"Slope: {least_squares_beta[0]:.4f}")
print(f"Intercept: {least_squares_beta[1]:.4f}")
