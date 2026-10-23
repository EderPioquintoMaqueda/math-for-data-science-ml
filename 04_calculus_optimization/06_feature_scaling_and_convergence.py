
"""
Lesson 6: Feature scaling and convergence.

Compare gradient descent on raw and standardized features.
"""

import numpy as np

# Two features with very different scales
x1 = np.array([1, 2, 3, 4, 5, 6], dtype=float)
x2 = np.array([1000, 1800, 2500, 3700, 4600, 5200], dtype=float)

# Target values
y = 2 * x1 + 0.003 * x2 + np.array(
    [0.2, -0.1, 0.1, -0.2, 0.15, -0.15]
)

# Standardize each feature
X_raw = np.column_stack((x1, x2))
means = X_raw.mean(axis=0)
stds = X_raw.std(axis=0)
X_scaled = (X_raw - means) / stds

# Add an intercept column
X_raw = np.column_stack((np.ones(len(y)), X_raw))
X_scaled = np.column_stack((np.ones(len(y)), X_scaled))


def cost_function(X, y, beta):
    """Calculate half mean squared error."""
    errors = X @ beta - y
    return np.sum(errors ** 2) / (2 * len(y))


def train_gradient_descent(X, y, learning_rate, iterations):
    """Train a linear model using gradient descent."""
    beta = np.zeros(X.shape[1])
    m = len(y)
    costs = []

    for _ in range(iterations):
        errors = X @ beta - y
        gradient = (X.T @ errors) / m
        beta = beta - learning_rate * gradient
        costs.append(cost_function(X, y, beta))

    return beta, costs


iterations = 2000

# Raw features need a much smaller learning rate
beta_raw, costs_raw = train_gradient_descent(
    X_raw, y, learning_rate=1e-8, iterations=iterations
)

# Standardized features can use a larger learning rate
beta_scaled, costs_scaled = train_gradient_descent(
    X_scaled, y, learning_rate=0.05, iterations=iterations
)

print("Initial cost:", cost_function(X_raw, y, np.zeros(3)))

print("\nRaw features")
for step in [1, 100, 500, 1000, 2000]:
    print(f"Step {step:4d}: cost = {costs_raw[step - 1]:.6f}")

print("\nStandardized features")
for step in [1, 100, 500, 1000, 2000]:
    print(f"Step {step:4d}: cost = {costs_scaled[step - 1]:.6f}")

print("\nFeature means:", means)
print("Feature standard deviations:", stds)
print("Standardized feature means:", X_scaled[:, 1:].mean(axis=0))
print("Standardized feature standard deviations:", X_scaled[:, 1:].std(axis=0))
