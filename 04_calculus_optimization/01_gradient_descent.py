
import numpy as np

# Training data
x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
y = np.array([2.0, 3.0, 5.0, 4.0, 6.0])

# Standardize the input feature
x_mean = x.mean()
x_std = x.std()
z = (x - x_mean) / x_std

# Design matrix: intercept + feature
X = np.column_stack([np.ones_like(z), z])

# Initialize model parameters
theta = np.zeros(2)

# Gradient descent settings
learning_rate = 0.1
n_iterations = 1000
m = len(y)

# Gradient descent
for step in range(n_iterations):
    predictions = X @ theta
    errors = predictions - y

    # Cost function: half the mean squared error
    cost = np.sum(errors ** 2) / (2 * m)

    # Compute the gradient
    gradient = (X.T @ errors) / m

    # Update parameters
    theta -= learning_rate * gradient

    if step in [0, 1, 9, 99, 999]:
        print(f"Step {step + 1}: cost = {cost:.6f}")

# Final predictions and evaluation
predictions = X @ theta
mse = np.mean((y - predictions) ** 2)

# Convert coefficients to the original x scale
slope = theta[1] / x_std
intercept = theta[0] - slope * x_mean

# Compare against the least-squares solution
X_original = np.column_stack([np.ones_like(x), x])
beta_lstsq = np.linalg.lstsq(
    X_original, y, rcond=None
)[0]

print("\nGradient descent results")
print(f"Intercept: {intercept:.4f}")
print(f"Slope: {slope:.4f}")
print(f"MSE: {mse:.4f}")

print("\nLeast-squares reference")
print(f"Intercept: {beta_lstsq[0]:.4f}")
print(f"Slope: {beta_lstsq[1]:.4f}")
