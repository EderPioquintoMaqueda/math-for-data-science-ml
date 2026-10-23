
import numpy as np

# Linear regression gradient derivation
# Derivación del gradiente de regresión lineal

# Training data
x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
y = np.array([2.0, 3.0, 5.0, 4.0, 6.0])

m = len(y)

# Design matrix: intercept + feature
X = np.column_stack([np.ones(m), x])

# Initial model parameters: intercept and slope
beta = np.array([0.0, 0.0])

# Calculate predictions and errors
predictions = X @ beta
errors = predictions - y

# Loss: half the mean squared error
loss = np.sum(errors ** 2) / (2 * m)

# Analytical gradient: (X.T @ errors) / m
gradient = (X.T @ errors) / m

print("Initial parameters")
print(f"Intercept: {beta[0]:.4f}")
print(f"Slope: {beta[1]:.4f}")

print("\nPredictions:", predictions)
print("Errors:", errors)
print(f"Loss: {loss:.4f}")

print("\nAnalytical gradient")
print(f"dJ/dbeta_0: {gradient[0]:.4f}")
print(f"dJ/dbeta_1: {gradient[1]:.4f}")


# Numerical gradient using finite differences
# Gradiente numérico mediante diferencias finitas

def calculate_loss(parameters):
    residuals = X @ parameters - y
    return np.sum(residuals ** 2) / (2 * m)


epsilon = 1e-6
numerical_gradient = np.zeros_like(beta)

for j in range(len(beta)):
    beta_plus = beta.copy()
    beta_minus = beta.copy()

    beta_plus[j] += epsilon
    beta_minus[j] -= epsilon

    numerical_gradient[j] = (
        calculate_loss(beta_plus) - calculate_loss(beta_minus)
    ) / (2 * epsilon)

print("\nNumerical gradient")
print(f"dJ/dbeta_0: {numerical_gradient[0]:.4f}")
print(f"dJ/dbeta_1: {numerical_gradient[1]:.4f}")

print("\nGradient check")
print("Analytical:", gradient)
print("Numerical: ", numerical_gradient)
print("Match:", np.allclose(
    gradient, numerical_gradient, atol=1e-5
))
