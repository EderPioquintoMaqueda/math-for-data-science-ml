
import numpy as np

# Observed data / Datos observados
x = np.array([1., 2., 3., 4., 5.])
y = np.array([2., 3., 5., 4., 6.])

# Design matrix: intercept + feature
# Matriz de diseño: intercepto + variable
X = np.column_stack([np.ones(len(x)), x])

# Solve least-squares problem / Resolver mínimos cuadrados
beta, residual_sum, rank, singular_values = np.linalg.lstsq(
    X, y, rcond=None
)

# Predictions and residuals / Predicciones y residuos
y_pred = X @ beta
residuals = y - y_pred
sse = np.sum(residuals**2)

print("Intercept:", beta[0])
print("Slope:", beta[1])
print("Predictions:", y_pred)
print("Residuals:", residuals)
print("Sum of squared errors:", sse)
print("Rank of design matrix:", rank)
print("Singular values:", singular_values)
