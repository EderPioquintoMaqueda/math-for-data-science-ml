
import numpy as np

# Design matrix with a duplicated feature
# Matriz de diseño con una variable duplicada
x = np.array([1., 2., 3., 4., 5.])

X = np.column_stack([
    np.ones_like(x),
    x,
    x
])

y = np.array([2., 3., 5., 4., 6.])

# Singular value decomposition / Descomposición SVD
U, singular_values, Vt = np.linalg.svd(
    X, full_matrices=False
)

# Rank and condition information / Rango e información de condición
rank = np.linalg.matrix_rank(X)
condition_number = np.linalg.cond(X)

print("Matrix shape:", X.shape)
print("Singular values:", singular_values)
print("Rank:", rank)
print("Condition number:", condition_number)

# Moore-Penrose pseudoinverse
# Pseudoinversa de Moore-Penrose
X_pinv = np.linalg.pinv(X)
beta_pinv = X_pinv @ y

print("Pseudoinverse coefficients:", beta_pinv)
print("Predictions:", X @ beta_pinv)
print("Residual sum of squares:",
      np.sum((y - X @ beta_pinv) ** 2))
