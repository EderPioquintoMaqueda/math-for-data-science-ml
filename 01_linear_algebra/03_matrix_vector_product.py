
import numpy as np

# Rows: companies; columns: indicators
# Filas: empresas; columnas: indicadores
X = np.array([
    [2.0, 1.0, 2.0],
    [1.0, 3.0, 2.0],
    [4.0, 0.0, 1.0]
])

# Model weights / Pesos del modelo
w = np.array([0.5, -0.2, 0.8])

# Matrix-vector multiplication / Multiplicación matriz-vector
s = X @ w

print("Matrix shape:", X.shape)
print("Weights shape:", w.shape)
print("Scores:", s)
print("Scores shape:", s.shape)
