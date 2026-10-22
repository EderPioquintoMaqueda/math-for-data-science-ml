
import numpy as np

# Coefficient matrix / Matriz de coeficientes
A = np.array([
    [2.0, 1.0, 1.0],
    [1.0, 2.0, 1.0],
    [1.0, 1.0, 2.0]
])

# Right-hand-side vector / Vector de resultados
b = np.array([7.0, 8.0, 9.0])

# Solve A @ x = b / Resolver el sistema
x = np.linalg.solve(A, b)

print("Solution [a, b, c]:", x)

# Verify the solution / Verificar la solución
print("Reconstructed b:", A @ x)
print("Original b:", b)
print("Correct solution:", np.allclose(A @ x, b))
det_A = np.linalg.det(A)
rank_A = np.linalg.matrix_rank(A)

print("Determinant:", det_A)
print("Rank:", rank_A)


# A system with dependent rows / Sistema con filas dependientes
C = np.array([
    [1.0, 2.0],
    [2.0, 4.0]
])

print("Matrix C:")
print(C)
print("Rank:", np.linalg.matrix_rank(C))
print("Determinant:", np.linalg.det(C))

# The equations are x + 2y = 3 and 2x + 4y = 6.
# Ambas ecuaciones representan la misma recta.
d = np.array([3.0, 6.0])

print("Consistent system: infinitely many solutions.")

# Now change the second equation to 2x + 4y = 7.
# Ahora las ecuaciones son contradictorias.
d_inconsistent = np.array([3.0, 7.0])

print("Inconsistent system rank:",
      np.linalg.matrix_rank(np.column_stack([C, d_inconsistent])))
