
import numpy as np

# Define two observations / Definir dos observaciones
x = np.array([2.0, 1.0, 2.0])
y = np.array([1.0, 3.0, 2.0])

# Dot product / Producto punto
dot_product = np.dot(x, y)

# Euclidean norms / Normas euclidianas
norm_x = np.linalg.norm(x)
norm_y = np.linalg.norm(y)

# Cosine similarity / Similitud coseno
cosine_similarity = dot_product / (norm_x * norm_y)

# Angle in degrees / Ángulo en grados
angle_rad = np.arccos(np.clip(cosine_similarity, -1.0, 1.0))
angle_deg = np.degrees(angle_rad)

print(f"Dot product: {dot_product:.4f}")
print(f"Norm of x: {norm_x:.4f}")
print(f"Norm of y: {norm_y:.4f}")
print(f"Cosine similarity: {cosine_similarity:.4f}")
print(f"Angle: {angle_deg:.2f} degrees")

# Projection of y onto x / Proyección de y sobre x
projection_y_onto_x = (np.dot(y, x) / np.dot(x, x)) * x

# Orthogonal residual / Residuo ortogonal
residual = y - projection_y_onto_x

print("Projection:", projection_y_onto_x)
print("Residual:", residual)
print("Residual dot x:", np.dot(residual, x))
