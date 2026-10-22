
import numpy as np

# Define a vector / Definir un vector
x = np.array([3.0, 4.0])

print("Vector:", x)
print("Shape:", x.shape)
print("Dimension:", x.ndim)
print("Number of elements:", x.size)

alpha = 2

scaled_x = alpha * x

print("Scaled vector:", scaled_x)

y = np.array([1.0, 2.0])

sum_xy = x + y

print("Vector x:", x)
print("Vector y:", y)
print("Sum:", sum_xy)

norm_x = np.linalg.norm(x)

print("Euclidean norm:", norm_x)
