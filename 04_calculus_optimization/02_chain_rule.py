
# Chain rule and its application to machine learning
# Regla de la cadena y su aplicación en machine learning

# Example 1: A composite mathematical function
# Ejemplo 1: Una función matemática compuesta

x = 2.0

# y = (3x + 2)^2
u = 3 * x + 2
y_function = u ** 2

# Apply the chain rule: dy/dx = 2u * 3
derivative = 2 * u * 3

print("Composite function")
print(f"Function value: {y_function:.4f}")
print(f"Derivative: {derivative:.4f}")


# Example 2: One training observation
# Ejemplo 2: Una observación de entrenamiento

x = 2.0
y_true = 5.0

# Initial model parameters
w = 1.0
b = 0.0

# Forward pass: calculate the prediction and loss
# Propagación hacia adelante: calcular predicción y pérdida

y_pred = w * x + b
error = y_pred - y_true
loss = 0.5 * error ** 2

# Chain rule: calculate parameter gradients
# Regla de la cadena: calcular los gradientes

d_loss_d_pred = error
d_pred_d_w = x
d_pred_d_b = 1.0

d_loss_d_w = d_loss_d_pred * d_pred_d_w
d_loss_d_b = d_loss_d_pred * d_pred_d_b

print("\nBefore the update")
print(f"Prediction: {y_pred:.4f}")
print(f"True value: {y_true:.4f}")
print(f"Loss: {loss:.4f}")
print(f"Gradient for w: {d_loss_d_w:.4f}")
print(f"Gradient for b: {d_loss_d_b:.4f}")

# Gradient descent: update the parameters
# Descenso del gradiente: actualizar los parámetros

learning_rate = 0.1

w = w - learning_rate * d_loss_d_w
b = b - learning_rate * d_loss_d_b

# Recalculate the prediction and loss
# Recalcular la predicción y la pérdida

y_pred_updated = w * x + b
loss_updated = 0.5 * (y_pred_updated - y_true) ** 2

print("\nAfter the update")
print(f"Updated w: {w:.4f}")
print(f"Updated b: {b:.4f}")
print(f"New prediction: {y_pred_updated:.4f}")
print(f"New loss: {loss_updated:.4f}")
