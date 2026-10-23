
"""
Lesson 7: Regression evaluation metrics.

Calculate residuals, MSE, RMSE, and R-squared
without using a machine learning library.
"""

import numpy as np

# Observed and predicted values
y_true = np.array([100, 200, 300, 400, 500], dtype=float)
y_pred = np.array([110, 190, 280, 420, 500], dtype=float)

# Calculate residuals
residuals = y_true - y_pred

# Sum of squared errors
sse = np.sum(residuals ** 2)

# Mean squared error
mse = np.mean(residuals ** 2)

# Root mean squared error
rmse = np.sqrt(mse)

# Total sum of squares
mean_y = np.mean(y_true)
sst = np.sum((y_true - mean_y) ** 2)

# Coefficient of determination
r_squared = 1 - (sse / sst)

# Baseline: predict the mean for every observation
baseline_predictions = np.full_like(y_true, mean_y)
baseline_mse = np.mean((y_true - baseline_predictions) ** 2)

# Display results
print("Observed values:     ", y_true)
print("Predicted values:    ", y_pred)
print("Residuals:           ", residuals)

print(f"\nSSE:                 {sse:.2f}")
print(f"MSE:                 {mse:.2f}")
print(f"RMSE:                {rmse:.2f}")
print(f"R-squared:           {r_squared:.4f}")
print(f"Baseline MSE:        {baseline_mse:.2f}")

print("\nInterpretation:")
print(f"RMSE is {rmse:.2f} units of the target variable.")
print(f"The model reduces squared error by {r_squared:.1%} "
      "compared with the mean baseline.")
