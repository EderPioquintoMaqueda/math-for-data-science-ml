
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Generate correlated features / Generar variables correlacionadas
rng = np.random.default_rng(42)
n = 200

x1 = rng.normal(size=n)
x2 = x1 + 0.05 * rng.normal(size=n)
X = np.column_stack([x1, x2])

# Generate a target / Generar la variable objetivo
y = 3 * x1 + 3 * x2 + rng.normal(scale=0.5, size=n)

# Ordinary least squares / Mínimos cuadrados ordinarios
ols = LinearRegression()
ols.fit(X, y)

# Ridge with standardized features / Ridge con variables estandarizadas
ridge = make_pipeline(
    StandardScaler(),
    Ridge(alpha=1.0)
)
ridge.fit(X, y)

print("OLS coefficients:", ols.coef_)
print("Ridge coefficients:", ridge.named_steps["ridge"].coef_)
print("OLS R^2:", ols.score(X, y))
print("Ridge R^2:", ridge.score(X, y))
