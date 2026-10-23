
"""
Lesson 8: Descriptive statistics.

Calculate measures of central tendency and dispersion
using NumPy.
"""

import numpy as np

# Weekly expenses, in thousands of pesos
data = np.array([10, 12, 12, 14, 52], dtype=float)

# Central tendency
mean = np.mean(data)
median = np.median(data)

# Population dispersion
population_variance = np.var(data, ddof=0)
population_std = np.std(data, ddof=0)

# Sample dispersion
sample_variance = np.var(data, ddof=1)
sample_std = np.std(data, ddof=1)

# Quantiles and interquartile range
q1 = np.quantile(data, 0.25)
q2 = np.quantile(data, 0.50)
q3 = np.quantile(data, 0.75)
iqr = q3 - q1

print("Data:", data)

print("\nCentral tendency")
print(f"Mean:   {mean:.2f}")
print(f"Median: {median:.2f}")

print("\nPopulation dispersion")
print(f"Variance: {population_variance:.2f}")
print(f"Std dev:  {population_std:.2f}")

print("\nSample dispersion")
print(f"Variance: {sample_variance:.2f}")
print(f"Std dev:  {sample_std:.2f}")

print("\nQuantiles")
print(f"Q1 (25th percentile): {q1:.2f}")
print(f"Q2 (50th percentile): {q2:.2f}")
print(f"Q3 (75th percentile): {q3:.2f}")
print(f"IQR:                  {iqr:.2f}")
