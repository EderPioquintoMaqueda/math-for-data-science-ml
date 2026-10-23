
"""
Lesson 9: Distributions, skewness, and outliers.

Calculate moment-based skewness and detect potential outliers
using the interquartile range (IQR).
"""

import numpy as np

# Weekly expenses, in thousands of pesos
data = np.array([10, 12, 12, 14, 52], dtype=float)

# Measures of central tendency
mean = np.mean(data)
median = np.median(data)

# Population standard deviation
std = np.std(data, ddof=0)

# Moment-based skewness
if std > 0:
    skewness = np.mean(((data - mean) / std) ** 3)
else:
    skewness = 0.0

# Quartiles and interquartile range
q1 = np.quantile(data, 0.25)
q3 = np.quantile(data, 0.75)
iqr = q3 - q1

# IQR outlier fences
lower_fence = q1 - 1.5 * iqr
upper_fence = q3 + 1.5 * iqr

# Identify potential outliers
outliers = data[(data < lower_fence) | (data > upper_fence)]

print("Data:", data)
print(f"Mean: {mean:.2f}")
print(f"Median: {median:.2f}")
print(f"Population standard deviation: {std:.2f}")
print(f"Moment-based skewness: {skewness:.4f}")

print(f"\nQ1: {q1:.2f}")
print(f"Q3: {q3:.2f}")
print(f"IQR: {iqr:.2f}")
print(f"Lower fence: {lower_fence:.2f}")
print(f"Upper fence: {upper_fence:.2f}")
print("Potential outliers:", outliers)

if skewness > 0:
    print("\nThe data show positive skewness.")
elif skewness < 0:
    print("\nThe data show negative skewness.")
else:
    print("\nThe moment-based skewness is zero.")
