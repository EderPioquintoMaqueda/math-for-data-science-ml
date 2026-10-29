
"""
Lesson 05: Expected Value and Variance.

This script calculates:
1. The expected value of a fair die.
2. The variance and standard deviation.
3. The same measures for a Bernoulli random variable.
4. A verification using two equivalent variance formulas.
"""

import numpy as np


# --------------------------------------------------
# 1. Define the discrete random variable X
# --------------------------------------------------

# X represents the possible outcomes of a fair die.
values = np.array([1, 2, 3, 4, 5, 6])

# Each outcome has probability 1/6 because the die is fair.
probabilities = np.full(len(values), 1 / len(values))


# --------------------------------------------------
# 2. Calculate the expected value
# --------------------------------------------------

# np.sum() adds all elements.
# Multiplying each value by its probability gives
# the weighted average of the possible outcomes.
expected_value = np.sum(values * probabilities)


# --------------------------------------------------
# 3. Calculate the variance
# --------------------------------------------------

# Calculate the squared distance from each value
# to the expected value.
squared_deviations = (values - expected_value) ** 2

# Weight each squared deviation by its probability,
# then add the results.
variance = np.sum(squared_deviations * probabilities)

# Standard deviation is the square root of variance.
standard_deviation = np.sqrt(variance)


# --------------------------------------------------
# 4. Verify the alternative variance formula
# --------------------------------------------------

# E[X^2] is the expected value of the squared outcomes.
expected_squared_value = np.sum((values ** 2) * probabilities)

# Equivalent formula: Var(X) = E[X^2] - E[X]^2.
variance_alternative = expected_squared_value - expected_value**2

# Confirm that both calculations produce the same result.
assert np.isclose(variance, variance_alternative)


print("FAIR DIE")
print("--------")
print(f"Expected value: {expected_value:.4f}")
print(f"Variance: {variance:.4f}")
print(f"Standard deviation: {standard_deviation:.4f}")
print(f"Variance using alternative formula: {variance_alternative:.4f}")


# --------------------------------------------------
# 5. Analyze a Bernoulli random variable Y
# --------------------------------------------------

# Y = 1 means the die result is even.
# Y = 0 means the die result is odd.
bernoulli_values = np.array([0, 1])

# Three of the six die outcomes are odd,
# and three are even. Both probabilities are 0.5.
bernoulli_probabilities = np.array([0.5, 0.5])

# Calculate the expected value of Y.
bernoulli_expected_value = np.sum(
    bernoulli_values * bernoulli_probabilities
)

# Calculate the variance of Y.
bernoulli_variance = np.sum(
    (bernoulli_values - bernoulli_expected_value) ** 2
    * bernoulli_probabilities
)

print("\nBERNOULLI RANDOM VARIABLE")
print("-------------------------")
print(f"Expected value: {bernoulli_expected_value:.4f}")
print(f"Variance: {bernoulli_variance:.4f}")
print(f"Standard deviation: {np.sqrt(bernoulli_variance):.4f}")
