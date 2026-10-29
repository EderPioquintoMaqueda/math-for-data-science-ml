"""
Lesson 04: Random Variables and Probability Mass Functions (PMF).

This script demonstrates:
1. A discrete random variable representing a die roll.
2. Its probability mass function (PMF).
3. A Bernoulli random variable indicating whether the roll is even.
4. Validation of the probability distribution.
"""

import numpy as np


# --------------------------------------------------
# 1. Define the sample space
# --------------------------------------------------

# A fair die can produce six possible outcomes.
sample_space = np.array([1, 2, 3, 4, 5, 6])

# len() returns the number of elements in the array.
number_of_outcomes = len(sample_space)


# --------------------------------------------------
# 2. Define a discrete random variable X
# --------------------------------------------------

# X represents the numerical result of rolling the die.
# Here, each possible outcome maps to itself.
X = sample_space.copy()


# --------------------------------------------------
# 3. Calculate the Probability Mass Function (PMF)
# --------------------------------------------------

# Because the die is fair, every outcome is equally likely.
# np.full() creates an array filled with the same value.
probabilities_X = np.full(
    shape=number_of_outcomes,
    fill_value=1 / number_of_outcomes,
)

print("DISCRETE RANDOM VARIABLE X")
print("--------------------------")

# zip() combines corresponding elements from two iterables.
# :.3f formats each probability to three decimal places.
for value, probability in zip(X, probabilities_X):
    print(f"P(X = {value}) = {probability:.3f}")


# --------------------------------------------------
# 4. Validate the probability distribution
# --------------------------------------------------

# np.all() checks whether every element satisfies a condition.
# np.isclose() compares floating-point numbers with tolerance.
assert np.all(probabilities_X >= 0)
assert np.isclose(np.sum(probabilities_X), 1.0)

print(f"\nSum of probabilities: {np.sum(probabilities_X):.3f}")
print("Valid probability distribution: True")


# --------------------------------------------------
# 5. Define a Bernoulli random variable Y
# --------------------------------------------------

# np.where(condition, value_if_true, value_if_false):
# returns 1 when the die result is even, and 0 otherwise.
Y = np.where(X % 2 == 0, 1, 0)

# np.unique() identifies the distinct values in Y.
# return_counts=True also returns their frequencies.
possible_values_Y, counts_Y = np.unique(
    Y,
    return_counts=True,
)

# The PMF is the frequency of each value divided by
# the total number of possible die outcomes.
probabilities_Y = counts_Y / number_of_outcomes

print("\nBERNOULLI RANDOM VARIABLE Y")
print("---------------------------")

for value, probability in zip(
    possible_values_Y,
    probabilities_Y,
):
    print(f"P(Y = {value}) = {probability:.3f}")

assert np.all(probabilities_Y >= 0)
assert np.isclose(np.sum(probabilities_Y), 1.0)

print(f"\nSum of probabilities: {np.sum(probabilities_Y):.3f}")
print("Valid probability distribution: True")
