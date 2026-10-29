
"""
Conditional Probability and Independence
----------------------------------------
Calculate conditional probabilities and test event independence
using a fair six-sided die and two independent die rolls.
"""

import numpy as np


# ---------------------------------------------------------
# PART 1: Conditional probability with one die
# ---------------------------------------------------------

# Define every possible outcome of a fair six-sided die.
sample_space = np.array([1, 2, 3, 4, 5, 6])

# Event A: the outcome is even.
# Boolean indexing selects values that satisfy the condition.
event_a = sample_space[sample_space % 2 == 0]

# Event B: the outcome is greater than 3.
event_b = sample_space[sample_space > 3]

# Intersection: outcomes that belong to both events.
# np.intersect1d returns the common values without duplicates.
intersection_ab = np.intersect1d(event_a, event_b)

# Calculate the individual probabilities.
# Each outcome is equally likely, so count / total is valid.
probability_a = len(event_a) / len(sample_space)
probability_b = len(event_b) / len(sample_space)

# Calculate the intersection probability.
probability_intersection = (
    len(intersection_ab) / len(sample_space)
)

# Conditional probability:
# P(A | B) = P(A intersection B) / P(B).
probability_a_given_b = (
    probability_intersection / probability_b
)


# ---------------------------------------------------------
# PART 2: Check whether A and B are independent
# ---------------------------------------------------------

# Independent events satisfy P(A intersection B) = P(A) * P(B).
independence_product = probability_a * probability_b

# np.isclose compares floating-point values with a tolerance.
# This avoids unreliable exact comparisons between decimals.
are_independent = np.isclose(
    probability_intersection,
    independence_product
)


# ---------------------------------------------------------
# PART 3: Two independent die rolls
# ---------------------------------------------------------

# np.array([...]) defines the possible outcomes for one die.
die = np.array([1, 2, 3, 4, 5, 6])

# np.meshgrid creates combinations of the outcomes of two dice.
# indexing="ij" preserves the expected row/column orientation.
first_die, second_die = np.meshgrid(die, die, indexing="ij")

# .ravel() converts each 6-by-6 grid into a one-dimensional array.
# Each index now represents one possible pair of die outcomes.
first_roll = first_die.ravel()
second_roll = second_die.ravel()

# There are 6 * 6 = 36 equally likely ordered outcomes.
total_pairs = len(first_roll)

# Event C: the first die shows an even number.
event_c = first_roll % 2 == 0

# Event D: the second die shows a number greater than 3.
event_d = second_roll > 3

# Boolean arrays can be combined with & for logical AND.
# Parentheses are important when combining NumPy conditions.
event_c_and_d = event_c & event_d

# The mean of a Boolean array equals the proportion of True values:
# True behaves like 1 and False behaves like 0.
probability_c = np.mean(event_c)
probability_d = np.mean(event_d)
probability_c_and_d = np.mean(event_c_and_d)

# Check independence for the two-die experiment.
two_dice_independent = np.isclose(
    probability_c_and_d,
    probability_c * probability_d
)


# ---------------------------------------------------------
# PART 4: Display and verify results
# ---------------------------------------------------------

print("CONDITIONAL PROBABILITY")
print("-" * 40)

print(f"Event A (even): {event_a}")
print(f"Event B (greater than 3): {event_b}")
print(f"Intersection A and B: {intersection_ab}")

print(f"\nP(A) = {probability_a:.3f}")
print(f"P(B) = {probability_b:.3f}")
print(f"P(A intersection B) = {probability_intersection:.3f}")
print(f"P(A | B) = {probability_a_given_b:.3f}")

print(f"\nAre A and B independent? {are_independent}")

print("\nTWO INDEPENDENT DIE ROLLS")
print("-" * 40)

print(f"Number of possible pairs: {total_pairs}")
print(f"P(C) = {probability_c:.3f}")
print(f"P(D) = {probability_d:.3f}")
print(f"P(C intersection D) = {probability_c_and_d:.3f}")
print(f"Are C and D independent? {two_dice_independent}")

# Confirm the mathematical relationships.
assert np.isclose(
    probability_a_given_b,
    2 / 3
)

assert not are_independent

assert np.isclose(
    probability_c_and_d,
    probability_c * probability_d
)

assert two_dice_independent

print("\nAll probability checks passed.")
