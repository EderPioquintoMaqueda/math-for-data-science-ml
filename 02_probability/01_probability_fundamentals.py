
"""
Probability Fundamentals
------------------------
Learn basic probability rules using a fair six-sided die.

Main concepts:
- Sample space
- Events
- Classical probability
- Complement probability
- Union probability
"""

import numpy as np


# 1. Define the sample space.
# np.array([...]) creates a NumPy array from a sequence of values.
# Each value represents one possible outcome of the die.
sample_space = np.array([1, 2, 3, 4, 5, 6])


# 2. Define the event: obtaining an even number.
# The condition (sample_space % 2 == 0) checks which values are even.
# The % operator returns the remainder after division.
# Boolean indexing keeps only the values where the condition is True.
even_numbers = sample_space[sample_space % 2 == 0]


# 3. Calculate classical probability.
# len(...) returns the number of elements in an array.
# Division by the total number of equally likely outcomes gives P(A).
probability_even = len(even_numbers) / len(sample_space)


# 4. Calculate the complement probability.
# The complement is the event that an even number does not occur.
probability_not_even = 1 - probability_even


# 5. Define another event: obtaining a number greater than 3.
greater_than_three = sample_space[sample_space > 3]


# 6. Calculate the probability of the second event.
probability_greater_than_three = (
    len(greater_than_three) / len(sample_space)
)


# 7. Calculate the probability of the union of two events.
# Event A: even numbers = {2, 4, 6}
# Event B: numbers greater than 3 = {4, 5, 6}
#
# np.union1d(...) returns the unique values present in either array.
union_events = np.union1d(even_numbers, greater_than_three)

probability_union = len(union_events) / len(sample_space)


# 8. Calculate the intersection of the two events.
# np.intersect1d(...) returns the values shared by both arrays.
intersection_events = np.intersect1d(
    even_numbers,
    greater_than_three
)

probability_intersection = (
    len(intersection_events) / len(sample_space)
)


# 9. Verify the union formula.
# This expression should equal probability_union.
union_formula = (
    probability_even
    + probability_greater_than_three
    - probability_intersection
)


# 10. Display the results.
# f-strings insert variable values inside text using {variable_name}.
# :.2f formats a numeric value with two decimal places.
print("PROBABILITY FUNDAMENTALS")
print("-" * 35)

print(f"Sample space: {sample_space}")
print(f"Even numbers: {even_numbers}")
print(f"Numbers greater than 3: {greater_than_three}")

print(f"\nP(Even) = {probability_even:.2f}")
print(f"P(Not even) = {probability_not_even:.2f}")

print(
    f"P(Greater than 3) = "
    f"{probability_greater_than_three:.2f}"
)

print(f"P(Union) = {probability_union:.2f}")
print(f"P(Intersection) = {probability_intersection:.2f}")
print(f"Union formula result = {union_formula:.2f}")

# np.isclose(...) checks whether two floating-point values
# are approximately equal, avoiding precision issues.
assert np.isclose(probability_union, union_formula)

print("\nUnion formula verified successfully.")
