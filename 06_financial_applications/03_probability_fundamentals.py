
import numpy as np


# Define the sample space for a fair six-sided die.
outcomes = np.array([1, 2, 3, 4, 5, 6])

# Define events using Boolean masks.
event_even = outcomes % 2 == 0
event_greater_than_four = outcomes > 4

# Calculate individual probabilities.
probability_even = np.mean(event_even)
probability_greater_than_four = np.mean(event_greater_than_four)

# Calculate intersection and union.
event_intersection = event_even & event_greater_than_four
event_union = event_even | event_greater_than_four

probability_intersection = np.mean(event_intersection)
probability_union = np.mean(event_union)

# Calculate the complement of the even-number event.
event_not_even = ~event_even
probability_not_even = np.mean(event_not_even)

# Verify the addition rule.
probability_union_formula = (
    probability_even
    + probability_greater_than_four
    - probability_intersection
)

print("Sample space:", outcomes)
print("Even outcomes:", outcomes[event_even])
print("Outcomes greater than four:", outcomes[event_greater_than_four])
print("Intersection:", outcomes[event_intersection])
print("Union:", outcomes[event_union])

print(f"\nP(A) = {probability_even:.4f}")
print(f"P(B) = {probability_greater_than_four:.4f}")
print(f"P(A intersection B) = {probability_intersection:.4f}")
print(f"P(A union B) = {probability_union:.4f}")
print(f"P(A complement) = {probability_not_even:.4f}")

print(f"\nAddition rule result: {probability_union_formula:.4f}")
