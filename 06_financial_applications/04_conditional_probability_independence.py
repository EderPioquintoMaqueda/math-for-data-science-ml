
import numpy as np


# Define the sample space for a fair six-sided die.
die_outcomes = np.array([1, 2, 3, 4, 5, 6])

# Define events for the die.
event_even = die_outcomes % 2 == 0
event_greater_than_three = die_outcomes > 3

# Calculate the probability of each event.
probability_even = np.mean(event_even)
probability_greater_than_three = np.mean(event_greater_than_three)

# Calculate the intersection probability.
event_intersection = event_even & event_greater_than_three
probability_intersection = np.mean(event_intersection)

# Calculate conditional probability using the reduced sample space.
reduced_sample_space = die_outcomes[event_greater_than_three]
probability_even_given_greater_than_three = np.mean(
    reduced_sample_space % 2 == 0
)

print("CONDITIONAL PROBABILITY")
print("Reduced sample space:", reduced_sample_space)
print(f"P(even) = {probability_even:.4f}")
print(f"P(greater than three) = {probability_greater_than_three:.4f}")
print(f"P(intersection) = {probability_intersection:.4f}")
print(
    "P(even | greater than three) = "
    f"{probability_even_given_greater_than_three:.4f}"
)


# Define the sample spaces for a fair coin and a fair die.
coin_outcomes = np.array([0, 1])  # 0 = tails, 1 = heads

# Generate every possible coin-and-die combination.
joint_outcomes = np.array([
    (coin, die)
    for coin in coin_outcomes
    for die in die_outcomes
])

# Identify heads and even-number outcomes.
event_heads = joint_outcomes[:, 0] == 1
event_even_joint = joint_outcomes[:, 1] % 2 == 0

# Calculate individual and joint probabilities.
probability_heads = np.mean(event_heads)
probability_even_joint = np.mean(event_even_joint)

probability_heads_and_even = np.mean(
    event_heads & event_even_joint
)

# Check the independence condition.
product_of_probabilities = probability_heads * probability_even_joint

print("\nINDEPENDENCE")
print(f"P(heads) = {probability_heads:.4f}")
print(f"P(even) = {probability_even_joint:.4f}")
print(f"P(heads and even) = {probability_heads_and_even:.4f}")
print(f"P(heads) * P(even) = {product_of_probabilities:.4f}")

print(
    "Independent:",
    np.isclose(probability_heads_and_even, product_of_probabilities)
)
