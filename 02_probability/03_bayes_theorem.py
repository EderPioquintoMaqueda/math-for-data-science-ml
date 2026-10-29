
"""
Bayes' Theorem
--------------
Update the probability of fraud after observing a suspicious alert.
All probabilities are hypothetical and used for learning.
"""


# Prior probability:
# P(Fraud) is the probability of fraud before observing an alert.
prior_fraud = 0.01

# Complement:
# If a transaction is either fraudulent or legitimate,
# the probabilities of these two cases sum to 1.
prior_legitimate = 1 - prior_fraud

# Likelihood:
# P(Alert | Fraud) is the probability of an alert when fraud exists.
alert_given_fraud = 0.90

# False-positive rate:
# P(Alert | Legitimate) is the probability of an alert
# even though the transaction is legitimate.
alert_given_legitimate = 0.05


# Total probability:
# An alert can occur because of fraud or because of a false positive.
probability_alert = (
    alert_given_fraud * prior_fraud
    + alert_given_legitimate * prior_legitimate
)


# Bayes' theorem:
# P(Fraud | Alert) = P(Alert | Fraud) * P(Fraud) / P(Alert).
posterior_fraud = (
    alert_given_fraud * prior_fraud
    / probability_alert
)


# Calculate the probability that an alerted transaction
# is actually legitimate.
posterior_legitimate = 1 - posterior_fraud


# Display the results.
# :.4f shows four decimal places.
# :.2% formats a decimal as a percentage with two decimals.
print("BAYES' THEOREM")
print("-" * 40)

print(f"Prior probability of fraud: {prior_fraud:.2%}")
print(f"Probability of an alert: {probability_alert:.2%}")
print(f"Probability of fraud given alert: {posterior_fraud:.2%}")
print(
    "Probability of legitimate transaction given alert: "
    f"{posterior_legitimate:.2%}"
)


# Verify that the posterior probabilities sum to 1.
assert abs(
    posterior_fraud + posterior_legitimate - 1
) < 1e-10

print("\nBayes' theorem calculation verified.")
