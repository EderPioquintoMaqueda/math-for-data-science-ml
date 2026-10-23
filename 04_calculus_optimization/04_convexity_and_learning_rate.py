
"""
Lesson 4: Convexity and learning rate.

Compare gradient descent with different learning rates
on a simple convex cost function.
"""


def cost(w):
    """Calculate the cost function."""
    return (w - 3) ** 2


def gradient(w):
    """Calculate the derivative of the cost function."""
    return 2 * (w - 3)


def gradient_descent(learning_rate, initial_w=0.0, iterations=10):
    """Optimize w using gradient descent."""
    w = initial_w
    history = [(0, w, cost(w))]

    for step in range(1, iterations + 1):
        w = w - learning_rate * gradient(w)
        history.append((step, w, cost(w)))

    return history


learning_rates = [0.1, 0.5, 1.1]

for learning_rate in learning_rates:
    history = gradient_descent(learning_rate)

    print(f"\nLearning rate: {learning_rate}")
    print("Step | Parameter w | Cost J(w)")

    for step, w, loss in history:
        if step in [0, 1, 2, 5, 10]:
            print(f"{step:>4} | {w:>11.4f} | {loss:>10.4f}")
