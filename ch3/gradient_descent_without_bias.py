import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def predict(X, w, b):
    return X * w + b


def loss(X, Y, w, b):
    return np.average((predict(X, w, b) - Y) ** 2)


def gradient(X, Y, w, b):
    w_gradient = 2 * np.average(X * (predict(X, w, b) - Y))
    b_gradient = 2 * np.average(predict(X, w, b) - Y)

    return w_gradient, b_gradient


# lr: learning rate
def train(X, Y, iterations, lr):
    b = 0
    w = 0
    for i in range(iterations):
        current_loss = loss(X, Y, w, b)
        print("Iteration %4d => loss: %.10f" % (i, current_loss))
        w_gradient, b_gradient = gradient(X, Y, w, b)
        w -= w_gradient * lr
        b -= b_gradient * lr

    return w, b


X, Y = np.loadtxt("pizzas.txt", skiprows=1, unpack=True)
w, b = train(X, Y, iterations=10_000, lr=0.001)
print("\nw=%.3f, b=%.3f" % (w, b))

print("Prediction: x=%d => y=%.2f" % (20, predict(20, w, b)))

# Plot the chart
sns.set()
plt.plot(X, Y, "bo")
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel("Reservations", fontsize=30)
plt.ylabel("Pizzas", fontsize=30)
x_edge, y_edge = 50, 50
plt.axis([0, x_edge, 0, y_edge])
plt.plot([0, x_edge], [b, predict(x_edge, w, b)], linewidth=1.0, color="g")
plt.show()
