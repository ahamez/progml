import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def predict(X, w, b):
    return X * w + b


def loss(X, Y, w, b):
    return np.average((predict(X, w, b) - Y) ** 2)


# lr: learning rate
def train(X, Y, iterations, lr):
    b = 0
    w = 0
    for i in range(iterations):
        current_loss = loss(X, Y, w, b)
        print("Iteration %4d => loss: %.6f" % (i, current_loss))
        # print("loss(X, Y, w + lr): %.6f", loss(X, Y, w + lr))
        # print("loss(X, Y, w - lr): %.6f", loss(X, Y, w - lr))

        if loss(X, Y, w + lr, b) < current_loss:
            w += lr
        elif loss(X, Y, w - lr, b) < current_loss:
            w -= lr
        elif loss(X, Y, w, b + lr) < current_loss:
            b += lr
        elif loss(X, Y, w, b - lr) < current_loss:
            b -= lr
        else:
            return w, b

    raise Exception("Could not converge")


X, Y = np.loadtxt("pizzas.txt", skiprows=1, unpack=True)
w, b = train(X, Y, iterations=100_000, lr=0.01)
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
