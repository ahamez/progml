import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def predict(X, w):
    return np.matmul(X, w)


def loss(X, Y, w):
    return np.average((predict(X, w) - Y) ** 2)


def gradient(X, Y, w):
    return 2 * np.matmul(X.T, (predict(X, w) - Y)) / X.shape[0]


# lr: learning rate
def train(X, Y, iterations, lr):
    w = np.zeros((X.shape[1], 1))
    for i in range(iterations):
        current_loss = loss(X, Y, w)
        print("Iteration %4d => loss: %.20f" % (i, current_loss))
        w -= gradient(X, Y, w) * lr

    return w


x1, x2, x3, y = np.loadtxt('../book/data/life-expectancy-without-country-names.txt', skiprows=1, unpack=True)

# build matrix X from all columns
X = np.column_stack((np.ones(x1.size), x1, x2, x3))
# convert y array into a matrix
Y = y.reshape(-1, 1)

# print("{}".format(X[:2]))

w = train(X, Y, iterations=100_000, lr=0.001)

print("Weights: {}".format(w.T))
print("Some predictions")
for i in range(5):
    print("X[{}] -> {} (label: {})".format(i, predict(X[i], w), Y[i]))
