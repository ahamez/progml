import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# "y-hat"
def forward(X, w):
    weighted_sum = np.matmul(X, w)
    return sigmoid(weighted_sum)


def classify(X, w):
    return np.round(forward(X, w))


def loss(X, Y, w):
    y_hat = forward(X, w)
    first_term = Y * np.log(y_hat)
    second_term = (1 - Y) * np.log(1 - y_hat)
    return -np.average(first_term + second_term)


def gradient(X, Y, w):
    nb_examples = X.shape[0]
    return np.matmul(X.T, (forward(X, w) - Y)) / nb_examples


# lr: learning rate
def train(X, Y, iterations, lr):
    nb_weights = X.shape[1]
    w = np.zeros((nb_weights, 1))
    for i in range(iterations):
        print("Iteration {:4d} => loss: {:20f}".format(i, loss(X, Y, w)))
        w -= gradient(X, Y, w) * lr

    return w


def test(X, Y, w):
    total_examples = X.shape[0]
    correct_results = np.sum(classify(X, w) == Y)
    success_percent = correct_results * 100 / total_examples
    print("Success: {}/{} ({:.2f}%)".format(correct_results, total_examples, success_percent))


x1, x2, x3, y = np.loadtxt("../book/05_discerning/police.txt", skiprows=1, unpack=True)

# build matrix X from all columns
X = np.column_stack((np.ones(x1.size), x1, x2, x3))
# convert y array into a matrix
Y = y.reshape(-1, 1)

w = train(X, Y, iterations=10_000, lr=0.001)

print(f"{w.T}")

test(X, Y, w)
