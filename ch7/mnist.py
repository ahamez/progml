import numpy as np
import gzip
import struct


def load_images(filename):
    with gzip.open(filename, 'rb') as f:
        _, n_images, columns, rows = struct.unpack('>IIII', f.read(16))
        all_pixels = np.frombuffer(f.read(), dtype=np.uint8)
        return all_pixels.reshape(n_images, columns * rows)


def prepend_bias(X):
    # axis=1 -> insert a column, not a row
    return np.insert(X, 0, 1, axis=1)


def load_labels(filename):
    with gzip.open(filename, 'rb') as f:
        f.read(8)
        all_labels = f.read()
        return np.frombuffer(all_labels, dtype=np.uint8).reshape(-1, 1)


def encode_fives(Y):
    # convert all 5s to 1, and everything else to 0
    return (Y == 5).astype(int)


def one_hot_encode(Y):
    n_labels = Y.shape[0]
    n_classes = 10
    encoded_Y = np.zeros((n_labels, n_classes))
    for i in range(n_labels):
        label = Y[i]
        encoded_Y[i][label] = 1
    return encoded_Y


X_train = prepend_bias(load_images("../book/data/mnist/train-images-idx3-ubyte.gz"))

for i in range(28*28 + 1):
    print("{},".format(X_train[0, i]), end='')
print("\n")

X_test = prepend_bias(load_images("../book/data/mnist/t10k-images-idx3-ubyte.gz"))

Y_train_unencoded = load_labels("../book/data/mnist/train-labels-idx1-ubyte.gz")
Y_train = one_hot_encode(Y_train_unencoded)

Y_test = load_labels("../book/data/mnist/t10k-labels-idx1-ubyte.gz")
