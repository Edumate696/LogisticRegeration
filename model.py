import numpy as np


def sigmoid(z):
    return 1.0 / (1 + np.exp(-z))


def forward_prop(X, W, b):
    z = np.dot(W.T, X) + b
    a = sigmoid(z)
    return a


def calc_loss(y, a, m):
    J = 0
    for i in range(m):
        if y[0, i] == 1:
            k = -np.log(a[0, i])
            J += 9999 if k is np.inf else k
        else:
            k = -np.log(1 - a[0, i])
            J += 9999 if k is np.inf else k
    return J / m


def backward_prop(X, y, a, m):
    dz = a - y
    dW = np.sum(np.dot(X, dz.T), axis=1) / m
    db = np.sum(dz) / m
    return dW, db


def update_params(W, b, dW, db, alpha):
    W = W - alpha * dW
    b = b - alpha * db
    return W, b


class LogisticRegression:
    def __init__(self, learning_rate=0.01):
        self.__params = {}
        self.__lr = learning_rate

    def fit(self, X, y, num_iter=1000, refit=True):
        with np.errstate(all='ignore'):
            W = self.__params.get('W', None)
            b = self.__params.get('b', None)
            if refit or W is None or b is None:
                W = np.random.randn(X.shape[0], 1)
                b = np.random.randn()

            m = y.shape[1]

            for i in range(num_iter):
                a = forward_prop(X, W, b)

                J = calc_loss(y, a, m)
                print(J)

                dW, db = backward_prop(X, y, a, m)

                W, b = update_params(W, b, dW, db, self.__lr)
