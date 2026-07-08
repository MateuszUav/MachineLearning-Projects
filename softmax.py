from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np


def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):

        # konfiguruje generator znaczników i mapę kolorów
        markers = ('s', 'x', 'o', '^', 'v')
        colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
        cmap = ListedColormap(colors[:len(np.unique(y))])

        # rysuje wykres powierzchni decyzyjnej
        x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
        Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
        Z = Z.reshape(xx1.shape)
        plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
        plt.xlim(xx1.min(), xx1.max())
        plt.ylim(xx2.min(), xx2.max())

        # rysuje wykres wszystkich próbek
        for idx, cl in enumerate(np.unique(y)):
            plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8, c=cmap(idx), marker=markers[idx], label=cl, edgecolor='black')

import numpy as np
import matplotlib.pylab as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from plotka import plot_decision_regions


class LogisticRegressionGD(object):
    def __init__(self, eta=0.05, n_iter=100, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        #self.cost_ = []

        for i in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            #cost = (-y.dot(np.log(output)) - ((1 - y).dot(np.log(1 - output))))
            #self.cost_.append(cost)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, z):
        return 1. / (1. + np.exp(-np.clip(z, -250, 250)))

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, 0)

    def predict_proba(self, X):
        return self.activation(self.net_input(X))


class MultiClassLogisticRegressionGD(object):
    def __init__(self, eta=0.05, n_iter=100, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.classifiers_ = []
        for c in self.classes_:
            y_binary = np.where(y == c, 1, 0)
            clf = LogisticRegressionGD(eta=self.eta, n_iter=self.n_iter, random_state=self.random_state)
            clf.fit(X, y_binary)
            self.classifiers_.append(clf)
        return self

    def predict_proba(self, X):
        # collect raw probabilities from each binary classifier
        raw = np.column_stack([clf.predict_proba(X) for clf in self.classifiers_])
        # softmax normalization
        exp_raw = np.exp(raw - raw.max(axis=1, keepdims=True))
        return exp_raw / exp_raw.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


def main():
    iris = datasets.load_iris()
    X = iris.data[:, [2, 3]]
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

    mclr = MultiClassLogisticRegressionGD(eta=0.05, n_iter=1000, random_state=1)
    mclr.fit(X_train, y_train)

    print('Predicted probabilities for first 5 test samples:')
    print(mclr.predict_proba(X_test[:5]))
    print('Predicted classes:', mclr.predict(X_test))
    print('True classes:     ', y_test)
    accuracy = np.mean(mclr.predict(X_test) == y_test)
    print(f'Test accuracy: {accuracy:.3f}')

    plot_decision_regions(X=X_train, y=y_train, classifier=mclr)
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.legend(loc='upper left')
    plt.title('Multi-class Logistic Regression (Softmax)')
    plt.show()


if __name__ == '__main__':
    main()
