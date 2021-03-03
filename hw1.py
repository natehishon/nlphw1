import numpy as np
import sklearn.model_selection as model_selection
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt

mnist = fetch_openml('mnist_784', version=1)
encoder = OneHotEncoder(sparse=False)

class FC_NN(object):
    def __init__(self, layers, learning_rate):
        self.num_layers = len(layers)
        self.layers = layers
        self.epochs = 100
        self.mini_batch = 30
        self.learning_rate = learning_rate
        self.random = np.random.RandomState(1)

        self.weights = {}
        self.biases = {}
        self.lineplot = []
        for i in range(1, self.num_layers, 1):
            self.weights[i] = self.random.normal(size=(self.layers[i - 1], self.layers[i]))
            self.biases[i] = self.random.normal(size=(self.layers[i]))

    def forward_propagate(self, X):
        output = {}
        for i in range(1, self.num_layers, 1):
            if i == 1:
                z = np.dot(X, self.weights[i]) + self.biases[i]
                a = self.sigmoid(z)
            else:
                z = np.dot(a, self.weights[i]) + self.biases[i]
                a = self.sigmoid(z)
            output[i] = {
                'a': a,
                'z': z
            }

        return output

    def calc_cost(self, y, a_output):
        step1 = a_output - y
        step2 = step1 ** 2
        twenty = 1 / 20
        cost1 = np.sum(step2) * twenty
        return cost1

    def back_prop(self, y, X, ff_output):
        for i in range(self.num_layers - 1, 0, -1):
            if i == self.num_layers - 1:
                delta = ff_output[i]['a'] - y
                self.weights[i] -= np.dot(ff_output[i - 1]['a'].T, delta) * self.learning_rate
                self.biases[i] -= np.sum(delta) * self.learning_rate

            elif i == 1:
                sig_deriv = ff_output[i]['a'] * (1 - ff_output[i]['a'])
                delta = (np.dot(delta, self.weights[i + 1].T) * sig_deriv)
                self.weights[i] -= np.dot(X.T, delta) * self.learning_rate
                self.biases[i] -= np.sum(delta) * self.learning_rate

            else:
                sig_deriv = ff_output[i]['a'] * (1 - ff_output[i]['a'])
                delta = (np.dot(delta, self.weights[i + 1].T) * sig_deriv)
                self.weights[i] -= np.dot(ff_output[i - 1]['a'].T, delta) * self.learning_rate
                self.biases[i] -= np.sum(delta) * self.learning_rate

    def fit(self, X_train, y_train, X_test, y_test):
        yt_encoded = encoder.fit_transform(y_train.reshape(-1, 1))
        for i in range(self.epochs):
            index = np.arange(X_train.shape[0])
            self.random.shuffle(index)
            for batch_index in range(0, index.shape[0], self.mini_batch):
                batch = index[batch_index:batch_index + self.mini_batch]
                output = self.forward_propagate(X_train[batch])
                self.back_prop(yt_encoded[batch], X_train[batch], output)
            self.evaluate_accuracy(X_train, y_train, X_test, y_test, yt_encoded, i)
        return self

    def evaluate_accuracy(self, X_train, y_train, X_test, y_test, y_encoded, i):
        output = self.forward_propagate(X_train)
        cost = self.calc_cost(y_encoded, output[self.num_layers - 1]['a'])
        y_train_predict = self.max(X_train)
        y_test_predict = self.max(X_test)
        training = ((np.sum(y_train == y_train_predict)) / X_train.shape[0])
        testing = ((np.sum(y_test == y_test_predict)) / X_test.shape[0])
        self.lineplot.append(cost)

        print("Epoch: {} Cost: {} \n Training Acc: {} Testing Acc: {}".format(i + 1, cost, training, testing))

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def sigmoidPrime(self, z):
        return np.exp(-z) / ((1 + np.exp(-z)) ** 2)

    def max(self, X):
        output = self.forward_propagate(X)
        return np.argmax(output[self.num_layers - 1]['a'], axis=1)

    def get_plot(self):
        return self.lineplot


X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
y = y.astype(int)
X = ((X / 255.) - .5) * 2

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=20000, random_state=17)


# q-3
nn = FC_NN([784, 16, 10], .01)
nn.fit(X_train, y_train, X_test, y_test)

# q-4
nn = FC_NN([784, 16, 16, 10], .001)
nn.fit(X_train, y_train, X_test, y_test)
lineOne = nn.get_plot()

nn = FC_NN([784, 16, 16, 10], .01)
nn.fit(X_train, y_train, X_test, y_test)
lineTwo = nn.get_plot()

nn = FC_NN([784, 16, 16, 10], .1)
nn.fit(X_train, y_train, X_test, y_test)
lineThree = nn.get_plot()

fig, axs = plt.subplots(3)
fig.suptitle('Cost function per epoch')
axs[0].plot(lineOne)
axs[1].plot(lineTwo)
axs[2].plot(lineThree)

plt.show()
