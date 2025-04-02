import numpy as np


class ThreeLayerNN:
    def __init__(self, input_size, hidden_size, output_size, activation):
        self.w1 = np.random.randn(
            input_size, hidden_size) * np.sqrt(2/input_size)
        self.w2 = np.random.randn(
            hidden_size, hidden_size) * np.sqrt(2/hidden_size)
        self.w3 = np.random.randn(
            hidden_size, output_size) * np.sqrt(2/hidden_size)

        self.b1 = np.zeros((1, hidden_size))
        self.b2 = np.zeros((1, hidden_size))
        self.b3 = np.zeros((1, output_size))

        self.activation = activation
        if activation == 'relu':
            self.activation_fn = lambda x: np.maximum(0, x)
            self.activation_deriv = lambda x: (x > 0).astype(float)
        elif activation == 'sigmoid':
            self.activation_fn = lambda x: 1 / (1 + np.exp(-x))
            self.activation_deriv = lambda x: self.activation_fn(
                x) * (1 - self.activation_fn(x))
        elif activation == 'tanh':
            self.activation_fn = lambda x: np.tanh(x)
            self.activation_deriv = lambda x: 1 - np.tanh(x)**2
        elif activation == "leaky_relu":
            self.activation_fn = lambda x: np.where(x > 0, x, 0.01*x)
            self.activation_deriv = lambda x: np.where(x > 0, 1, 0.01)

    def forward(self, x):
        self.z1 = np.dot(x, self.w1) + self.b1
        self.a1 = self.activation_fn(self.z1)

        self.z2 = np.dot(self.a1, self.w2) + self.b2
        self.a2 = self.activation_fn(self.z2)

        self.z3 = np.dot(self.a2, self.w3) + self.b3
        scores = np.exp(self.z3 - np.max(self.z3, axis=1, keepdims=True))
        self.probs = scores / np.sum(scores, axis=1, keepdims=True)

        return self.probs

    def backward(self, x, y, reg_lambda):
        n = x.shape[0]
        d3 = self.probs
        d3[range(n), y] -= 1
        d3 /= n

        dw3 = np.dot(self.a2.T, d3) + reg_lambda * self.w3
        db3 = np.sum(d3, axis=0, keepdims=True)
        d2 = np.dot(d3, self.w3.T) * self.activation_deriv(self.z2)

        dw2 = np.dot(self.a1.T, d2) + reg_lambda * self.w2
        db2 = np.sum(d2, axis=0, keepdims=True)
        d1 = np.dot(d2, self.w2.T) * self.activation_deriv(self.z1)

        dw1 = np.dot(x.T, d1) + reg_lambda * self.w1
        db1 = np.sum(d1, axis=0, keepdims=True)

        return dw1, db1, dw2, db2, dw3, db3

    def loss_fun(self, x, y, reg_lambda):
        n = x.shape[0]
        probs = self.forward(x)
        n_log_probs = -np.log(probs[range(n), y])
        loss = np.sum(n_log_probs) / n + 0.5 * reg_lambda * (np.sum(self.w1 **
                                                                    2) + np.sum(self.w2**2) + np.sum(self.w3**2))

        return loss

    def predict(self, x):
        probs = self.forward(x)
        return np.argmax(probs, axis=1)

    def accuracy(self, x, y):
        preds = self.predict(x)
        return np.mean(preds == y)
