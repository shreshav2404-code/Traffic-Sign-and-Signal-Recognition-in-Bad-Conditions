"""A tiny MLP with forward pass and backpropagation written by hand in NumPy."""
import numpy as np


def relu(x):
    return np.maximum(x, 0)


def softmax(logits):
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def cross_entropy(probabilities, labels):
    return float(-np.mean(np.log(probabilities[np.arange(len(labels)), labels] + 1e-12)))


class NumpyMLP:
    """One hidden layer: inputs -> hidden (ReLU) -> outputs (softmax)."""

    def __init__(self, inputs, hidden, outputs, rng):
        self.W1 = rng.normal(0, np.sqrt(2 / inputs), (inputs, hidden))
        self.b1 = np.zeros(hidden)
        self.W2 = rng.normal(0, np.sqrt(1 / hidden), (hidden, outputs))
        self.b2 = np.zeros(outputs)

    def forward(self, x):
        """Return the hidden activations and class probabilities."""
        hidden = relu(x @ self.W1 + self.b1)
        return hidden, softmax(hidden @ self.W2 + self.b2)

    def loss(self, x, labels):
        return cross_entropy(self.forward(x)[1], labels)

    def backward(self, x, labels):
        """Apply the chain rule from the loss back to each weight."""
        hidden, probabilities = self.forward(x)
        count = len(labels)
        one_hot = np.zeros_like(probabilities)
        one_hot[np.arange(count), labels] = 1

        # dL/dz = p - y, where z are the outputs before softmax.
        output_error = probabilities - one_hot
        dW2 = hidden.T @ output_error / count
        db2 = output_error.mean(axis=0)

        # Push the error through W2, then keep only neurons where ReLU was active.
        hidden_error = (output_error @ self.W2.T) * (hidden > 0)
        dW1 = x.T @ hidden_error / count
        db1 = hidden_error.mean(axis=0)
        return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}

    def update(self, gradients, learning_rate):
        """weight = weight - learning_rate * gradient"""
        for name, gradient in gradients.items():
            getattr(self, name).__isub__(learning_rate * gradient)

    def numerical_gradient(self, x, labels, name, position, step=1e-5):
        """Estimate one gradient by nudging one weight up and down."""
        weights = getattr(self, name)
        original = weights[position]
        weights[position] = original + step
        above = self.loss(x, labels)
        weights[position] = original - step
        below = self.loss(x, labels)
        weights[position] = original
        return (above - below) / (2 * step)
