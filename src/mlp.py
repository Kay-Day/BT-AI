import numpy as np

class MLP:
    def __init__(self, input_size=784, hidden1_size=128, hidden2_size=64, output_size=10):
        """
        Initialize the MLP with the specified architecture.
        Args:
            input_size (int): Size of input layer (784 for MNIST, as specified).
            hidden1_size (int): Number of neurons in the first hidden layer.
            hidden2_size (int): Number of neurons in the second hidden layer.
            output_size (int): Number of neurons in the output layer (10 for MNIST).
        """
        # Use Xavier initialization for weights
        np.random.seed(42)  # For reproducibility
        self.W1 = np.random.randn(input_size, hidden1_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((1, hidden1_size))
        self.W2 = np.random.randn(hidden1_size, hidden2_size) * np.sqrt(2.0 / hidden1_size)
        self.b2 = np.zeros((1, hidden2_size))
        self.W3 = np.random.randn(hidden2_size, output_size) * np.sqrt(2.0 / hidden2_size)
        self.b3 = np.zeros((1, output_size))

        # Initialize momentum terms
        self.momentum = 0.9
        self.vW1 = np.zeros_like(self.W1)
        self.vb1 = np.zeros_like(self.b1)
        self.vW2 = np.zeros_like(self.W2)
        self.vb2 = np.zeros_like(self.b2)
        self.vW3 = np.zeros_like(self.W3)
        self.vb3 = np.zeros_like(self.b3)

    def relu(self, x):
        """ReLU activation function, as specified for the first hidden layer."""
        return np.maximum(0, x)

    def relu_derivative(self, x):
        """Derivative of ReLU function for backpropagation."""
        return (x > 0).astype(float)

    def sigmoid(self, x):
        """Sigmoid activation function, as specified for the second hidden layer."""
        # Clip x to avoid overflow in exp
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        """Derivative of Sigmoid function for backpropagation."""
        s = self.sigmoid(x)
        return s * (1 - s)

    def softmax(self, x):
        """Softmax activation function, as specified for the output layer."""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # Subtract max for numerical stability
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def forward(self, X):
        """
        Forward propagation (lan truyền thuận), as required.
        Args:
            X (ndarray): Input data.
        Returns:
            Z1, A1, Z2, A2, Z3, A3: Intermediate and final outputs of each layer.
        """
        # Layer 1: Input -> Hidden 1 (ReLU)
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = self.relu(self.Z1)

        # Layer 2: Hidden 1 -> Hidden 2 (Sigmoid)
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = self.sigmoid(self.Z2)

        # Layer 3: Hidden 2 -> Output (Softmax)
        self.Z3 = np.dot(self.A2, self.W3) + self.b3
        self.A3 = self.softmax(self.Z3)

        return self.Z1, self.A1, self.Z2, self.A2, self.Z3, self.A3

    def backward(self, X, y, Z1, A1, Z2, A2, Z3, A3, learning_rate):
        """
        Backward propagation (lan truyền ngược), as required.
        Args:
            X (ndarray): Input data.
            y (ndarray): True labels (one-hot encoded).
            Z1, A1, Z2, A2, Z3, A3: Outputs from forward propagation.
            learning_rate (float): Learning rate for gradient descent.
        """
        m = X.shape[0]

        # Gradient for output layer (Softmax + Cross-Entropy)
        dZ3 = A3 - y  # Simplified gradient for cross-entropy loss with softmax
        dW3 = np.dot(A2.T, dZ3) / m
        db3 = np.sum(dZ3, axis=0, keepdims=True) / m

        # Gradient for hidden layer 2 (Sigmoid)
        dA2 = np.dot(dZ3, self.W3.T)
        dZ2 = dA2 * self.sigmoid_derivative(Z2)
        dW2 = np.dot(A1.T, dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m

        # Gradient for hidden layer 1 (ReLU)
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * self.relu_derivative(Z1)
        dW1 = np.dot(X.T, dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m

        # Update momentum terms
        self.vW1 = self.momentum * self.vW1 - learning_rate * dW1
        self.vb1 = self.momentum * self.vb1 - learning_rate * db1
        self.vW2 = self.momentum * self.vW2 - learning_rate * dW2
        self.vb2 = self.momentum * self.vb2 - learning_rate * db2
        self.vW3 = self.momentum * self.vW3 - learning_rate * dW3
        self.vb3 = self.momentum * self.vb3 - learning_rate * db3

        # Update weights and biases using momentum
        self.W1 += self.vW1
        self.b1 += self.vb1
        self.W2 += self.vW2
        self.b2 += self.vb2
        self.W3 += self.vW3
        self.b3 += self.vb3

    def train(self, X_train, y_train, X_test, y_test, epochs=20, batch_size=1000, learning_rate=0.05):
        """
        Train the MLP model using mini-batch gradient descent, as required (batch size: 1000).
        Args:
            X_train, y_train: Training data and labels.
            X_test, y_test: Test data and labels.
            epochs (int): Number of training epochs.
            batch_size (int): Size of mini-batches (1000, as specified).
            learning_rate (float): Learning rate for gradient descent.
        """
        for epoch in range(epochs):
            # Mini-batch training
            for i in range(0, len(X_train), batch_size):
                X_batch = X_train[i:i + batch_size]
                y_batch = y_train[i:i + batch_size]

                # Forward propagation
                Z1, A1, Z2, A2, Z3, A3 = self.forward(X_batch)

                # Backward propagation
                self.backward(X_batch, y_batch, Z1, A1, Z2, A2, Z3, A3, learning_rate)

            # Compute training accuracy
            _, _, _, _, _, A3_train = self.forward(X_train)
            train_acc = np.mean(np.argmax(A3_train, axis=1) == np.argmax(y_train, axis=1))
            print(f"Epoch {epoch + 1}/{epochs}, Training Accuracy: {train_acc:.4f}")

        # Compute test accuracy
        _, _, _, _, _, A3_test = self.forward(X_test)
        test_acc = np.mean(np.argmax(A3_test, axis=1) == np.argmax(y_test, axis=1))
        print(f"Final Test Accuracy: {test_acc:.4f}")