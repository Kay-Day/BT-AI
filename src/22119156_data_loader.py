import numpy as np
from tensorflow.keras.datasets import mnist
from sklearn.preprocessing import LabelBinarizer

def load_mnist_data():
    """
    Load and preprocess the MNIST dataset using TensorFlow.
    Returns:
        X_train, X_test, y_train, y_test: Preprocessed training and test data.
    """
    print("Loading MNIST dataset using TensorFlow...")
    # Load MNIST dataset
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # Flatten the images (28x28 -> 784)
    X_train = X_train.reshape(-1, 28 * 28)
    X_test = X_test.reshape(-1, 28 * 28)

    # Normalize pixel values to [0, 1]
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    # Convert labels to one-hot encoding
    lb = LabelBinarizer()
    y_train = lb.fit_transform(y_train)
    y_test = lb.transform(y_test)

    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    return X_train, X_test, y_train, y_test