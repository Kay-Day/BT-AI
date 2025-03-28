from src.data_loader import load_mnist_data
from src.mlp import MLP

def train_mlp():
    """
    Train the MLP model on the MNIST dataset with the specified requirements.
    """
    # Load data
    X_train, X_test, y_train, y_test = load_mnist_data()

    # Initialize MLP with the specified architecture
    mlp = MLP(input_size=784, hidden1_size=128, hidden2_size=64, output_size=10)

    # Train the model with mini-batch gradient descent (batch size: 1000)
    mlp.train(
        X_train, y_train, X_test, y_test,
        epochs=20,  # Increased to 20
        batch_size=1000,  # As specified in the requirements
        learning_rate=0.05  # Updated to match the new learning rate
    )