import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator
from abc import ABC, abstractmethod
from typing import Union, Dict, Any

# --- 1. MnistClassifierInterface (Abstract Interface) ---
class MnistClassifierInterface(ABC):
    """
    Interface that all MNIST classifiers must implement.
    It ensures a unified structure for the train and predict methods.
    """
    @abstractmethod
    def train(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> None:
        """Trains the model on the provided data."""
        pass

    @abstractmethod
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Performs prediction on the given test data."""
        pass


# --- 2. Implementations of Models That Implement the Interface ---

# Helper class for PyTorch models
class BasePyTorchClassifier(MnistClassifierInterface):
    """Base class for NN and CNN with shared PyTorch logic."""
    def __init__(self, model: nn.Module):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def _prepare_data(self, X: np.ndarray, y: np.ndarray = None, is_training: bool = False):
        """Converts NumPy arrays into PyTorch tensors."""
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        
        # For NN and CNN, MNIST 28x28 pixels (784 features) must be shaped as (N, C, H, W) for CNN or (N, D) for NN
        if isinstance(self.model, CNN):
            # Reshape X for CNN: (N, 784) -> (N, 1, 28, 28)
            X_tensor = X_tensor.view(-1, 1, 28, 28)
        elif isinstance(self.model, FNN):
            # Reshape X for NN: (N, 784) -> (N, 784)
            X_tensor = X_tensor.view(-1, 28*28)

        if y is not None:
            y_tensor = torch.tensor(y, dtype=torch.long).to(self.device)
            return X_tensor, y_tensor
        return X_tensor

    def train(self, X_train: np.ndarray, y_train: np.ndarray, epochs: int = 5, batch_size: int = 64) -> None:
        """Train the PyTorch model."""
        self.model.train()
        X_tensor, y_tensor = self._prepare_data(X_train, y_train)
        
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        for epoch in range(epochs):
            for inputs, labels in dataloader:
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
        print(f"PyTorch model trained on {self.device}.")

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Perform prediction with the PyTorch model."""
        self.model.eval()
        X_tensor = self._prepare_data(X_test)
        with torch.no_grad():
            outputs = self.model(X_tensor)
            _, predicted = torch.max(outputs.data, 1)
        return predicted.cpu().numpy()


# 2.1. Feed-Forward Neural Network (FNN)
class FNN(nn.Module):
    """Architecture of a Fully Connected Neural Network."""
    def __init__(self, input_size: int = 784, num_classes: int = 10):
        super(FNN, self).__init__()
        self.layer_1 = nn.Linear(input_size, 128)
        self.relu = nn.ReLU()
        self.layer_2 = nn.Linear(128, 64)
        self.layer_3 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = x.view(-1, 28*28)  # Flatten the image into a vector
        x = self.relu(self.layer_1(x))
        x = self.relu(self.layer_2(x))
        x = self.layer_3(x)
        return x


class NNClassifier(BasePyTorchClassifier):
    """FNN Classifier implementing the interface."""
    def __init__(self):
        super().__init__(FNN())


# 2.2. Convolutional Neural Network (CNN)
class CNN(nn.Module):
    """Architecture of a Convolutional Neural Network."""
    def __init__(self, num_classes: int = 10):
        super(CNN, self).__init__()
        self.layer_1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer_2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(7 * 7 * 32, num_classes)  # 7x7 after two MaxPool2d layers

    def forward(self, x):
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = x.reshape(x.size(0), -1)  # Flatten before fully connected layer
        x = self.fc(x)
        return x


class CNNClassifier(BasePyTorchClassifier):
    """CNN Classifier implementing the interface."""
    def __init__(self):
        super().__init__(CNN())


# 2.3. Random Forest Classifier
class RFClassifier(MnistClassifierInterface):
    """Random Forest Classifier implementing the interface."""
    def __init__(self):
        # Uses Scikit-learn
        self.model: Union[RandomForestClassifier, None] = None

    def _prepare_data(self, X: np.ndarray):
        """Reshape X for Random Forest: (N, 28, 28) -> (N, 784)."""
        if X.ndim > 2:
            return X.reshape(X.shape[0], -1)
        return X

    def train(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> None:
        """Train the Random Forest model."""
        X_flat = self._prepare_data(X_train)
        # Configure model; allow passing n_estimators through kwargs
        n_estimators = kwargs.get('n_estimators', 100)
        self.model = RandomForestClassifier(n_estimators=n_estimators, random_state=42, n_jobs=-1)
        self.model.fit(X_flat, y_train)
        print(f"Random Forest model trained with n_estimators={n_estimators}.")

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Perform prediction with the Random Forest model."""
        if self.model is None:
            raise RuntimeError("Model has not been trained yet.")
        X_flat = self._prepare_data(X_test)
        return self.model.predict(X_flat)


# --- 3. MnistClassifier (Facade Class) ---
class MnistClassifier:
    """
    Facade class that hides the specific model implementation.
    Implements the Strategy Pattern.
    """
    ALGORITHMS = {
        'rf': RFClassifier,
        'nn': NNClassifier,
        'cnn': CNNClassifier,
    }

    def __init__(self, algorithm: str):
        """
        Initializes a classifier based on the specified algorithm.
        :param algorithm: 'rf', 'nn', or 'cnn'.
        """
        if algorithm not in self.ALGORITHMS:
            raise ValueError(f"Unknown algorithm '{algorithm}'. Must be one of: {list(self.ALGORITHMS.keys())}")
        
        # Create a specific classifier object (Strategy)
        self.classifier: MnistClassifierInterface = self.ALGORITHMS[algorithm]()
        print(f"Initialized MnistClassifier with {algorithm} algorithm.")

    def train(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> None:
        """
        Unified training method. Calls train() of the internal classifier.
        """
        self.classifier.train(X_train, y_train, **kwargs)

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Unified prediction method. Calls predict() of the internal classifier.
        """
        return self.classifier.predict(X_test)

    def get_algorithm_name(self) -> str:
        """Returns the name of the active algorithm."""
        return type(self.classifier).__name__
