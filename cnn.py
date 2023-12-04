from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.figure import Figure
import random
from dataset import customDataset
import torchvision.transforms as transform
from torch.utils.data import DataLoader, random_split

class PyTorchCNN(nn.Module):
    """
    A PyTorch convolutional neural network model.

    Attributes:
        num_classes (int): The number of classes.
        model (nn.Sequential): The model.
    """

    def __init__(self, num_classes: int = 2, random_seed: int = 42) -> None:
        """
        Initialize the PyTorchCNN model.

        Parameters:
            num_classes (int): The number of classes.
            random_seed (int): The random seed.

        Returns:
            None

        Examples:
            >>> pytorch_cnn = PyTorchCNN()
            >>> pytorch_cnn.num_classes
            10
            >>> pytorch_cnn.model
            Sequential...
        """
        
        super(PyTorchCNN, self).__init__()

        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        random.seed(random_seed)
        
        self.num_classes = num_classes

        
        image_size = (256, 256)
        self.model = nn.Sequential(
           nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(num_features=32),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(num_features=64),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(num_features=128),
            nn.Dropout(p=0.25),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(num_features=256),
            nn.Dropout(p=0.25),

            nn.Flatten(),
            nn.Linear(in_features=256 * (256 // (2**4)) * (256 // (2**4)), out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=num_classes)
        )
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the neural network.

        Parameters:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The output of the neural network.
        """
        output = self.model(x)
        return output
    
    def train(self,
              train_loader: DataLoader,
              val_loader: DataLoader,
              epochs: int = 10,
              learning_rate: float = 0.001) -> None:
        """
        Train the neural network.

        Parameters:
            train_loader (DataLoader): The training data loader.
            val_loader (DataLoader): The validation data loader.
            epochs (int): The number of epochs to train for.
            learning_rate (float): The learning rate.

        Returns:
            None
        """

        # Define the loss function and optimizer
        loss_function = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        train_losses = []
        train_accs = []
        val_losses = []
        val_accs = []

        for epoch in range(epochs):
            self.model.train()
            tl = 0.0
            tc = 0.0
            for X_train, y_train in train_loader:
                X_train, y_train = X_train.to(device), y_train.to(device)

                optimizer.zero_grad()

                output = self.forward(X_train)

                loss = loss_function(output, y_train)
                loss.backward()

                optimizer.step()
                tl += loss.item()
                tc += self.accuracy(X_train, y_train)
            train_loss = tl/len(train_loader)
            train_losses.append(tl/len(train_loader))
            train_accs.append(tc/len(train_loader))

            self.model.eval()
            with torch.no_grad():
                vl = 0
                va = 0
                for X_val, y_val in val_loader:
                    X_val, y_val = X_val.to(device), y_val.to(device)
                    vl += loss_function(self.forward(X_val), y_val).item()
                    va += self.accuracy(X_val, y_val)
                val_losses.append(vl / len(val_loader))
                val_accs.append(va / len(val_loader))

            if (epoch + 1) % 1 == 0:
                print(f"[{epoch+1}/{epochs}] | Train Loss: {train_loss:.5f} | Train Accuracy: {train_accs[-1]:.5f} | Val Loss: {val_losses[-1]:.5f} | Val Accuracy: {val_accs[-1]:.5f}")

        self.train_val_metrics = {
            "train_losses": train_losses,
            "train_accs": train_accs,
            "val_losses": val_losses,
            "val_accs": val_accs
        }

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """
        Predict the class labels for the input data.
        
        Parameters:
            X (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The predicted class labels.

        Examples:
            >>> pytorch_cnn = PyTorchCNN()
            >>> X = torch.randn(10, 1, 28, 28)
            >>> pytorch_cnn.predict(X)
            tensor([5, 5, 5, 5, 5, 5, 5, 5, 5, 5])
        """

        # Set the model to eval mode and use torch.no_grad()
        self.model.eval()
        with torch.no_grad():
            return self.forward(X).argmax(dim=1)

    def plot_train_val_metrics(self) -> Tuple[Figure, np.ndarray]:
        """
        Plot the training and validation metrics.

        Parameters:
            None

        Returns:
            Tuple[Figure, np.ndarray]: A tuple containing the matplotlib
                Figure and Axes objects.
        """

        fig, ax = plt.subplots(1, 2, figsize=(12, 6))

        # Plot the training and validation losses
        ax[0].plot(self.train_val_metrics["train_losses"], label="Train Loss")
        ax[0].plot(self.train_val_metrics["val_losses"], label="Val Loss")

        ax[0].set_xlabel("Epoch")
        ax[0].set_ylabel("Loss")
        ax[0].legend()

        # Plot the training and validation accuracies
        ax[1].plot(self.train_val_metrics["train_accs"], label="Train Accuracy")
        ax[1].plot(self.train_val_metrics["val_accs"], label="Val Accuracy")

        ax[1].set_xlabel("Epoch")
        ax[1].set_ylabel("Accuracy")
        ax[1].legend()

        fig.suptitle("Train/Val Metrics")
        fig.tight_layout()

        plt.savefig("pytorch_cnn_train_val_metrics.png")

        return fig, ax

    def accuracy(self, X: torch.Tensor, y: torch.Tensor) -> float:
        """
        Calculate the accuracy of the neural network on the input data.

        Parameters:
            X (torch.Tensor): The input data.
            y (torch.Tensor): The true class labels.

        Returns:
            float: The accuracy of the neural network.
        """    
        assert X.shape[0] == y.shape[0], f"X.shape[0] != y.shape[0] ({X.shape[0]} != {y.shape[0]})"
        correct = torch.sum(self.predict(X) == y)
        return (correct / X.shape[0]).item()

if __name__=="__main__":
    from torch.utils.data import DataLoader, random_split
    from sklearn.model_selection import train_test_split
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    # process the data
    # Define a transformation that resizes images to a specific size
    image_size = (256, 256)  # Adjust this to the desired size
    data_transform = transform.Compose([
        transform.ToPILImage(),  # Convert numpy array to PIL Image
        transform.Grayscale(num_output_channels=3),
        transform.Resize(image_size),
        transform.ToTensor(),
        transform.Normalize(mean=[0.5], std=[0.5])
    ])
    data = customDataset('art_real.csv', transform=data_transform)
    # split data into traning and testing
    train_size = int(0.8 * len(data))
    val_size = len(data) - train_size
    train_dataset, val_dataset = random_split(data, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)

    # Initialize and move model to device
    pytorch_cnn = PyTorchCNN(num_classes=2).to(device)

    # Training
    pytorch_cnn.train(train_loader, val_loader, epochs=10, learning_rate=0.002)

    # Evaluate on test set
    
    test_accuracy = 0.0
    with torch.no_grad():
        for X_val, y_val in val_loader:
            X_val, y_val = X_val.to(device), y_val.to(device)
            test_accuracy += pytorch_cnn.accuracy(X_val, y_val)

    test_accuracy /= len(val_loader)
    print(f"Test Accuracy: {test_accuracy}")
    pytorch_cnn.plot_train_val_metrics()
    # saving the model

    checkpoint = {
    'model_state_dict': pytorch_cnn.state_dict(),
    }

    # Specify the path where you want to save the model checkpoint
    checkpoint_path = 'model1.pth'

    # Save the checkpoint
    torch.save(checkpoint, checkpoint_path)
    
