import os
import json
from torch import nn
import torch.nn.functional as F
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from torch.utils.data import Dataset, DataLoader
from app.models.neural_network import NeuralNetwork
from app.utils.data_processing import read_labels

# Define dataset class
class PSDDataset(Dataset):
    def __init__(self, data_directory, labels_file):
        self.labels = read_labels(labels_file)
        self.data_directory = data_directory

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        data_path = os.path.join(self.data_directory, f"data_{idx}.npy")
        data = torch.from_numpy(np.load(data_path)).float()
        label = self.labels.get(idx, -1)
        return data, label

def train_model():
    # Paths
    training_data_dir = "./data/training_data_psd"
    training_labels_path = "./data/training_data_psd/labels.csv"
    testing_data_dir = "./data/testing_data_psd"
    testing_labels_path = "./data/testing_data_psd/labels.csv"
    model_save_path = "./models/model.pth"

    # Hyperparameters
    batch_size = 32
    learning_rate = 1e-3
    epochs = 50
    dropout_rate = 0.3

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")

    # Datasets and DataLoaders
    training_dataset = PSDDataset(training_data_dir, training_labels_path)
    testing_dataset = PSDDataset(testing_data_dir, testing_labels_path)
    training_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    testing_dataloader = DataLoader(testing_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Initialize model, loss function, and optimizer
    model = NeuralNetwork(dropout_rate=dropout_rate).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Lists to store training history
    train_losses = []
    test_losses = []
    accuracies = []

    # Training loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch, (data, labels) in enumerate(training_dataloader):
            data, labels = data.to(device), labels.to(device)

            # Forward pass
            outputs = model(data)
            loss = loss_fn(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(training_dataloader)
        train_losses.append(avg_train_loss)

        # Evaluation
        model.eval()
        total, correct = 0, 0
        test_loss = 0.0
        with torch.no_grad():
            for data, labels in testing_dataloader:
                data, labels = data.to(device), labels.to(device)
                outputs = model(data)
                loss = loss_fn(outputs, labels)
                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_test_loss = test_loss / len(testing_dataloader)
        test_losses.append(avg_test_loss)
        accuracy = 100 * correct / total
        accuracies.append(accuracy)

        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}, Accuracy: {accuracy:.2f}%")

    print("Training complete!")

    # Save the model
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

    # Plot training history
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(test_losses, label="Test Loss")
    plt.title("Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(accuracies, label="Accuracy")
    plt.title("Accuracy over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    train_model()