import numpy as np
import torch
from torch import nn
from scipy import signal
from torch.utils.data import Dataset, DataLoader
import sys
import os
import json
import matplotlib.pyplot as plt


def interleaved_to_complex(source_data):
    """
    Convert interleaved IQ data to complex numbers
    """
    source_data = np.array(source_data)
    source_data = source_data / 32768
    source_data = source_data.astype(np.float32).view(np.complex64)
    return source_data


def iq_to_psd(iq_data, num_fft, sample_rate):
    """
    # Compute the PSD of the input IQ data
    """
    freq, psd = signal.welch(iq_data, fs=sample_rate, nperseg=num_fft, return_onesided=False)
    return psd


def read_labels(file_path):
    """
    Reads a CSV file with two columns (x, y) and returns a dictionary with x as the key and y as the value
    """
    labels = {}
    with open(file_path, "r") as f:
        for line in f:
            label = line.strip().split(",")
            label[0] = int(label[0])
            labels[label[0]] = {"wifi": 0, "bluetooth": 1}.get(label[1], -1)
    return labels


def read_json(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    return data


class psd_dataset(Dataset):
    def __init__(self, data_directory, labels, config):
        self.labels = read_labels(labels)
        self.data_directory = data_directory
        self.config = read_json(config)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        data_path = os.path.join(self.data_directory, f"data_{idx}.npy")
        data = np.load(data_path)
        data = interleaved_to_complex(data)
        data = iq_to_psd(data, self.config["ffts"], self.config["sample_rate"])
        data = torch.from_numpy(data)
        label = self.labels[idx]
        return data, label


training_dataset = psd_dataset(
    "./data/training_data", "./data/training_data/labels.csv", "./config.json"
)
testing_dataset = psd_dataset(
    "./data/testing_data", "./data/testing_data/labels.csv", "./config.json"
)
training_dataloader = DataLoader(
    training_dataset, batch_size=1, shuffle=False, num_workers=1
)
testing_dataloader = DataLoader(
    testing_dataset, batch_size=1, shuffle=False, num_workers=1
)
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
print(f"Using {device} device")

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (sample, label) in enumerate(dataloader):
        sample, label = sample.to(device), label.to(device)

        # Compute prediction and loss
        pred = model(sample)
        loss = loss_fn(pred, label)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(sample)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0

    with torch.no_grad():
        for sample, label in dataloader:
            sample, label = sample.to(device), label.to(device)
            pred = model(sample)
            test_loss += loss_fn(pred, label).item()
            correct += (pred.argmax(1) == label).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(
        f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
    )


# Training loop
epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(training_dataloader, model, loss_fn, optimizer)
    test(testing_dataloader, model, loss_fn)
print("Done!")
torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")
