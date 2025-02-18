# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: hydrogen
#       format_version: '1.3'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %%
import torch
import sys
import numpy as np
from torch import nn
sys.path.append("../")
from MLRF.datatools import h5kit
from scipy.signal import find_peaks


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.conv1 = nn.Conv1d(
            in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1
        )
        self.conv2 = nn.Conv1d(
            in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1
        )
        self.conv3 = nn.Conv1d(
            in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1
        )
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 256, 128)
        self.dropout = nn.Dropout(p=0.1)
        self.fc2 = nn.Linear(128, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)

        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))

        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        # x = self.dropout(x)
        x = self.fc2(x)

        return x


# %%
# Load the model

model = torch.load('model.pth', map_location=device)
model.to(device)
model.eval()

# Example data for inference
dataset = h5kit("../overair.h5")

keys = dataset.keys()
NOISE_THRESHOLD = -1000
PEAK_PROMINENCE = 2
# Perform inference
with torch.no_grad():
    wifi_count = 0
    bluetooth_count = 0

    for key in keys:
        data = dataset.read(key).astype(np.float32)
        peaks, properties = find_peaks(
        data, height=NOISE_THRESHOLD, prominence=PEAK_PROMINENCE
        )
        if len(peaks) > 0:
            data = torch.tensor(data).unsqueeze(0).unsqueeze(0).to(device)
            output = model(data)
            pred = torch.argmax(output).item()
            if pred == 0:
                wifi_count += 1
                decision = "wifi"
            else:
                bluetooth_count += 1
                decision = "bluetooth"

    print(f"Total wifi: {wifi_count}, Total bluetooth: {bluetooth_count}")
