import torch.nn as nn
import torch.nn.functional as F

class NeuralNetwork(nn.Module):
    def __init__(self, dropout_rate=0.3):
        super(NeuralNetwork, self).__init__()
        
        # First convolutional block
        self.conv1 = nn.Conv1d(1, 8, kernel_size=8, stride=1, padding=3)
        self.bn1 = nn.BatchNorm1d(8)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        # Second convolutional block
        self.conv2 = nn.Conv1d(8, 16, kernel_size=4, stride=1, padding=2)
        self.bn2 = nn.BatchNorm1d(16)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        # Third convolutional block
        self.conv3 = nn.Conv1d(16, 32, kernel_size=2, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(32)
        self.dropout3 = nn.Dropout(dropout_rate)
        
        # Pooling layers
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.pool4 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.pool5 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        
        # Flatten layer
        self.flatten = nn.Flatten()
        
        # Fully connected layers
        self.fc1 = nn.Linear(32 * 256, 64)  # 32 channels * 256 length = 8192
        self.bn4 = nn.BatchNorm1d(64)
        self.dropout4 = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool(x)
        
        # Additional pooling layers
        x = self.pool4(x)
        x = self.pool5(x)
        
        x = self.dropout3(x)
        
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.bn4(x)
        x = F.relu(x)
        
        x = self.fc2(x)
        return x
