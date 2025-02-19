import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------------------------------------------------
# Helper Function: Ensure Input Has a Channel Dimension
# -------------------------------------------------------------------
def ensure_channel_dim(x: torch.Tensor) -> torch.Tensor:
    """
    If x is of shape (batch, length), unsqueeze to (batch, 1, length).
    Assumes length is 2048.
    """
    if x.dim() == 2:
        return x.unsqueeze(1)
    return x


# -------------------------------------------------------------------
# Model 1: BasicCNN with Batch Normalization and Dropout
# -------------------------------------------------------------------


class BasicCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Convolutional layers with BatchNorm
        self.conv1 = nn.Conv1d(1, 32, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 1)  # Single output for binary classification
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Split into two segments
        x1, x2 = torch.split(x, x.size(2) // 2, dim=2)

        # Process first segment
        x1 = self.pool(F.relu(self.bn1(self.conv1(x1))))
        x1 = self.pool(F.relu(self.bn2(self.conv2(x1))))
        x1 = self.pool(F.relu(self.bn3(self.conv3(x1))))
        x1 = self.adaptive_pool(x1)
        x1 = x1.view(x1.size(0), -1)
        x1 = F.relu(self.fc1(x1))
        x1 = self.dropout(x1)
        x1 = self.fc2(x1)  # Raw logits for binary classification

        # Process second segment
        x2 = self.pool(F.relu(self.bn1(self.conv1(x2))))
        x2 = self.pool(F.relu(self.bn2(self.conv2(x2))))
        x2 = self.pool(F.relu(self.bn3(self.conv3(x2))))
        x2 = self.adaptive_pool(x2)
        x2 = x2.view(x2.size(0), -1)
        x2 = F.relu(self.fc1(x2))
        x2 = self.dropout(x2)
        x2 = self.fc2(x2)  # Raw logits for binary classification

        # Combine predictions by selecting the segment with the highest confidence
        confidence1 = torch.sigmoid(x1).squeeze()  # Probability of being Bluetooth
        confidence2 = torch.sigmoid(x2).squeeze()  # Probability of being Bluetooth

        # Final decision based on the highest confidence
        combined_output = torch.where(confidence1 > confidence2, x1, x2)
        return combined_output


# -------------------------------------------------------------------
# Model 2: DSCNN Using Depthwise Separable Convolutions
# -------------------------------------------------------------------
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        self.depthwise = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size,
            padding=kernel_size // 2,
            groups=in_channels,
        )
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        return self.pointwise(x)


class DSCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = DepthwiseSeparableConv(1, 16, kernel_size=7)
        self.conv2 = DepthwiseSeparableConv(16, 32, kernel_size=7)
        self.conv3 = DepthwiseSeparableConv(32, 64, kernel_size=7)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = ensure_channel_dim(x)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.adaptive_pool(x).squeeze(-1)
        return self.fc(x)


# -------------------------------------------------------------------
# Weight Initialization Function
# -------------------------------------------------------------------
def init_weights(m):
    """
    Initialize convolutional and linear layers using Kaiming normal initialization.
    """
    if isinstance(m, (nn.Conv1d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)


#
