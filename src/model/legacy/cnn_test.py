class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        # Input shape: (16, 1, 2048)
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
        self.fc1 = nn.Linear(
            64 * 256, 128
        )  # 2048 -> 1024 -> 512 -> 256 after 3 pooling layers
        self.fc2 = nn.Linear(128, 2)
        self.relu = nn.ReLU()

    def forward(self, x):

        if len(x.shape) == 2:
            x = x.unsqueeze(1)

        # Convolutional layers with pooling
        x = self.pool(self.relu(self.conv1(x)))  # Output: (16, 16, 1024)
        x = self.pool(self.relu(self.conv2(x)))  # Output: (16, 32, 512)
        x = self.pool(self.relu(self.conv3(x)))  # Output: (16, 64, 256)

        # Flatten and fully connected layers
        x = self.flatten(x)  # Output: (16, 64 * 256)
        x = self.relu(self.fc1(x))  # Output: (16, 128)
        x = self.fc2(x)  # Output: (16, 2)

        return x
