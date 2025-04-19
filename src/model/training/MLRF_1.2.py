import multiprocessing
import os
import time

import h5py
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm, trange

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SpectrumClassifier(nn.Module):
    """CNN Architecture for Full Spectrum Analysis."""

    def __init__(self):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            # First layer looks for 2.5 MHz; (100/2048) ≈ 48.8 kHz resolution
            nn.Conv1d(1, 64, kernel_size=51, padding=25),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(64, 128, kernel_size=25, padding=12),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(128, 256, kernel_size=11, padding=5),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.5), nn.Linear(128, 1)
        )
        self._init_weights()

    def _init_weights(self):
        """Initialize weights for Conv1d and Linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


class FullPSDDataset(Dataset):
    """Dataset class that loads and normalizes PSD data from an HDF5 file, normalizing each item individually."""

    def __init__(self, file_path):
        self.file_path = file_path
        with h5py.File(file_path, "r") as f:
            self.keys = list(f.keys())

        # preload all data into gpu memory after finding normalizing constats
        self.data = []
        self.labels = []
        with h5py.File(file_path, "r") as f:
            for key in tqdm(
                self.keys, desc=f"Loading data to {device}", colour="green", ncols=160
            ):
                # load data
                data = (
                    torch.tensor(f[key][:], dtype=torch.float32).unsqueeze(0).to(device)
                )

                # Calculate min and max for this key
                # psd_min = torch.min(data)  # Use torch.min for single tensor
                # psd_max = torch.max(data)  # Use torch.max for single tensor
                psd_min = -200
                psd_max = 0

                # Handle -inf min (if present)
                # if psd_min == float("-inf"):
                #     # Replace -inf with the next smallest value
                #     finite_vals = data[
                #         torch.isfinite(data)
                #     ]  # Filter out infinite values
                #     if len(finite_vals) > 0:  # Check if there are any finite values
                #         psd_min = torch.min(
                #             finite_vals
                #         )  # Find the minimum of the finite values
                #     else:
                #         psd_min = (
                #             -180
                #         )  # Or some other reasonable default.  Crucially, HANDLE THE CASE WHERE ALL ARE -INF.
                #         psd_max = (
                #             -20
                #         )  # if all -inf then it is likely that the data is corrupt, so set max to a low value
                #         print(
                #             f"WARNING: Key '{key}' contains only -inf values. Using default psd_min and psd_max. CHECK DATA"
                #         )

                # normalize data
                data = (data - psd_min) / (psd_max - psd_min)
                data = torch.clamp(data, 0, 1)

                # add to data
                self.data.append(data)

                # label assignment: 0.0 if key starts with "wifi", otherwise 1.0
                label = 0.0 if key.startswith("wifi") else 1.0
                self.labels.append(torch.tensor(label, device=device))

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def train(model, loader, loss_fn, optimizer):
    """Train the model for one epoch."""
    model.train()
    total_loss = 0
    scaler = torch.GradScaler()

    for samples, labels in tqdm(
        loader, desc="Training", leave=False, colour="green", ncols=160
    ):
        optimizer.zero_grad()

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            outputs = model(samples)
            loss = loss_fn(outputs, labels.unsqueeze(1))

        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * samples.size(0)

    return total_loss / len(loader.dataset)


def test(model, loader, loss_fn):
    """Evaluate the model on the test set."""
    model.eval()
    total_loss, correct = 0, 0
    with torch.no_grad():
        for samples, labels in tqdm(
            loader, desc="Testing", leave=False, colour="green", ncols=160
        ):
            outputs = model(samples)
            loss = loss_fn(outputs, labels.unsqueeze(1))
            total_loss += loss.item() * samples.size(0)
            preds = torch.sigmoid(outputs) > 0.5
            correct += preds.eq(labels.unsqueeze(1)).sum().item()

    accuracy = 100 * correct / len(loader.dataset)
    return total_loss / len(loader.dataset), accuracy


def run_training_and_testing(config):
    """Run the training and testing loops, and plot the results."""
    # Clear the terminal screen
    os.system("cls" if os.name == "nt" else "clear")

    # Prepare data loaders
    train_loader = DataLoader(
        FullPSDDataset("train.h5"),
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
    )
    test_loader = DataLoader(
        FullPSDDataset("test.h5"),
        batch_size=config["batch_size"] * 2,
        num_workers=config["num_workers"],
    )

    # Initialize model, optimizer, scheduler, and loss function
    model = SpectrumClassifier().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config["lr"] * 10,
        steps_per_epoch=len(train_loader),
        epochs=config["epochs"],
    )
    loss_fn = nn.BCEWithLogitsLoss()

    train_losses = []
    test_accuracies = []
    start_time = time.time()

    # Epoch training loop with blue progress bar
    for epoch in trange(
        config["epochs"], desc="Training Epochs", unit="epoch", colour="blue", ncols=160
    ):
        train_loss = train(model, train_loader, loss_fn, optimizer)
        test_loss, test_acc = test(model, test_loader, loss_fn)
        scheduler.step()

        torch.save(model, config["model_path"])

        train_losses.append(train_loss)
        test_accuracies.append(test_acc)
        tqdm.write(
            f"Epoch {epoch + 1}/{config['epochs']}: "
            f"Train Loss: {train_loss:.4f}, "
            f"Test Acc: {test_acc:.1f}%,"
            f"Test Loss: {test_loss:.4f},"
        )

    total_time = time.time() - start_time
    tqdm.write(
        f"Training completed in {total_time:.2f} seconds over {config['epochs']} epochs."
    )

    return train_losses, test_accuracies


if __name__ == "__main__":
    # Set start method to 'spawn' to avoid CUDA initialization issues
    multiprocessing.set_start_method("spawn", force=True)

    config = {
        "batch_size": 4096,
        "epochs": 8,
        "lr": 2e-4,
        "weight_decay": 1e-4,
        "model_path": "MLRF_1.3.pth",
        "num_workers": 0,  # Disable multiprocessing
    }

    run_training_and_testing(config)
