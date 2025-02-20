import torch
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import h5py
import multiprocessing

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set the start method to 'spawn' to avoid CUDA initialization issues
if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)


# CNN Architecture for Full Spectrum Analysis
class SpectrumClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = nn.Sequential(
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
    def __init__(self, file_path):
        self.file_path = file_path
        with h5py.File(file_path, "r") as f:
            self.keys = list(f.keys())

        # Normalization parameters
        self.psd_min = -80
        self.psd_max = -20

        # Preload all data into GPU memory
        self.data = []
        self.labels = []
        with h5py.File(file_path, "r") as f:
            for key in self.keys:
                # Load, normalize, and move to GPU
                data = torch.tensor(f[key][:], dtype=torch.float32).unsqueeze(0)
                data = (data - self.psd_min) / (self.psd_max - self.psd_min)
                data = torch.clamp(data, 0, 1).to(device)

                self.data.append(data)
                self.labels.append(
                    torch.tensor(0.0 if key.startswith("wifi") else 1.0, device=device)
                )

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


# Training/Testing Functions (No data movement needed)
def train(model, loader, loss_fn, optimizer):
    model.train()
    total_loss = 0
    scaler = torch.GradScaler()

    for samples, labels in tqdm(loader, desc="Training"):
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
    model.eval()
    total_loss, correct = 0, 0
    with torch.no_grad():
        for samples, labels in tqdm(loader, desc="Testing"):
            outputs = model(samples)
            loss = loss_fn(outputs, labels.unsqueeze(1))
            total_loss += loss.item() * samples.size(0)

            preds = torch.sigmoid(outputs) > 0.5
            correct += preds.eq(labels.unsqueeze(1)).sum().item()

    return (total_loss / len(loader.dataset)), (100 * correct / len(loader.dataset))


# Main Execution
if __name__ == "__main__":
    config = {
        "batch_size": 4096,
        "epochs": 8,
        "lr": 2e-4,
        "weight_decay": 1e-4,
        "model_path": "full_spectrum_classifier.pth",
        "num_workers": 0,  # Disable multiprocessing
    }

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

    best_acc = 0
    train_losses = []
    test_accuracies = []

    for epoch in range(config["epochs"]):
        train_loss = train(model, train_loader, loss_fn, optimizer)
        test_loss, test_acc = test(model, test_loader, loss_fn)
        scheduler.step()

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), config["model_path"])
            print(f"New best model saved with accuracy {test_acc:.2f}%")

        train_losses.append(train_loss)
        test_accuracies.append(test_acc)

        print(
            f"Epoch {epoch+1}/{config['epochs']}: "
            f"Train Loss: {train_loss:.4f}, "
            f"Test Acc: {test_acc:.1f}%"
        )

    plt.style.use("dark_background")
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)

    plt.subplot(1, 2, 2)
    plt.plot(test_accuracies, label="Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.yticks(range(0, 101, 10))
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)

    plt.tight_layout()
    plt.savefig("test.png")
