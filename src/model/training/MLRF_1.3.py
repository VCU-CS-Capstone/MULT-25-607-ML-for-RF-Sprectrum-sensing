import os
import time
import h5py
import torch
import numpy as np
from torch import nn
from tqdm import tqdm
from datetime import datetime


# ------------------- Terminal Colors and UI -------------------
class Colors:
    """ANSI terminal color codes"""

    # Text colors
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"

    # Text styles
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    RESET = "\033[0m"

    # Shorthand combinations
    HEADER = BOLD + BLUE
    SUCCESS = GREEN
    WARNING = YELLOW
    ERROR = RED
    INFO = CYAN
    RESULT = MAGENTA


def print_header(text, width=160):
    """Print a formatted header with consistent width"""
    print(f"\n{Colors.HEADER}{('=' * width)[:width]}")
    print(f"{text}")
    print(f"{('=' * width)[:width]}{Colors.RESET}")


def print_subheader(text, width=160):
    """Print a formatted subheader with consistent width"""
    print(f"\n{Colors.BOLD}{Colors.CYAN}{('-' * width)[:width]}")
    print(f"{text}")
    print(f"{('-' * width)[:width]}{Colors.RESET}")


def clean_memory():
    """Clean CUDA memory and Python garbage collection"""
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ------------------- Environment Setup -------------------
def setup_environment():
    """Configure training environment and return device"""
    # Enable optimized training
    torch.backends.cudnn.benchmark = True

    # Memory management settings for CUDA
    if torch.cuda.is_available():
        # Set to reduce memory fragmentation
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        # Print CUDA memory info
        print(
            f"{Colors.INFO}CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB total{Colors.RESET}"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Clear terminal screen
    os.system("cls" if os.name == "nt" else "clear")

    return device


# ------------------- Model Architecture -------------------
class ChannelAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // 4),
            nn.LeakyReLU(0.1),
            nn.Linear(channels // 4, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        attn = x.mean(-1)
        attn = self.fc(attn).unsqueeze(-1)
        return x * attn


class ShapeAwareRFClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        # Enhanced feature extraction
        self.conv_layers = nn.Sequential(
            nn.Conv1d(2, 64, 51, padding=25),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.MaxPool1d(4),
            nn.Conv1d(64, 128, 25, stride=2, padding=12),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Conv1d(128, 256, 15, stride=2, padding=7),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),
            ChannelAttention(256),
            nn.AdaptiveAvgPool1d(1),
        )

        self.fc = nn.Sequential(
            nn.Linear(256, 64), nn.LeakyReLU(0.1), nn.Dropout(0.5), nn.Linear(64, 1)
        )

        # Improved gradient operator
        self.diff_conv = nn.Conv1d(1, 1, 5, padding=2, bias=False)
        nn.init.constant_(self.diff_conv.weight[0, 0, 2], -1)
        nn.init.constant_(self.diff_conv.weight[0, 0, [1, 3]], 0.5)
        self.diff_conv.requires_grad_(False)

    def forward(self, x):
        # Input sanitation
        x = torch.nan_to_num(x, nan=-100.0, posinf=-100.0, neginf=-100.0)
        x = torch.clamp(x, min=-200, max=200)

        # Gradient computation
        x_grad = self.diff_conv(x)
        x_grad = torch.clamp(x_grad, -10, 10)

        # Normalization
        x = (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + 1e-4)
        x_grad = (x_grad - x_grad.mean(dim=-1, keepdim=True)) / (
            x_grad.std(dim=-1, keepdim=True) + 1e-4
        )

        # Feature combination
        x_combined = torch.cat([x, x_grad], dim=1)
        features = self.conv_layers(x_combined)
        return self.fc(features.squeeze(-1))


# ------------------- Data Handling -------------------
def load_data_to_gpu(file_path, device):
    """Load data with noise floor calibration"""
    print(f"\n{Colors.INFO}📊 Loading data from {file_path}...{Colors.RESET}")

    if not os.path.exists(file_path):
        print(f"{Colors.ERROR}Error: Data file not found: {file_path}{Colors.RESET}")
        raise FileNotFoundError(f"Data file not found: {file_path}")

    with h5py.File(file_path, "r") as f:
        keys = list(f.keys())
        sample_length = f[keys[0]].shape[0]

        data_array = np.zeros((len(keys), 1, sample_length), dtype=np.float32)
        labels = np.zeros(len(keys), dtype=np.float32)

        # Calculate noise floor
        min_db = np.inf
        for key in tqdm(
            keys,
            desc=f"{Colors.CYAN}Calculating noise floor{Colors.RESET}",
            ncols=160,
            bar_format="{l_bar}{bar:50}{r_bar}",
            leave=True,
        ):
            sample = f[key][:]
            valid = sample[sample > -np.inf]
            if valid.size > 0:
                min_db = min(min_db, valid.min())

        print(f"{Colors.INFO}🔊 Noise floor: {min_db:.2f} dB{Colors.RESET}")

        # Load data with calibrated replacement
        for i, key in enumerate(
            tqdm(
                keys,
                desc=f"{Colors.CYAN}Loading data{Colors.RESET}",
                ncols=160,
                bar_format="{l_bar}{bar:50}{r_bar}",
                leave=True,
            )
        ):
            sample = f[key][:]
            sample[sample == -np.inf] = min_db - 10  # 10dB below noise floor
            data_array[i, 0, :] = sample
            labels[i] = 0.0 if key.startswith("wifi") else 1.0

        data_tensor = torch.from_numpy(data_array).to(device)
        labels_tensor = torch.from_numpy(labels).to(device)

        # Data statistics
        print(
            f"{Colors.INFO}📈 Data range: {data_tensor.min().item():.2f} dB to {data_tensor.max().item():.2f} dB{Colors.RESET}"
        )
        wifi_count = (labels_tensor == 0).sum().item()
        bt_count = (labels_tensor == 1).sum().item()
        print(
            f"{Colors.INFO}📊 Class balance - WiFi: {wifi_count}, BT: {bt_count}{Colors.RESET}"
        )

    return data_tensor, labels_tensor


def augment_psd(batch_data):
    """Apply spectral augmentations to data"""
    # Frequency shifting
    shift = torch.randint(-5, 5, (1,)).item()
    batch_data = torch.roll(batch_data, shifts=shift, dims=2)

    # Additive noise
    batch_data += torch.randn_like(batch_data) * 1.0

    # Amplitude variation
    batch_data *= torch.empty(1, device=batch_data.device).uniform_(0.9, 1.1)

    return batch_data


# ------------------- Training Infrastructure -------------------
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            inputs, targets, reduction="none"
        )
        pt = torch.exp(-BCE_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return focal_loss.mean()


def train(model, data, labels, loss_fn, optimizer, scaler, batch_size):
    """Mixed-precision training with gradient clipping"""
    model.train()
    total_loss = 0
    indices = torch.randperm(data.size(0), device=data.device)

    with tqdm(
        total=data.size(0) // batch_size,
        ncols=160,
        desc=f"{Colors.YELLOW}Training{Colors.RESET}",
        bar_format="{l_bar}{bar:50}{r_bar}",
        leave=True,
    ) as pbar:
        for i in range(0, data.size(0), batch_size):
            batch_data = data[indices[i : i + batch_size]].clone()
            batch_labels = labels[indices[i : i + batch_size]].unsqueeze(1)

            # Apply augmentations
            batch_data = augment_psd(batch_data)

            # Mixed precision forward
            with torch.amp.autocast(device_type="cuda"):
                outputs = model(batch_data)
                loss = loss_fn(outputs, batch_labels)

            # Scaled backward
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Optimizer step
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            total_loss += loss.item() * batch_data.size(0)
            pbar.update(1)

    return total_loss / data.size(0)


def test(model, data, labels, loss_fn):
    """Validation with metrics calculation"""
    model.eval()
    total_loss, correct = 0, 0
    batch_size = 512  # Fixed batch size for evaluation

    with torch.no_grad(), tqdm(
        total=(data.size(0) + batch_size - 1) // batch_size,
        ncols=160,
        desc=f"{Colors.MAGENTA}Testing{Colors.RESET}",
        bar_format="{l_bar}{bar:50}{r_bar}",
        leave=True,
    ) as pbar:
        for i in range(0, data.size(0), batch_size):
            batch_data = data[i : i + batch_size]
            batch_labels = labels[i : i + batch_size].unsqueeze(1)

            outputs = model(batch_data)
            loss = loss_fn(outputs, batch_labels)

            total_loss += loss.item() * batch_data.size(0)
            correct += (torch.sigmoid(outputs) > 0.5).eq(batch_labels).sum().item()
            pbar.update(1)

    accuracy = 100 * correct / data.size(0)
    avg_loss = total_loss / data.size(0)
    return avg_loss, accuracy


# ------------------- Visualization -------------------


def visualize_results(
    model, test_data, test_labels, training_history=None, config=None
):
    """
    Generate comprehensive visualization of model performance metrics.
    Processes full dataset in batches to avoid memory issues.

    Args:
        model: Trained model
        test_data: Full test dataset tensor
        test_labels: Full test labels tensor
        training_history: Dictionary with training metrics (train_loss, val_loss, val_accuracy)
        config: Configuration dictionary
    """
    # Use Agg backend to avoid display issues in WSL
    import matplotlib

    matplotlib.use("Agg")  # Must be before importing pyplot
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
    import warnings
    import gc

    print(
        f"\n{Colors.INFO}Generating comprehensive performance visualization...{Colors.RESET}"
    )

    # Create timestamp for filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Set up plot style
    plt.style.use(
        "seaborn-v0_8" if "seaborn-v0_8" in plt.style.available else "default"
    )

    # Always create a 2x2 layout
    fig = plt.figure(figsize=(16, 14))
    gs = fig.add_gridspec(2, 2)
    ax1 = fig.add_subplot(gs[0, 0])  # Loss plot
    ax2 = fig.add_subplot(gs[0, 1])  # Accuracy plot
    ax3 = fig.add_subplot(gs[1, 0])  # Confusion matrix
    ax4 = fig.add_subplot(gs[1, 1])  # ROC curve

    # Process the full dataset in batches to generate predictions
    model.eval()
    batch_size = 64  # Small batch size to prevent OOM
    test_size = test_data.size(0)

    # Arrays to collect all predictions
    all_probs = []

    print(
        f"{Colors.INFO}Generating predictions on full test set ({test_size} samples)...{Colors.RESET}"
    )

    with torch.no_grad(), tqdm(
        total=(test_size + batch_size - 1) // batch_size,
        ncols=160,
        desc=f"{Colors.CYAN}Predicting{Colors.RESET}",
        bar_format="{l_bar}{bar:50}{r_bar}",
    ) as pbar:
        for i in range(0, test_size, batch_size):
            # Get batch
            end_idx = min(i + batch_size, test_size)
            batch_data = test_data[i:end_idx]

            # Forward pass and sigmoid
            outputs = model(batch_data)
            probs = torch.sigmoid(outputs).squeeze().cpu().numpy()

            # Handle scalar case (batch size 1)
            if probs.ndim == 0:
                probs = np.array([probs])

            # Store results
            all_probs.append(probs)

            # Update progress bar
            pbar.update(1)

            # Clear GPU cache periodically
            if i % (batch_size * 20) == 0:
                torch.cuda.empty_cache()

    # Concatenate all predictions
    probabilities = np.concatenate(all_probs)
    predictions = (probabilities > 0.5).astype(int)
    test_labels_np = test_labels.cpu().numpy()

    # Force garbage collection
    gc.collect()
    torch.cuda.empty_cache()

    print(f"{Colors.INFO}Generating plots with full dataset metrics...{Colors.RESET}")

    # Plot 1: Training and Validation Loss
    if training_history and len(training_history["train_loss"]) > 1:
        epochs = range(1, len(training_history["train_loss"]) + 1)

        # Loss plot
        ax1.plot(
            epochs,
            training_history["train_loss"],
            "b-",
            marker="o",
            label="Training Loss",
        )
        ax1.plot(
            epochs,
            training_history["val_loss"],
            "r-",
            marker="x",
            label="Validation Loss",
        )
        ax1.set_xlabel("Epochs")
        ax1.set_ylabel("Loss")
        ax1.set_title("Training and Validation Loss per Epoch")
        ax1.legend(loc="upper right")
        ax1.grid(True, alpha=0.3)

        # Add min loss markers
        min_train_idx = np.argmin(training_history["train_loss"])
        min_val_idx = np.argmin(training_history["val_loss"])

        ax1.plot(
            min_train_idx + 1,
            training_history["train_loss"][min_train_idx],
            "bo",
            markersize=10,
            fillstyle="none",
            markeredgewidth=2,
        )
        ax1.plot(
            min_val_idx + 1,
            training_history["val_loss"][min_val_idx],
            "ro",
            markersize=10,
            fillstyle="none",
            markeredgewidth=2,
        )

        # Add loss value annotations
        ax1.annotate(
            f"{training_history['train_loss'][min_train_idx]:.4f}",
            (min_train_idx + 1, training_history["train_loss"][min_train_idx]),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
        )
        ax1.annotate(
            f"{training_history['val_loss'][min_val_idx]:.4f}",
            (min_val_idx + 1, training_history["val_loss"][min_val_idx]),
            xytext=(0, -15),
            textcoords="offset points",
            ha="center",
        )

        # Accuracy plot
        ax2.plot(
            epochs,
            training_history["val_accuracy"],
            "g-",
            marker="o",
            label="Validation Accuracy",
        )
        ax2.set_xlabel("Epochs")
        ax2.set_ylabel("Accuracy (%)")
        ax2.set_title("Validation Accuracy per Epoch")

        # Add max accuracy marker
        max_acc_idx = np.argmax(training_history["val_accuracy"])
        ax2.plot(
            max_acc_idx + 1,
            training_history["val_accuracy"][max_acc_idx],
            "go",
            markersize=10,
            fillstyle="none",
            markeredgewidth=2,
        )

        # Add accuracy value annotation
        ax2.annotate(
            f"{training_history['val_accuracy'][max_acc_idx]:.2f}%",
            (max_acc_idx + 1, training_history["val_accuracy"][max_acc_idx]),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
        )

        # Draw horizontal line at max accuracy
        ax2.axhline(
            y=training_history["val_accuracy"][max_acc_idx],
            color="g",
            linestyle="--",
            alpha=0.5,
        )

        ax2.legend(loc="lower right")
        ax2.grid(True, alpha=0.3)
    else:
        # If no history, display placeholders
        ax1.text(0.5, 0.5, "No training history available", ha="center", va="center")
        ax2.text(0.5, 0.5, "No training history available", ha="center", va="center")

    # Check class distribution
    unique_classes = np.unique(test_labels_np)
    class_counts = {c: np.sum(test_labels_np == c) for c in [0, 1]}
    print(
        f"{Colors.INFO}Class distribution - WiFi: {class_counts.get(0, 0)}, BT: {class_counts.get(1, 0)}{Colors.RESET}"
    )

    # Plot 3: Confusion Matrix
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cm = confusion_matrix(test_labels_np, predictions, labels=[0, 1])

    im = ax3.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax3.set_title("Confusion Matrix")
    tick_marks = np.arange(2)
    ax3.set_xticks(tick_marks)
    ax3.set_yticks(tick_marks)
    ax3.set_xticklabels(["WiFi", "Bluetooth"])
    ax3.set_yticklabels(["WiFi", "Bluetooth"])

    # Add text annotations to confusion matrix
    thresh = cm.max() / 2.0 if cm.size > 0 and cm.max() > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax3.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    # Add colorbar for confusion matrix
    cbar = fig.colorbar(im, ax=ax3)
    cbar.set_label("Count")

    # Calculate metrics
    accuracy = (
        100 * np.sum(predictions == test_labels_np) / len(test_labels_np)
        if len(test_labels_np) > 0
        else 0
    )
    sensitivity = cm[1, 1] / (cm[1, 0] + cm[1, 1]) if (cm[1, 0] + cm[1, 1]) > 0 else 0
    specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) > 0 else 0

    # Add metrics text
    metrics_text = f"Accuracy: {accuracy:.2f}%\nSensitivity: {sensitivity:.2f}\nSpecificity: {specificity:.2f}"
    ax3.set_xlabel(f"Predicted\n{metrics_text}")
    ax3.set_ylabel("True")

    # Plot 4: ROC curve - handle case where only one class is present
    if len(unique_classes) > 1:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fpr, tpr, _ = roc_curve(test_labels_np, probabilities)
            roc_auc = auc(fpr, tpr)

        ax4.plot(fpr, tpr, "b-", linewidth=2, label=f"AUC = {roc_auc:.3f}")
        ax4.plot([0, 1], [0, 1], "k--", linewidth=1)  # Random guessing line

        # Add points for specific thresholds
        thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
        threshold_indices = []
        for t in thresholds:
            pred_t = (probabilities >= t).astype(int)
            tn, fp, fn, tp = confusion_matrix(
                test_labels_np, pred_t, labels=[0, 1]
            ).ravel()
            tpr_t = tp / (tp + fn) if (tp + fn) > 0 else 0
            fpr_t = fp / (fp + tn) if (fp + tn) > 0 else 0
            # Find closest point on ROC curve
            idx = np.argmin(np.abs(fpr - fpr_t) + np.abs(tpr - tpr_t))
            threshold_indices.append(idx)
            ax4.plot(fpr[idx], tpr[idx], "ro", markersize=6)
            ax4.annotate(
                f"{t}", (fpr[idx], tpr[idx]), xytext=(5, 0), textcoords="offset points"
            )
    else:
        print(
            f"{Colors.WARNING}Warning: Only one class found in test set. ROC curve not available.{Colors.RESET}"
        )
        ax4.text(
            0.5,
            0.5,
            "ROC unavailable - only one class in test set",
            ha="center",
            va="center",
            fontsize=12,
        )
        roc_auc = float("nan")

    ax4.set_xlim([0.0, 1.0])
    ax4.set_ylim([0.0, 1.05])
    ax4.set_xlabel("False Positive Rate")
    ax4.set_ylabel("True Positive Rate")
    ax4.set_title("ROC Curve")
    if len(unique_classes) > 1:
        ax4.legend(loc="lower right")
    ax4.grid(True, alpha=0.3)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report = classification_report(
            test_labels_np,
            predictions,
            target_names=["WiFi", "Bluetooth"],
            output_dict=True,
        )

    # Print classification report - Fixed to use numeric indices instead of class names
    print(f"\n{Colors.INFO}Classification Report:{Colors.RESET}")
    print(
        f"{Colors.CYAN}{'':15s} {'Precision':10s} {'Recall':10s} {'F1-score':10s} {'Support':10s}{Colors.RESET}"
    )

    # Use numeric keys instead of class names
    class_names = ["WiFi", "Bluetooth"]
    for i, cls in enumerate(class_names):
        if str(i) in report:  # Use string keys for numeric classes
            print(
                f"{cls:15s} {report[str(i)]['precision']:10.2f} "
                f"{report[str(i)]['recall']:10.2f} "
                f"{report[str(i)]['f1-score']:10.2f} "
                f"{report[str(i)]['support']:10d}"
            )
        else:
            print(f"{cls:15s} {'N/A':10s} {'N/A':10s} {'N/A':10s} {0:10d}")

    print(
        f"{'Accuracy':15s} {'':<10s} {'':<10s} {report['accuracy']:10.2f} {sum(class_counts.values()):10d}"
    )
    # Set overall title with metadata
    title = "RF Signal Classifier Performance Analysis"
    if config:
        # Format subtitle with config info
        subtitle = f"LR={config['lr']}, Batch Size={config['batch_size']}, Epochs={config['epochs']}"
        fig.suptitle(f"{title}\n{subtitle}", fontsize=16)
    else:
        fig.suptitle(title, fontsize=16)

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for the suptitle

    # Save the figure with timestamp
    filename = f"results_MLRF_1.3_{timestamp}.png"
    print(f"{Colors.INFO}Saving visualization to {filename}...{Colors.RESET}")
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    print(
        f"\n{Colors.SUCCESS}📊 Results visualization saved to: {filename}{Colors.RESET}"
    )

    # Close the figure to free memory
    plt.close(fig)

    return filename


# ------------------- Training Loop -------------------
def run_training_and_testing(config):
    """Main training function with improved interface"""
    # Setup
    device = setup_environment()
    scaler = torch.amp.GradScaler()
    # Display header
    print_header(
        f"{Colors.BOLD}🚀 RF Signal Classifier Training - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    )
    print(f"{Colors.INFO}💻 Using device: {device}{Colors.RESET}")

    # Initialize model
    model = ShapeAwareRFClassifier().to(device)

    # Initialize training history tracking
    history = {"train_loss": [], "val_loss": [], "val_accuracy": []}

    # Model verification
    print(f"\n{Colors.INFO}🔍 Verifying model architecture...{Colors.RESET}")
    test_input = torch.randn(16, 1, 2048, device=device)
    _ = model(test_input)
    print(f"{Colors.SUCCESS}✅ Model forward check passed{Colors.RESET}")

    # Load data
    train_data, train_labels = load_data_to_gpu(config["train_data"], device)
    test_data, test_labels = load_data_to_gpu(config["test_data"], device)

    # Setup training components
    pos_weight = (train_labels == 0).sum() / (train_labels == 1).sum()
    print(
        f"{Colors.INFO}⚖️ Using positive weight in loss function: {pos_weight:.2f}{Colors.RESET}"
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
        betas=(0.95, 0.999),
    )

    scheduler = torch.optim.lr_scheduler.CyclicLR(
        optimizer,
        base_lr=config["lr"] / 10,
        max_lr=config["lr"],
        step_size_up=500,
        cycle_momentum=False,
    )

    loss_fn = FocalLoss(alpha=0.75, gamma=2.0)

    # Display training configuration
    print_subheader("📋 Training Configuration:")
    print(f"{Colors.CYAN}• Epochs: {config['epochs']}")
    print(f"• Batch Size: {config['batch_size']}")
    print(f"• Learning Rate: {config['lr']}")
    print(f"• Weight Decay: {config['weight_decay']}")
    print(f"• Model Save Path: {config['model_path']}{Colors.RESET}")

    # Confirm training start
    try:
        confirmation = (
            input(f"\n{Colors.YELLOW}🔥 Start training? (y/n): {Colors.RESET}")
            .lower()
            .strip()
        )
        if confirmation != "y":
            print(f"{Colors.WARNING}Training cancelled.{Colors.RESET}")
            return
    except KeyboardInterrupt:
        print(f"\n{Colors.WARNING}Training cancelled by user.{Colors.RESET}")
        return

    # Training loop
    best_acc = 0
    start_time = time.time()

    print_header(f"{Colors.BOLD}📈 Training Progress")

    try:
        for epoch in range(config["epochs"]):
            epoch_start = time.time()

            # Learning rate warmup
            if epoch < 3:
                for param_group in optimizer.param_groups:
                    param_group["lr"] = config["lr"] * (epoch + 1) / 3

            # Training epoch
            train_loss = train(
                model,
                train_data,
                train_labels,
                loss_fn,
                optimizer,
                scaler,
                config["batch_size"],
            )

            # Evaluation
            test_loss, test_acc = test(
                model,
                test_data,
                test_labels,
                loss_fn,
            )

            # Record metrics in history
            history["train_loss"].append(train_loss)
            history["val_loss"].append(test_loss)
            history["val_accuracy"].append(test_acc)

            # Update scheduler
            scheduler.step()

            # Save best model
            is_best = test_acc > best_acc
            if is_best:
                best_acc = test_acc
                torch.save(
                    model.state_dict(),
                    config["model_path"],
                    _use_new_zipfile_serialization=True,
                )
                saved_indicator = f"{Colors.SUCCESS}📝 [Saved]{Colors.RESET}"
            else:
                saved_indicator = ""

            # Calculate epoch time
            epoch_time = time.time() - epoch_start

            # Progress reporting
            print(
                f"{Colors.BOLD}Epoch {epoch+1:02d}/{config['epochs']} | {Colors.RESET}"
                f"{Colors.YELLOW}Train Loss: {train_loss:.4f} | {Colors.RESET}"
                f"{Colors.MAGENTA}Test Loss: {test_loss:.4f} | {Colors.RESET}"
                f"{Colors.GREEN}Acc: {test_acc:.2f}% | {Colors.RESET}"
                f"{Colors.CYAN}LR: {optimizer.param_groups[0]['lr']:.1e} | {Colors.RESET}"
                f"{Colors.BLUE}Time: {epoch_time:.1f}s{Colors.RESET} {saved_indicator}"
            )

    except KeyboardInterrupt:
        print(f"\n\n{Colors.WARNING}⚠️ Training interrupted by user{Colors.RESET}")

    # Final report
    total_time = time.time() - start_time
    minutes, seconds = divmod(total_time, 60)
    hours, minutes = divmod(minutes, 60)

    print_header(f"{Colors.BOLD}🏁 Training Summary")
    print(
        f"{Colors.CYAN}• Total duration: {int(hours)}h {int(minutes)}m {seconds:.2f}s"
    )
    print(f"{Colors.GREEN}• Best accuracy: {best_acc:.2f}%")
    print(f"{Colors.BLUE}• Model saved to: {config['model_path']}{Colors.RESET}")

    # Generate visualization of results
    print(f"\n{Colors.INFO}🔍 Generating performance visualization...{Colors.RESET}")
    if os.path.exists(config["model_path"]):
        # Load the best model for visualization
        model.load_state_dict(torch.load(config["model_path"], weights_only=True))

    visualize_results(model, test_data, test_labels, history, config)


# ------------------- Main Entry Point -------------------
if __name__ == "__main__":
    config = {
        "batch_size": 256,
        "epochs": 10,
        "lr": 3e-5,
        "weight_decay": 1e-5,
        "model_path": "MLRF_1.3.pth",
        "train_data": "train.h5",
        "test_data": "test.h5",
    }

    # Import matplotlib only when needed to avoid unnecessary dependencies
    try:
        import matplotlib
        import sklearn
    except ImportError:
        print(f"{Colors.WARNING}Warning: Visualization dependencies not installed.")
        print(
            "To enable visualization, install: matplotlib, scikit-learn{Colors.RESET}"
        )

    # Run training process
    run_training_and_testing(config)
