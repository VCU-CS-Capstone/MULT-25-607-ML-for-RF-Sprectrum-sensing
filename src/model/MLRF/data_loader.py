
import os
import h5py
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from .preprocessing import preprocess_with_event_detection


def load_and_split_data(file_path, test_size=0.2, random_state=42):
    """Load data from h5 file and split into train/test sets with equal class distribution"""
    print(f"Loading data from {file_path}...")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")

    with h5py.File(file_path, "r") as f:
        keys = list(f.keys())
        sample_length = f[keys[0]].shape[0]

        # Initialize arrays
        data_array = np.zeros((len(keys), sample_length), dtype=np.float32)
        labels = np.zeros(len(keys), dtype=np.float32)

        # Calculate noise floor
        min_db = np.inf
        for key in tqdm(keys, desc="Calculating noise floor"):
            sample = f[key][:]
            valid = sample[sample > -np.inf]
            if valid.size > 0:
                min_db = min(min_db, valid.min())

        print(f"Noise floor: {min_db:.2f} dB")

        # Load data with noise floor replacement
        for i, key in enumerate(tqdm(keys, desc="Loading data")):
            sample = f[key][:]
            # Replace -inf values with 10dB below noise floor
            sample[sample == -np.inf] = min_db - 10
            data_array[i, :] = sample
            labels[i] = 0.0 if key.startswith("wifi") else 1.0

        # Data statistics
        print(
            f"Data range: {data_array.min():.2f} dB to {data_array.max():.2f} dB"
        )
        wifi_count = np.sum(labels == 0)
        bt_count = np.sum(labels == 1)
        print(f"Class balance - WiFi: {wifi_count}, BT: {bt_count}")

    # Process data with event detection
    print("Preprocessing data with event detection...")
    data_array, event_widths = preprocess_with_event_detection(
        data_array, min_db - 10
    )

    # Split data while preserving class distribution
    X_train, X_test, y_train, y_test = train_test_split(
        data_array,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,  # This ensures equal distribution of classes
    )

    # Also split event widths accordingly
    train_indices_full = np.arange(len(data_array))
    # We need to split event_widths based on the indices from the data_array split.
    # The previous split of train_indices was incorrect.
    # Instead, we can get the indices from the train_test_split directly if we split indices.

    # To get corresponding event_widths for X_train and X_test:
    # We need to ensure the split is consistent.
    # One way is to split indices first, then use these indices to get X, y, and event_widths.

    indices = np.arange(len(data_array))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )

    X_train = data_array[train_idx]
    X_test = data_array[test_idx]
    y_train = labels[train_idx]
    y_test = labels[test_idx]

    event_widths_np = np.array(event_widths)
    train_event_widths = event_widths_np[train_idx].tolist()
    test_event_widths = event_widths_np[test_idx].tolist()

    print(f"Train set size: {X_train.shape[0]} samples")
    print(f"Test set size: {X_test.shape[0]} samples")

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        train_event_widths,
        test_event_widths,
        min_db,
    )
