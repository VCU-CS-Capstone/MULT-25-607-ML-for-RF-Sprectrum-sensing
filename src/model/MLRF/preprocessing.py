
import numpy as np
from .utils import event_detector


def preprocess_with_event_detection(X, noise_floor=-190.0):
    """Preprocess data with event detection to isolate signal regions"""
    X_processed = X.copy()
    event_widths = []  # Store the widths of detected events for each sample

    for i in range(X.shape[0]):
        # Detect events in the signal
        events = event_detector(X[i])

        # Create a mask with noise floor
        signal_mask = np.ones(X.shape[1]) * noise_floor

        # Fill the mask with actual signal at event locations
        total_width = 0
        for start, end in events:
            if start != 0 or end != 0:  # Skip if no event detected
                signal_mask[start : end + 1] = X[i, start : end + 1]
                total_width += end - start + 1

        # Store the width of the detected event (in bins)
        event_widths.append(total_width)

        # Replace the original signal with the masked version
        X_processed[i] = signal_mask

    return X_processed, event_widths


def estimate_signal_width(signal, noise_floor):
    """Estimate the width of a signal in bins"""
    # Find regions where signal is above noise floor by a threshold
    threshold = noise_floor + 5  # 5 dB above noise floor
    signal_mask = signal > threshold

    # Find the start and end indices of contiguous regions
    changes = np.diff(np.hstack(([0], signal_mask.astype(int), [0])))
    start_indices = np.where(changes == 1)[0]
    end_indices = np.where(changes == -1)[0] - 1

    if len(start_indices) == 0:
        return 0

    # Find the widest region
    widths = end_indices - start_indices + 1
    max_width_idx = np.argmax(widths)

    return widths[max_width_idx]


def adjust_predictions_by_width(
    X, y_pred_proba, event_widths, noise_floor, width_threshold=200
):
    """
    Adjust model predictions based on estimated signal width
    width_threshold: ~10 MHz in bins (assuming 2048 bins over 100 MHz)
    """
    # Convert from probability to class prediction
    y_pred = (y_pred_proba > 0.5).astype(int)
    y_adjusted = y_pred.copy()

    # WiFi channel width ~20MHz = ~410 bins
    # Bluetooth width ~1MHz = ~20 bins
    # We use 200 bins (~10 MHz) as the decision boundary

    for i in range(len(y_pred)):
        width = event_widths[i]

        # Calculate model confidence
        confidence = max(y_pred_proba[i], 1 - y_pred_proba[i])

        # Only override if confidence is not very high
        if confidence < 0.85:
            if width < width_threshold:
                # Narrow signal - more likely Bluetooth (class 1)
                y_adjusted[i] = 1
            else:
                # Wide signal - more likely WiFi (class 0)
                y_adjusted[i] = 0

    # Count the number of adjustments made
    num_adjustments = np.sum(y_adjusted != y_pred)

    return y_adjusted, num_adjustments
