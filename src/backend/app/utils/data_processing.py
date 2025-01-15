import numpy as np

def read_labels(file_path):
    """
    Reads a CSV file with two columns (x, y) and returns a dictionary with x as the key and y as the value
    """
    labels = {}
    with open(file_path, "r") as f:
        for line in f:
            label = line.strip().split(",")
            if len(label) != 2:
                continue  # Skip malformed lines
            try:
                idx = int(label[0])
                lbl = label[1].strip().lower()
                labels[idx] = {"wifi": 0, "bluetooth": 1}.get(lbl, -1)
            except ValueError:
                continue  # Skip lines with invalid integers
    return labels

def normalize_psd(psd):
    """
    Normalizes the PSD data to a range of [0, 1].

    Parameters:
    - psd (numpy.ndarray): The Power Spectral Density data.

    Returns:
    - numpy.ndarray: The normalized PSD data.
    """
    min_val = np.min(psd)
    max_val = np.max(psd)
    if max_val - min_val == 0:
        return np.zeros_like(psd)
    normalized_psd = (psd - min_val) / (max_val - min_val)
    return normalized_psd