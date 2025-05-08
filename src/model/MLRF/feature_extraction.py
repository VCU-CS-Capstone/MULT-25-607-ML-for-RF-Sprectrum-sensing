
import numpy as np
import time
import multiprocessing
from tqdm import tqdm
from .utils import get_chunk_indices
from .feature_extraction_chunks import (
    extract_spectral_shape_features_chunk,
    extract_bandwidth_features_chunk,
)


def extract_spectral_shape_features(X):
    """Extract features based on spectral shape characteristics using multiprocessing"""
    wifi_width = 410
    bt_width = 20
    num_samples = X.shape[0]
    num_processes = max(1, multiprocessing.cpu_count() - 1)
    print(f"Using {num_processes} processes for spectral shape features")

    chunk_indices = get_chunk_indices(num_samples, num_processes)
    chunk_args = [
        (X[start:end], start, end, wifi_width, bt_width)
        for start, end in chunk_indices
    ]

    start_time = time.time()
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = list(
            tqdm(
                pool.imap(extract_spectral_shape_features_chunk, chunk_args),
                total=len(chunk_args),
                desc="Extracting spectral shape features",
                ncols=100,
            )
        )

    all_features = [item for sublist in results for item in sublist]
    print(
        f"Spectral shape feature extraction completed in {time.time() - start_time:.2f} seconds"
    )
    return np.array(all_features)


def extract_segmented_bandwidth_features(X):
    """Extract features from segments focused on bandwidth characteristics using multiprocessing"""
    wifi_width = 410
    bt_width = 20
    num_samples = X.shape[0]
    num_processes = max(1, multiprocessing.cpu_count() - 1)
    print(f"Using {num_processes} processes for bandwidth features")

    chunk_indices = get_chunk_indices(num_samples, num_processes)
    chunk_args = [
        (X[start:end], start, end, wifi_width, bt_width)
        for start, end in chunk_indices
    ]

    start_time = time.time()
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = list(
            tqdm(
                pool.imap(extract_bandwidth_features_chunk, chunk_args),
                total=len(chunk_args),
                desc="Extracting bandwidth features",
                ncols=100,
            )
        )

    all_features = [item for sublist in results for item in sublist]
    print(
        f"Bandwidth feature extraction completed in {time.time() - start_time:.2f} seconds"
    )
    return np.array(all_features)


def extract_features(X, feature_method="combined"):
    """Extract features from PSD data using different methods with multiprocessing"""
    print(f"Extracting features using method: {feature_method}")

    if feature_method == "spectral_shape":
        return extract_spectral_shape_features(X)
    elif feature_method == "bandwidth":
        return extract_segmented_bandwidth_features(X)
    elif feature_method == "combined":
        shape_features = extract_spectral_shape_features(X)
        bandwidth_features = extract_segmented_bandwidth_features(X)
        return np.hstack((shape_features, bandwidth_features))
    else:
        print(
            f"Unknown feature method: {feature_method}. Using combined features."
        )
        shape_features = extract_spectral_shape_features(X)
        bandwidth_features = extract_segmented_bandwidth_features(X)
        return np.hstack((shape_features, bandwidth_features))
