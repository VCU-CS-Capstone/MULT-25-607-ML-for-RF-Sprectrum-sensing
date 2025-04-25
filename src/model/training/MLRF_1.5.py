import os
import time
import h5py
import numpy as np
import json
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
from scipy import signal as scipy_signal
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
import lightgbm as lgbm
import joblib
import warnings
import multiprocessing
warnings.filterwarnings('ignore')


# ------------------- Event Detection -------------------
def event_detector(PSD, threshold_factor=1.05**-1):
    """
    Detect events in PSD data where signal exceeds a threshold relative to the signal mean
    """
    window_size = 64
    event_ranges = []
    in_event = False
    start_idx = 0

    # Calculate mean of the entire PSD
    signal_mean = np.mean(PSD)

    # Set threshold relative to the mean
    dynamic_threshold = signal_mean * threshold_factor

    i = 0
    while i < len(PSD) - window_size + 1:
        window = PSD[i : i + window_size]
        window_average = round(sum(window) / window_size, 2)

        if window_average > dynamic_threshold and not in_event:
            start_idx = i
            in_event = True
            i += window_size
        elif window_average <= dynamic_threshold and in_event:
            event_ranges.append((start_idx, i - 1))
            in_event = False
            i += window_size
        else:
            i += window_size

    if in_event:
        event_ranges.append((start_idx, len(PSD) - 1))
    elif not event_ranges:  # Only add (0,0) if no events were found
        event_ranges.append((0, 0))

    return event_ranges


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
                signal_mask[start:end+1] = X[i, start:end+1]
                total_width += (end - start + 1)
        
        # Store the width of the detected event (in bins)
        event_widths.append(total_width)
        
        # Replace the original signal with the masked version
        X_processed[i] = signal_mask
    
    return X_processed, event_widths


# ------------------- Data Handling -------------------
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
        print(f"Data range: {data_array.min():.2f} dB to {data_array.max():.2f} dB")
        wifi_count = np.sum(labels == 0)
        bt_count = np.sum(labels == 1)
        print(f"Class balance - WiFi: {wifi_count}, BT: {bt_count}")
    
    # Process data with event detection
    print("Preprocessing data with event detection...")
    data_array, event_widths = preprocess_with_event_detection(data_array, min_db - 10)
    
    # Split data while preserving class distribution
    X_train, X_test, y_train, y_test = train_test_split(
        data_array, labels, 
        test_size=test_size,
        random_state=random_state,
        stratify=labels  # This ensures equal distribution of classes
    )
    
    # Also split event widths accordingly
    train_indices = np.arange(len(data_array))
    _, _, train_indices, test_indices = train_test_split(
        data_array, train_indices,
        test_size=test_size,
        random_state=random_state,
        stratify=labels
    )
    
    train_event_widths = [event_widths[i] for i in train_indices]
    test_event_widths = [event_widths[i] for i in test_indices]
    
    print(f"Train set size: {X_train.shape[0]} samples")
    print(f"Test set size: {X_test.shape[0]} samples")
    
    return X_train, X_test, y_train, y_test, train_event_widths, test_event_widths, min_db


# ------------------- Multiprocessing Helper Functions -------------------
def get_chunk_indices(total_size, num_chunks):
    """Split data into roughly equal chunks for multiprocessing"""
    chunk_size = total_size // num_chunks
    indices = []
    for i in range(num_chunks):
        start = i * chunk_size
        end = (i + 1) * chunk_size if i < num_chunks - 1 else total_size
        indices.append((start, end))
    return indices


def extract_spectral_shape_features_chunk(args):
    """Process a chunk of data for spectral shape feature extraction"""
    X_chunk, start_idx, end_idx, wifi_width, bt_width = args
    chunk_size = end_idx - start_idx
    features = []
    
    for i in range(chunk_size):
        signal = X_chunk[i]
        feature_vector = []
        
        # 1. Find peaks and analyze their widths
        peaks, properties = scipy_signal.find_peaks(
            signal, 
            height=np.median(signal),
            distance=bt_width//2
        )
        
        if len(peaks) == 0:
            # Handle case with no peaks
            peak_widths = [0]
            peak_heights = [0]
            peak_prominences = [0]
        else:
            # Calculate widths at different relative heights
            widths_3db = scipy_signal.peak_widths(signal, peaks, rel_height=0.5)[0]
            widths_6db = scipy_signal.peak_widths(signal, peaks, rel_height=0.75)[0]
            widths_10db = scipy_signal.peak_widths(signal, peaks, rel_height=0.9)[0]
            
            # Get peak heights and prominences
            peak_heights = properties['peak_heights']
            peak_prominences = scipy_signal.peak_prominences(signal, peaks)[0]
        
        # 2. Peak width statistics
        if len(peaks) > 0:
            bt_like_peaks = np.sum((widths_3db > 5) & (widths_3db < bt_width*2))
            wifi_like_peaks = np.sum((widths_3db > bt_width*2) & (widths_3db < wifi_width*1.5))
            wide_peaks = np.sum(widths_3db > wifi_width)
            
            # Width/height ratio features
            width_height_ratios = widths_3db / peak_heights
            
            # Add basic width statistics
            feature_vector.extend([
                len(peaks),  # Number of peaks
                np.mean(widths_3db) if len(widths_3db) > 0 else 0,  # Mean 3dB width
                np.std(widths_3db) if len(widths_3db) > 0 else 0,   # Std of 3dB width
                np.mean(widths_6db) if len(widths_6db) > 0 else 0,  # Mean 6dB width
                np.std(widths_6db) if len(widths_6db) > 0 else 0,   # Std of 6dB width
                np.mean(widths_10db) if len(widths_10db) > 0 else 0,  # Mean 10dB width
                np.std(widths_10db) if len(widths_10db) > 0 else 0,   # Std of 10dB width
                bt_like_peaks,  # Number of Bluetooth-like peaks
                wifi_like_peaks,  # Number of WiFi-like peaks
                wide_peaks,  # Number of very wide peaks
                np.mean(width_height_ratios) if len(width_height_ratios) > 0 else 0,
                np.max(width_height_ratios) if len(width_height_ratios) > 0 else 0,
            ])
            
            # Calculate peak width distributions
            width_bins = [0, bt_width/2, bt_width, bt_width*2, wifi_width/2, wifi_width, wifi_width*1.5, 2048]
            width_hist, _ = np.histogram(widths_3db, bins=width_bins)
            feature_vector.extend(width_hist)
            
            # Top 3 peak widths (if available)
            peak_indices = np.argsort(peak_prominences)[-3:]
            top_widths = [widths_3db[i] if i < len(widths_3db) else 0 for i in peak_indices]
            feature_vector.extend(top_widths)
        else:
            # No peaks found - add placeholder values
            feature_vector.extend([0] * 12)  # Basic width statistics
            feature_vector.extend([0] * 7)   # Width histogram
            feature_vector.extend([0] * 3)   # Top peak widths
        
        # 3. Energy distribution in different bandwidths
        # Calculate energy in windows of different widths
        bt_energy = []
        wifi_energy = []
        
        # Slide a BT-width window and a WiFi-width window
        for start in range(0, len(signal)-wifi_width, wifi_width//4):
            if start + wifi_width <= len(signal):
                wifi_window = signal[start:start+wifi_width]
                wifi_energy.append(np.sum(10**(wifi_window/10)))
            
        for start in range(0, len(signal)-bt_width, bt_width//2):
            if start + bt_width <= len(signal):
                bt_window = signal[start:start+bt_width]
                bt_energy.append(np.sum(10**(bt_window/10)))
        
        # Energy statistics
        feature_vector.extend([
            np.max(wifi_energy) if wifi_energy else 0,
            np.std(wifi_energy) if wifi_energy else 0,
            np.max(bt_energy) if bt_energy else 0,
            np.std(bt_energy) if bt_energy else 0,
            np.max(wifi_energy)/np.max(bt_energy) if wifi_energy and bt_energy and np.max(bt_energy) > 0 else 0
        ])
        
        # 4. Spectral shape template correlation
        # Create simplified templates for WiFi and BT
        x = np.linspace(-10, 10, wifi_width)
        wifi_template = np.exp(-0.5 * x**2 / 9)  # Wide Gaussian for WiFi
        
        x = np.linspace(-10, 10, bt_width)
        bt_template = np.exp(-0.5 * x**2)  # Narrow Gaussian for BT
        
        # Calculate correlation with templates
        wifi_corr = []
        bt_corr = []
        
        for start in range(0, len(signal)-wifi_width, wifi_width//2):
            if start + wifi_width <= len(signal):
                # Correlation
                window_signal = signal[start:start+wifi_width]
                if np.std(window_signal) > 0:
                    corr = np.corrcoef(window_signal, wifi_template)[0, 1]
                    wifi_corr.append(corr)
        
        for start in range(0, len(signal)-bt_width, bt_width//2):
            if start + bt_width <= len(signal):
                # Correlation
                window_signal = signal[start:start+bt_width]
                if np.std(window_signal) > 0:
                    corr = np.corrcoef(window_signal, bt_template)[0, 1]
                    bt_corr.append(corr)
        
        # Add correlation statistics
        feature_vector.extend([
            np.max(wifi_corr) if wifi_corr else 0,
            np.mean(wifi_corr) if wifi_corr else 0,
            np.max(bt_corr) if bt_corr else 0,
            np.mean(bt_corr) if bt_corr else 0,
        ])
        
        # 5. Additional statistical features (still useful)
        feature_vector.extend([
            np.mean(signal),
            np.std(signal),
            np.max(signal),
            np.min(signal),
            np.median(signal),
            np.percentile(signal, 25),
            np.percentile(signal, 75),
        ])
        
        features.append(feature_vector)
    
    return features


def extract_bandwidth_features_chunk(args):
    """Process a chunk of data for bandwidth feature extraction"""
    X_chunk, start_idx, end_idx, wifi_width, bt_width = args
    chunk_size = end_idx - start_idx
    features = []
    
    for i in range(chunk_size):
        signal = X_chunk[i]
        feature_vector = []
        
        # Divide signal into segments
        num_segments = 8  # Create 8 segments
        segment_size = len(signal) // num_segments
        
        for j in range(num_segments):
            start = j * segment_size
            end = (j + 1) * segment_size
            segment = signal[start:end]
            
            # Find peaks in this segment
            peaks, properties = scipy_signal.find_peaks(
                segment, 
                height=np.median(segment),
                distance=bt_width//2
            )
            
            if len(peaks) == 0:
                # No peaks in this segment
                feature_vector.extend([0, 0, 0, 0, 0])
                continue
                
            # Calculate peak widths
            widths = scipy_signal.peak_widths(segment, peaks, rel_height=0.5)[0]
            
            # Calculate bandwidth features for this segment
            bt_count = np.sum((widths > 5) & (widths < bt_width*1.5))
            wifi_count = np.sum((widths > bt_width*2) & (widths < wifi_width*1.2))
            
            # Average width
            avg_width = np.mean(widths) if len(widths) > 0 else 0
            
            # Peak height to width ratio
            heights = properties['peak_heights']
            hw_ratio = np.mean(heights / widths) if len(widths) > 0 else 0
            
            # Energy in peak regions vs total energy
            peak_energy = 0
            for p, w in zip(peaks, widths):
                left = max(0, int(p - w/2))
                right = min(len(segment), int(p + w/2))
                peak_region = segment[left:right]
                peak_energy += np.sum(10**(peak_region/10))
            
            total_energy = np.sum(10**(segment/10))
            energy_ratio = peak_energy / total_energy if total_energy > 0 else 0
            
            feature_vector.extend([
                bt_count,
                wifi_count,
                avg_width,
                hw_ratio,
                energy_ratio
            ])
        
        features.append(feature_vector)
    
    return features


# ------------------- Parallelized Feature Extraction -------------------
def extract_spectral_shape_features(X):
    """Extract features based on spectral shape characteristics using multiprocessing"""
    # With 2048 points over 100MHz:
    # - WiFi channel width ~20MHz = ~410 bins
    # - Bluetooth width ~1MHz = ~20 bins
    wifi_width = 410
    bt_width = 20
    
    num_samples = X.shape[0]
    
    # Determine number of processes to use (CPU cores - 1, minimum 1)
    num_processes = max(1, multiprocessing.cpu_count() - 1)
    print(f"Using {num_processes} processes for feature extraction")
    
    # Create chunks of data for parallel processing
    chunk_indices = get_chunk_indices(num_samples, num_processes)
    
    # Prepare arguments for each worker
    chunk_args = []
    for start_idx, end_idx in chunk_indices:
        chunk_args.append((X[start_idx:end_idx], start_idx, end_idx, wifi_width, bt_width))
    
    # Create process pool and extract features in parallel
    start_time = time.time()
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = list(tqdm(
            pool.imap(extract_spectral_shape_features_chunk, chunk_args),
            total=len(chunk_args),
            desc="Extracting spectral shape features",
            ncols=100
        ))
    
    # Combine results from all processes
    all_features = []
    for chunk_features in results:
        all_features.extend(chunk_features)
    
    print(f"Feature extraction completed in {time.time() - start_time:.2f} seconds")
    
    return np.array(all_features)


def extract_segmented_bandwidth_features(X):
    """Extract features from segments focused on bandwidth characteristics using multiprocessing"""
    # Define parameters based on expected signal characteristics
    wifi_width = 410  # ~20MHz in bins
    bt_width = 20     # ~1MHz in bins
    
    num_samples = X.shape[0]
    
    # Determine number of processes to use
    num_processes = max(1, multiprocessing.cpu_count() - 1)
    print(f"Using {num_processes} processes for feature extraction")
    
    # Create chunks of data for parallel processing
    chunk_indices = get_chunk_indices(num_samples, num_processes)
    
    # Prepare arguments for each worker
    chunk_args = []
    for start_idx, end_idx in chunk_indices:
        chunk_args.append((X[start_idx:end_idx], start_idx, end_idx, wifi_width, bt_width))
    
    # Create process pool and extract features in parallel
    start_time = time.time()
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = list(tqdm(
            pool.imap(extract_bandwidth_features_chunk, chunk_args),
            total=len(chunk_args),
            desc="Extracting bandwidth features",
            ncols=100
        ))
    
    # Combine results from all processes
    all_features = []
    for chunk_features in results:
        all_features.extend(chunk_features)
    
    print(f"Feature extraction completed in {time.time() - start_time:.2f} seconds")
    
    return np.array(all_features)


def extract_features(X, feature_method='combined'):
    """Extract features from PSD data using different methods with multiprocessing"""
    print(f"Extracting features using method: {feature_method}")
    
    if feature_method == 'spectral_shape':
        return extract_spectral_shape_features(X)
    elif feature_method == 'bandwidth':
        return extract_segmented_bandwidth_features(X)
    elif feature_method == 'combined':
        # Combine multiple feature types
        shape_features = extract_spectral_shape_features(X)
        bandwidth_features = extract_segmented_bandwidth_features(X)
        
        # Combine features horizontally
        return np.hstack((shape_features, bandwidth_features))
    else:
        print(f"Unknown feature method: {feature_method}. Using combined features instead.")
        return np.hstack((extract_spectral_shape_features(X), extract_segmented_bandwidth_features(X)))


# ------------------- Width-Based Classification Helper -------------------
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


def adjust_predictions_by_width(X, y_pred_proba, event_widths, noise_floor, width_threshold=200):
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


# ------------------- Training and Evaluation -------------------
def train_and_evaluate(X_train, y_train, X_test, y_test, train_event_widths, test_event_widths, feature_method='combined', noise_floor=-190.0):
    """Train and evaluate LightGBM model"""
    # Extract features
    X_train_features = extract_features(X_train, feature_method)
    X_test_features = extract_features(X_test, feature_method)
    
    print(f"Feature shapes - Train: {X_train_features.shape}, Test: {X_test_features.shape}")
    
    # Create LightGBM model
    print("\n--- Training LightGBM model ---")
    
    model = lgbm.LGBMClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=15,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        n_jobs=-1,
        random_state=42
    )
    
    # Train model
    start_time = time.time()
    model.fit(X_train_features, y_train)
    train_time = time.time() - start_time
    
    # Get initial predictions
    y_pred_proba = model.predict_proba(X_test_features)[:, 1]
    y_pred_initial = (y_pred_proba > 0.5).astype(int)
    
    # Adjust predictions based on signal width
    y_pred_adjusted, num_adjustments = adjust_predictions_by_width(X_test, y_pred_proba, test_event_widths, noise_floor)
    
    # Calculate metrics for initial predictions
    accuracy_initial = np.mean(y_pred_initial == y_test) * 100
    cm_initial = confusion_matrix(y_test, y_pred_initial)
    
    # Calculate metrics for adjusted predictions
    accuracy_adjusted = np.mean(y_pred_adjusted == y_test) * 100
    cm_adjusted = confusion_matrix(y_test, y_pred_adjusted)
    
    # Additional metrics for adjusted predictions
    tn, fp, fn, tp = cm_adjusted.ravel()
    sensitivity = tp / (tp + fn) if tp + fn > 0 else 0
    specificity = tn / (tn + fp) if tn + fp > 0 else 0
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
    
    # ROC AUC calculation (uses original probabilities)
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    # Print metrics
    print(f"Initial accuracy: {accuracy_initial:.2f}%")
    print(f"Width-adjusted accuracy: {accuracy_adjusted:.2f}%")
    print(f"Number of predictions adjusted: {num_adjustments} ({100 * num_adjustments / len(y_test):.2f}%)")
    print(f"Sensitivity (BT recall): {sensitivity:.4f}")
    print(f"Specificity (WiFi recall): {specificity:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"Training time: {train_time:.2f} seconds")
    print(f"Confusion Matrix (width-adjusted):")
    print(cm_adjusted)
    
    # Get feature importance
    feature_importance = model.feature_importances_
    
    # Create result dictionary
    result = {
        'accuracy_initial': accuracy_initial,
        'accuracy_adjusted': accuracy_adjusted,
        'num_adjustments': num_adjustments,
        'adjustment_percentage': 100 * num_adjustments / len(y_test),
        'sensitivity': sensitivity,
        'specificity': specificity,
        'f1': f1,
        'confusion_matrix_initial': cm_initial.tolist(),
        'confusion_matrix_adjusted': cm_adjusted.tolist(),
        'roc_auc': roc_auc,
        'training_time': train_time,
        'feature_importance': feature_importance.tolist()
    }
    
    return model, result, X_test_features


def create_outputs(model, result, feature_method, X_test_features, noise_floor):
    """Create the three required output files: model, confusion matrix, and metadata"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Save model file
    model_filename = f"lightgbm_model_{timestamp}.joblib"
    joblib.dump(model, model_filename)
    print(f"Model saved to {model_filename}")
    
    # 2. Create and save confusion matrix visualization
    plt.figure(figsize=(12, 6))
    
    # Plot initial confusion matrix
    plt.subplot(1, 2, 1)
    cm_initial = np.array(result['confusion_matrix_initial'])
    plt.imshow(cm_initial, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Initial Confusion Matrix')
    plt.colorbar()
    
    classes = ['WiFi', 'Bluetooth']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    
    # Add text annotations
    thresh = cm_initial.max() / 2
    for i in range(cm_initial.shape[0]):
        for j in range(cm_initial.shape[1]):
            plt.text(j, i, format(cm_initial[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if cm_initial[i, j] > thresh else "black")
    
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    # Plot width-adjusted confusion matrix
    plt.subplot(1, 2, 2)
    cm_adjusted = np.array(result['confusion_matrix_adjusted'])
    plt.imshow(cm_adjusted, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Width-Adjusted Confusion Matrix')
    plt.colorbar()
    
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    
    # Add text annotations
    thresh = cm_adjusted.max() / 2
    for i in range(cm_adjusted.shape[0]):
        for j in range(cm_adjusted.shape[1]):
            plt.text(j, i, format(cm_adjusted[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if cm_adjusted[i, j] > thresh else "black")
    
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    plt.tight_layout()
    
    cm_filename = f"confusion_matrix_{timestamp}.png"
    plt.savefig(cm_filename, dpi=300)
    plt.close()
    print(f"Confusion matrix saved to {cm_filename}")
    
    # 3. Create metadata file with key information
    # Get top features by importance
    feature_importance = np.array(result['feature_importance'])
    
    # Define feature names based on the feature extraction method
    if feature_method == 'spectral_shape':
        feature_names = [
            'Num Peaks', 'Mean 3dB Width', 'Std 3dB Width', 
            'Mean 6dB Width', 'Std 6dB Width', 'Mean 10dB Width', 'Std 10dB Width',
            'BT-like Peaks', 'WiFi-like Peaks', 'Wide Peaks', 
            'Mean Width/Height', 'Max Width/Height'
        ]
        feature_names.extend([f'Width Hist {i}' for i in range(7)])
        feature_names.extend([f'Top Peak Width {i}' for i in range(3)])
        feature_names.extend([
            'Max WiFi Energy', 'Std WiFi Energy', 
            'Max BT Energy', 'Std BT Energy', 
            'WiFi/BT Energy Ratio'
        ])
        feature_names.extend([
            'Max WiFi Corr', 'Mean WiFi Corr',
            'Max BT Corr', 'Mean BT Corr'
        ])
        feature_names.extend([
            'Mean', 'Std', 'Max', 'Min', 'Median', 'P25', 'P75'
        ])
    elif feature_method == 'bandwidth':
        segments = 8
        segment_features = ['BT Count', 'WiFi Count', 'Avg Width', 'H/W Ratio', 'Energy Ratio']
        feature_names = []
        
        for i in range(segments):
            for f in segment_features:
                feature_names.append(f'Segment {i+1} {f}')
    elif feature_method == 'combined':
        # Spectral shape features
        spectral_names = [
            'Num Peaks', 'Mean 3dB Width', 'Std 3dB Width', 
            'Mean 6dB Width', 'Std 6dB Width', 'Mean 10dB Width', 'Std 10dB Width',
            'BT-like Peaks', 'WiFi-like Peaks', 'Wide Peaks', 
            'Mean Width/Height', 'Max Width/Height'
        ]
        spectral_names.extend([f'Width Hist {i}' for i in range(7)])
        spectral_names.extend([f'Top Peak Width {i}' for i in range(3)])
        spectral_names.extend([
            'Max WiFi Energy', 'Std WiFi Energy', 
            'Max BT Energy', 'Std BT Energy', 
            'WiFi/BT Energy Ratio'
        ])
        spectral_names.extend([
            'Max WiFi Corr', 'Mean WiFi Corr',
            'Max BT Corr', 'Mean BT Corr'
        ])
        spectral_names.extend([
            'Mean', 'Std', 'Max', 'Min', 'Median', 'P25', 'P75'
        ])
        
        # Bandwidth features
        segments = 8
        segment_features = ['BT Count', 'WiFi Count', 'Avg Width', 'H/W Ratio', 'Energy Ratio']
        bandwidth_names = []
        for i in range(segments):
            for f in segment_features:
                bandwidth_names.append(f'Segment {i+1} {f}')
                
        feature_names = spectral_names + bandwidth_names
    else:
        feature_names = [f'Feature {i}' for i in range(len(feature_importance))]
    
    # Ensure we have the right number of names
    if len(feature_names) > len(feature_importance):
        feature_names = feature_names[:len(feature_importance)]
    elif len(feature_names) < len(feature_importance):
        feature_names.extend([f'Feature {i}' for i in range(len(feature_names), len(feature_importance))])
    
    # Get top 10 most important features
    top_indices = np.argsort(feature_importance)[::-1][:10]
    top_features = [(feature_names[i], float(feature_importance[i])) for i in top_indices]
    
    # Create model parameters dict
    model_params = {
        'n_estimators': model.n_estimators,
        'learning_rate': model.learning_rate,
        'max_depth': model.max_depth,
        'num_leaves': model.num_leaves,
        'subsample': model.subsample,
        'colsample_bytree': model.colsample_bytree,
        'reg_alpha': model.reg_alpha,
        'reg_lambda': model.reg_lambda
    }
    
    # Create metadata
    metadata = {
        'timestamp': timestamp,
        'feature_method': feature_method,
        'model_file': model_filename,
        'confusion_matrix_file': cm_filename,
        'performance': {
            'accuracy_initial': float(result['accuracy_initial']),
            'accuracy_adjusted': float(result['accuracy_adjusted']),
            'num_predictions_adjusted': int(result['num_adjustments']),
            'adjustment_percentage': float(result['adjustment_percentage']),
            'sensitivity': float(result['sensitivity']),
            'specificity': float(result['specificity']),
            'f1_score': float(result['f1']),
            'roc_auc': float(result['roc_auc']),
            'training_time_seconds': float(result['training_time'])
        },
        'preprocessing': {
            'event_detection': True,
            'noise_floor': float(noise_floor),
            'width_based_adjustment': True,
            'width_threshold_mhz': 10.0
        },
        'model_parameters': model_params,
        'top_features': top_features,
        'feature_dimensions': X_test_features.shape[1]
    }
    
    # Save metadata
    metadata_filename = f"model_metadata_{timestamp}.json"
    with open(metadata_filename, 'w') as f:
        json.dump(metadata, f, indent=4)
    
    print(f"Model metadata saved to {metadata_filename}")
    
    return model_filename, cm_filename, metadata_filename


# ------------------- Main Function -------------------
def run_rf_classification(data_path, feature_method='combined'):
    """Main function to run the RF classification pipeline using only LightGBM"""
    print("=" * 80)
    print("RF Signal Classifier - WiFi vs Bluetooth using LightGBM")
    print("=" * 80)
    
    # Load and split data
    X_train, X_test, y_train, y_test, train_event_widths, test_event_widths, noise_floor = load_and_split_data(data_path)
    
    print("\n" + "=" * 80)
    print(f"Feature Extraction Method: {feature_method}")
    print("=" * 80)
    
    # Train and evaluate model
    model, result, X_test_features = train_and_evaluate(
        X_train, y_train, X_test, y_test, 
        train_event_widths, test_event_widths, 
        feature_method, noise_floor
    )
    
    # Create output files
    model_path, cm_path, metadata_path = create_outputs(model, result, feature_method, X_test_features, noise_floor)
    
    print("\n" + "=" * 80)
    print("Classification Complete")
    print("=" * 80)
    print(f"Output files:")
    print(f"1. Model: {model_path}")
    print(f"2. Confusion Matrix: {cm_path}")
    print(f"3. Metadata: {metadata_path}")
    
    return model, result


# ------------------- Deployable Classifier -------------------
class RFClassifier:
    """RF Signal Classifier for WiFi vs Bluetooth using LightGBM"""
    
    def __init__(self, model, feature_method='combined', width_threshold=200):
        self.model = model
        self.feature_method = feature_method
        self.width_threshold = width_threshold  # ~10 MHz in bins
        # Define expected signal characteristics
        self.wifi_width = 410  # ~20MHz in bins
        self.bt_width = 20     # ~1MHz in bins
    
    def preprocess_signal(self, X, noise_floor):
        """Apply event detection and noise floor replacement"""
        X_processed = X.copy()
        event_widths = []
        
        for i in range(X.shape[0]):
            events = event_detector(X[i])
            signal_mask = np.ones(X.shape[1]) * noise_floor
            
            total_width = 0
            for start, end in events:
                if start != 0 or end != 0:  # Skip if no event detected
                    signal_mask[start:end+1] = X[i, start:end+1]
                    total_width += (end - start + 1)
            
            event_widths.append(total_width)
            X_processed[i] = signal_mask
        
        return X_processed, event_widths
    
    def extract_features(self, X):
        """Extract features for model prediction"""
        if self.feature_method == 'spectral_shape':
            return self._extract_spectral_shape_features_direct(X)
        elif self.feature_method == 'bandwidth':
            return self._extract_bandwidth_features_direct(X)
        elif self.feature_method == 'combined':
            spectral = self._extract_spectral_shape_features_direct(X)
            bandwidth = self._extract_bandwidth_features_direct(X)
            return np.hstack((spectral, bandwidth))
        else:
            # Default to combined features
            spectral = self._extract_spectral_shape_features_direct(X)
            bandwidth = self._extract_bandwidth_features_direct(X)
            return np.hstack((spectral, bandwidth))
            
    def _extract_spectral_shape_features_direct(self, X):
        """Direct feature extraction for model prediction"""
        features = []
        
        for i in range(X.shape[0]):
            signal = X[i]
            feature_vector = []
            
            # Find peaks
            peaks, properties = scipy_signal.find_peaks(
                signal, 
                height=np.median(signal),
                distance=self.bt_width//2
            )
            
            if len(peaks) == 0:
                peak_widths = [0]
                peak_heights = [0]
                peak_prominences = [0]
            else:
                # Calculate widths at different levels
                widths_3db = scipy_signal.peak_widths(signal, peaks, rel_height=0.5)[0]
                widths_6db = scipy_signal.peak_widths(signal, peaks, rel_height=0.75)[0]
                widths_10db = scipy_signal.peak_widths(signal, peaks, rel_height=0.9)[0]
                
                peak_heights = properties['peak_heights']
                peak_prominences = scipy_signal.peak_prominences(signal, peaks)[0]
            
            # Calculate features
            if len(peaks) > 0:
                bt_like_peaks = np.sum((widths_3db > 5) & (widths_3db < self.bt_width*2))
                wifi_like_peaks = np.sum((widths_3db > self.bt_width*2) & (widths_3db < self.wifi_width*1.5))
                wide_peaks = np.sum(widths_3db > self.wifi_width)
                
                width_height_ratios = widths_3db / peak_heights
                
                feature_vector.extend([
                    len(peaks),
                    np.mean(widths_3db) if len(widths_3db) > 0 else 0,
                    np.std(widths_3db) if len(widths_3db) > 0 else 0,
                    np.mean(widths_6db) if len(widths_6db) > 0 else 0,
                    np.std(widths_6db) if len(widths_6db) > 0 else 0,
                    np.mean(widths_10db) if len(widths_10db) > 0 else 0,
                    np.std(widths_10db) if len(widths_10db) > 0 else 0,
                    bt_like_peaks,
                    wifi_like_peaks,
                    wide_peaks,
                    np.mean(width_height_ratios) if len(width_height_ratios) > 0 else 0,
                    np.max(width_height_ratios) if len(width_height_ratios) > 0 else 0,
                ])
                
                # Width histogram
                width_bins = [0, self.bt_width/2, self.bt_width, self.bt_width*2, 
                              self.wifi_width/2, self.wifi_width, self.wifi_width*1.5, 2048]
                width_hist, _ = np.histogram(widths_3db, bins=width_bins)
                feature_vector.extend(width_hist)
                
                # Top 3 peak widths
                peak_indices = np.argsort(peak_prominences)[-3:]
                top_widths = [widths_3db[i] if i < len(widths_3db) else 0 for i in peak_indices]
                feature_vector.extend(top_widths)
            else:
                feature_vector.extend([0] * 12)  # Basic width statistics
                feature_vector.extend([0] * 7)   # Width histogram
                feature_vector.extend([0] * 3)   # Top peak widths
            
            # Energy distribution
            bt_energy = []
            wifi_energy = []
            
            for start in range(0, len(signal)-self.wifi_width, self.wifi_width//4):
                if start + self.wifi_width <= len(signal):
                    wifi_window = signal[start:start+self.wifi_width]
                    wifi_energy.append(np.sum(10**(wifi_window/10)))
            
            for start in range(0, len(signal)-self.bt_width, self.bt_width//2):
                if start + self.bt_width <= len(signal):
                    bt_window = signal[start:start+self.bt_width]
                    bt_energy.append(np.sum(10**(bt_window/10)))
            
            feature_vector.extend([
                np.max(wifi_energy) if wifi_energy else 0,
                np.std(wifi_energy) if wifi_energy else 0,
                np.max(bt_energy) if bt_energy else 0,
                np.std(bt_energy) if bt_energy else 0,
                np.max(wifi_energy)/np.max(bt_energy) if wifi_energy and bt_energy and np.max(bt_energy) > 0 else 0
            ])
            
            # Template correlation
            x = np.linspace(-10, 10, self.wifi_width)
            wifi_template = np.exp(-0.5 * x**2 / 9)
            
            x = np.linspace(-10, 10, self.bt_width)
            bt_template = np.exp(-0.5 * x**2)
            
            wifi_corr = []
            bt_corr = []
            
            for start in range(0, len(signal)-self.wifi_width, self.wifi_width//2):
                if start + self.wifi_width <= len(signal):
                    window_signal = signal[start:start+self.wifi_width]
                    if np.std(window_signal) > 0:
                        corr = np.corrcoef(window_signal, wifi_template)[0, 1]
                        wifi_corr.append(corr)
            
            for start in range(0, len(signal)-self.bt_width, self.bt_width//2):
                if start + self.bt_width <= len(signal):
                    window_signal = signal[start:start+self.bt_width]
                    if np.std(window_signal) > 0:
                        corr = np.corrcoef(window_signal, bt_template)[0, 1]
                        bt_corr.append(corr)
            
            feature_vector.extend([
                np.max(wifi_corr) if wifi_corr else 0,
                np.mean(wifi_corr) if wifi_corr else 0,
                np.max(bt_corr) if bt_corr else 0,
                np.mean(bt_corr) if bt_corr else 0,
            ])
            
            # Statistical features
            feature_vector.extend([
                np.mean(signal),
                np.std(signal),
                np.max(signal),
                np.min(signal),
                np.median(signal),
                np.percentile(signal, 25),
                np.percentile(signal, 75),
            ])
            
            features.append(feature_vector)
        
        return np.array(features)
    
    def _extract_bandwidth_features_direct(self, X):
        """Direct bandwidth feature extraction for model prediction"""
        features = []
        
        for i in range(X.shape[0]):
            signal = X[i]
            feature_vector = []
            
            # Divide signal into segments
            num_segments = 8
            segment_size = len(signal) // num_segments
            
            for j in range(num_segments):
                start = j * segment_size
                end = (j + 1) * segment_size
                segment = signal[start:end]
                
                # Find peaks in this segment
                peaks, properties = scipy_signal.find_peaks(
                    segment, 
                    height=np.median(segment),
                    distance=self.bt_width//2
                )
                
                if len(peaks) == 0:
                    # No peaks in this segment
                    feature_vector.extend([0, 0, 0, 0, 0])
                    continue
                    
                # Calculate peak widths
                widths = scipy_signal.peak_widths(segment, peaks, rel_height=0.5)[0]
                
                # Calculate bandwidth features for this segment
                bt_count = np.sum((widths > 5) & (widths < self.bt_width*1.5))
                wifi_count = np.sum((widths > self.bt_width*2) & (widths < self.wifi_width*1.2))
                
                # Average width
                avg_width = np.mean(widths) if len(widths) > 0 else 0
                
                # Peak height to width ratio
                heights = properties['peak_heights']
                hw_ratio = np.mean(heights / widths) if len(widths) > 0 else 0
                
                # Energy in peak regions vs total energy
                peak_energy = 0
                for p, w in zip(peaks, widths):
                    left = max(0, int(p - w/2))
                    right = min(len(segment), int(p + w/2))
                    peak_region = segment[left:right]
                    peak_energy += np.sum(10**(peak_region/10))
                
                total_energy = np.sum(10**(segment/10))
                energy_ratio = peak_energy / total_energy if total_energy > 0 else 0
                
                feature_vector.extend([
                    bt_count,
                    wifi_count,
                    avg_width,
                    hw_ratio,
                    energy_ratio
                ])
            
            features.append(feature_vector)
        
        return np.array(features)
    
    def predict(self, X, noise_floor=-190.0):
        """Predict WiFi (0) or Bluetooth (1) from raw PSD data with width-based adjustment"""
        # Replace -inf values with noise floor - 10
        X_clean = X.copy()
        X_clean[X_clean == -np.inf] = noise_floor - 10
        
        # Apply event detection preprocessing
        X_processed, event_widths = self.preprocess_signal(X_clean, noise_floor)
        
        # Extract features
        X_features = self.extract_features(X_processed)
        
        # Get model prediction probabilities
        y_proba = self.model.predict_proba(X_features)[:, 1]
        
        # Initial prediction
        y_pred = (y_proba > 0.5).astype(int)
        
        # Adjust based on signal width
        for i in range(len(y_pred)):
            width = event_widths[i]
            
            # Calculate model confidence
            confidence = max(y_proba[i], 1 - y_proba[i])
            
            # Only override if confidence is not very high
            if confidence < 0.85:
                if width < self.width_threshold:
                    # Narrow signal - more likely Bluetooth (class 1)
                    y_pred[i] = 1
                else:
                    # Wide signal - more likely WiFi (class 0)
                    y_pred[i] = 0
        
        return y_pred
    
    def predict_proba(self, X, noise_floor=-190.0):
        """Predict class probabilities with event detection preprocessing"""
        # Replace -inf values with noise floor - 10
        X_clean = X.copy()
        X_clean[X_clean == -np.inf] = noise_floor - 10
        
        # Apply event detection preprocessing
        X_processed, _ = self.preprocess_signal(X_clean, noise_floor)
        
        # Extract features
        X_features = self.extract_features(X_processed)
        
        # Make prediction
        return self.model.predict_proba(X_features)


if __name__ == "__main__":
    # Define path to data file
    data_path = "./data/dataset.h5"  # Replace with your actual h5 file
    
    # Feature extraction method - default to combined for best performance
    feature_method = 'combined'
    
    # Run classification
    run_rf_classification(data_path, feature_method)
