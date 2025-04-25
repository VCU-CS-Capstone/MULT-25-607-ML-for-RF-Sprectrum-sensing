import os
import time
import h5py
import numpy as np
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
from scipy import signal as scipy_signal
from sklearn.metrics import confusion_matrix, roc_curve, auc 
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgbm
import xgboost as xgb
import joblib
import warnings
import multiprocessing
warnings.filterwarnings('ignore')


# ------------------- Data Handling -------------------
def load_data(file_path):
    """Load data from h5 file and prepare for ML models"""
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
    
    return data_array, labels


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


def extract_wavelet_features_chunk(args):
    """Process a chunk of data for wavelet feature extraction"""
    try:
        import pywt
    except ImportError:
        print("PyWavelets not installed. Please install with: pip install pywt")
        return []
    
    X_chunk, start_idx, end_idx = args
    chunk_size = end_idx - start_idx
    features = []
    
    for i in range(chunk_size):
        signal = X_chunk[i]
        
        # Perform wavelet decomposition
        coeffs = pywt.wavedec(signal, 'db4', level=5)
        
        # Extract statistics from each level
        feature_vector = []
        for c in coeffs:
            feature_vector.extend([
                np.mean(np.abs(c)),
                np.std(c),
                np.max(np.abs(c)),
                np.sum(c**2),  # Energy
                np.sum(np.abs(c))  # L1 norm
            ])
        
        # Add basic shape features
        signal_diff = np.diff(signal)
        feature_vector.extend([
            np.mean(signal),
            np.std(signal),
            np.max(signal),
            np.mean(np.abs(signal_diff)),
            np.std(signal_diff),
            np.percentile(signal, 25),
            np.percentile(signal, 75),
        ])
        
        features.append(feature_vector)
    
    return features


def extract_wavelet_features(X):
    """Extract wavelet-based features using multiprocessing"""
    num_samples = X.shape[0]
    
    # Determine number of processes to use
    num_processes = max(1, multiprocessing.cpu_count() - 1)
    print(f"Using {num_processes} processes for feature extraction")
    
    # Create chunks of data for parallel processing
    chunk_indices = get_chunk_indices(num_samples, num_processes)
    
    # Prepare arguments for each worker
    chunk_args = []
    for start_idx, end_idx in chunk_indices:
        chunk_args.append((X[start_idx:end_idx], start_idx, end_idx))
    
    # Create process pool and extract features in parallel
    start_time = time.time()
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = list(tqdm(
            pool.imap(extract_wavelet_features_chunk, chunk_args),
            total=len(chunk_args),
            desc="Extracting wavelet features",
            ncols=100
        ))
    
    # Combine results from all processes
    all_features = []
    for chunk_features in results:
        all_features.extend(chunk_features)
    
    print(f"Feature extraction completed in {time.time() - start_time:.2f} seconds")
    
    return np.array(all_features)


def extract_features(X, feature_method='spectral_shape'):
    """Extract features from PSD data using different methods with multiprocessing"""
    print(f"Extracting features using method: {feature_method}")
    
    if feature_method == 'spectral_shape':
        return extract_spectral_shape_features(X)
    elif feature_method == 'wavelet':
        return extract_wavelet_features(X)
    elif feature_method == 'bandwidth':
        return extract_segmented_bandwidth_features(X)
    elif feature_method == 'combined':
        # Combine multiple feature types
        shape_features = extract_spectral_shape_features(X)
        bandwidth_features = extract_segmented_bandwidth_features(X)
        
        # Combine features horizontally
        return np.hstack((shape_features, bandwidth_features))
    else:
        print(f"Unknown feature method: {feature_method}. Using spectral_shape instead.")
        return extract_spectral_shape_features(X)


# ------------------- Models -------------------
def create_models(feature_dims):
    """Create different ML models for comparison"""
    print(f"Creating models for feature dimensionality: {feature_dims}")
    
    models = {
        "LightGBM": lgbm.LGBMClassifier(
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
        ),
        
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            bootstrap=True,
            n_jobs=-1,
            random_state=42
        ),
        
        "XGBoost": xgb.XGBClassifier(
            n_estimators=100,
            max_depth=10,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1,
            n_jobs=-1,
            random_state=42
        )
    }
    
    return models


# ------------------- Training and Evaluation -------------------
def train_and_evaluate(models, X_train, y_train, X_test, y_test, feature_method='spectral_shape'):
    """Train and evaluate multiple models"""
    results = {}
    
    # Extract features
    X_train_features = extract_features(X_train, feature_method)
    X_test_features = extract_features(X_test, feature_method)
    
    print(f"Feature shapes - Train: {X_train_features.shape}, Test: {X_test_features.shape}")
    
    # Create models with the right dimensions
    feature_dims = X_train_features.shape[1]
    if models is None:
        models = create_models(feature_dims)
    
    for name, model in models.items():
        print(f"\n--- Training {name} ---")
        
        # Train model
        start_time = time.time()
        model.fit(X_train_features, y_train)
        train_time = time.time() - start_time
        
        # Predict and evaluate
        y_pred = model.predict(X_test_features)
        y_proba = model.predict_proba(X_test_features)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Calculate metrics
        accuracy = np.mean(y_pred == y_test) * 100
        cm = confusion_matrix(y_test, y_pred)
        
        # Additional metrics
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if tp + fn > 0 else 0
        specificity = tn / (tn + fp) if tn + fp > 0 else 0
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
        
        # ROC AUC calculation
        if y_proba is not None:
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_auc = auc(fpr, tpr)
        else:
            roc_auc = None
        
        # Save results
        results[name] = {
            'accuracy': accuracy,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'f1': f1,
            'confusion_matrix': cm,
            'roc_auc': roc_auc,
            'training_time': train_time,
            'model': model,
            'features': X_test_features  # Save features for later analysis
        }
        
        # Print metrics
        print(f"Accuracy: {accuracy:.2f}%")
        print(f"Sensitivity (BT recall): {sensitivity:.4f}")
        print(f"Specificity (WiFi recall): {specificity:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"ROC AUC: {roc_auc if roc_auc is not None else 'N/A'}")
        print(f"Training time: {train_time:.2f} seconds")
        print(f"Confusion Matrix:")
        print(cm)
    
    return results


def visualize_results(results, feature_method):
    """Visualize model comparison results"""
    # Create a timestamp for filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Set up the figure
    plt.figure(figsize=(14, 10))
    
    # Extract performance metrics for plotting
    model_names = list(results.keys())
    accuracies = [results[name]['accuracy'] for name in model_names]
    sensitivities = [results[name]['sensitivity'] * 100 for name in model_names]
    specificities = [results[name]['specificity'] * 100 for name in model_names]
    f1_scores = [results[name]['f1'] * 100 for name in model_names]
    train_times = [results[name]['training_time'] for name in model_names]
    
    # Set width of bars
    barWidth = 0.15
    r1 = np.arange(len(model_names))
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r2]
    r4 = [x + barWidth for x in r3]
    
    # Make the plot
    plt.bar(r1, accuracies, width=barWidth, label='Accuracy (%)', color='blue', alpha=0.7)
    plt.bar(r2, sensitivities, width=barWidth, label='Sensitivity (%)', color='green', alpha=0.7)
    plt.bar(r3, specificities, width=barWidth, label='Specificity (%)', color='red', alpha=0.7)
    plt.bar(r4, f1_scores, width=barWidth, label='F1 Score (%)', color='purple', alpha=0.7)
    
    # Add labels and title
    plt.xlabel('Model', fontweight='bold')
    plt.ylabel('Performance (%)', fontweight='bold')
    plt.title(f'Model Performance Comparison - Feature Method: {feature_method}')
    plt.xticks([r + barWidth*1.5 for r in range(len(model_names))], model_names, rotation=45)
    plt.ylim(0, 105)  # Set y-axis limit
    
    # Add legend and grid
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Save figure
    filename = f"model_comparison_{feature_method}_{timestamp}.png"
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    
    # Training time plot
    plt.figure(figsize=(10, 6))
    plt.bar(model_names, train_times, color='orange', alpha=0.7)
    plt.xlabel('Model', fontweight='bold')
    plt.ylabel('Training Time (seconds)', fontweight='bold')
    plt.title('Model Training Time Comparison')
    plt.xticks(rotation=45)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add time values above bars
    for i, v in enumerate(train_times):
        plt.text(i, v + 0.1, f"{v:.2f}s", ha='center')
    
    # Save figure
    time_filename = f"training_time_{feature_method}_{timestamp}.png"
    plt.tight_layout()
    plt.savefig(time_filename, dpi=300)
    plt.close()
    
    print(f"Visualization saved to {filename} and {time_filename}")
    
    return filename, time_filename


def visualize_sample_signals(X, y, feature_method, best_model_name, results, num_samples=3):
    """Visualize sample signals and peak detection/feature analysis"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Get indices of correctly and incorrectly classified samples
    best_model = results[best_model_name]['model']
    features = results[best_model_name]['features']
    predictions = best_model.predict(features)
    
    correct_wifi = np.where((predictions == 0) & (y == 0))[0]
    correct_bt = np.where((predictions == 1) & (y == 1))[0]
    incorrect_wifi = np.where((predictions == 1) & (y == 0))[0]
    incorrect_bt = np.where((predictions == 0) & (y == 1))[0]
    
    # Randomly select samples for each category
    np.random.seed(42)
    sample_indices = {
        'correct_wifi': np.random.choice(correct_wifi, min(num_samples, len(correct_wifi)), replace=False),
        'correct_bt': np.random.choice(correct_bt, min(num_samples, len(correct_bt)), replace=False),
        'incorrect_wifi': np.random.choice(incorrect_wifi, min(num_samples, len(incorrect_wifi)), replace=False) if len(incorrect_wifi) > 0 else [],
        'incorrect_bt': np.random.choice(incorrect_bt, min(num_samples, len(incorrect_bt)), replace=False) if len(incorrect_bt) > 0 else []
    }
    
    # Define WiFi and BT width in bins
    wifi_width = 410  # ~20MHz
    bt_width = 20     # ~1MHz
    
    # Plot each category
    for category, indices in sample_indices.items():
        if len(indices) == 0:
            continue
            
        for i, idx in enumerate(indices):
            signal = X[idx]
            
            plt.figure(figsize=(12, 8))
            
            # Plot the signal
            plt.subplot(2, 1, 1)
            plt.plot(signal)
            plt.title(f"{category.replace('_', ' ').title()} Sample {i+1}")
            plt.ylabel("Power (dB)")
            plt.xlabel("Frequency Bin")
            
            # Add BT and WiFi width indicators
            max_y = np.max(signal)
            min_y = np.min(signal)
            plt.plot([0, bt_width], [min_y, min_y], 'r-', linewidth=2, label='BT Width (1MHz)')
            plt.plot([0, wifi_width], [min_y-5, min_y-5], 'g-', linewidth=2, label='WiFi Width (20MHz)')
            
            # Find and mark peaks
            peaks, properties = scipy_signal.find_peaks(
                signal, height=np.median(signal), distance=bt_width//2
            )
            
            plt.plot(peaks, signal[peaks], "rx", markersize=8)
            
            if len(peaks) > 0:
                # Calculate peak widths
                widths = scipy_signal.peak_widths(signal, peaks, rel_height=0.5)[0]
                
                # Mark peak widths
                for j, (peak, width) in enumerate(zip(peaks, widths)):
                    left = int(peak - width/2)
                    right = int(peak + width/2)
                    if j < 3:  # Only annotate first 3 peaks for clarity
                        plt.annotate(f"Width: {width:.1f}", 
                                   (peak, signal[peak]), 
                                   xytext=(10, 10),
                                   textcoords='offset points',
                                   arrowprops=dict(arrowstyle="->"))
                
                # Classify peaks as BT-like or WiFi-like
                bt_like = (widths > 5) & (widths < bt_width*1.5)
                wifi_like = (widths > bt_width*2) & (widths < wifi_width*1.2)
                
                plt.plot(peaks[bt_like], signal[peaks[bt_like]], "bo", markersize=10, label='BT-like peak')
                plt.plot(peaks[wifi_like], signal[peaks[wifi_like]], "go", markersize=10, label='WiFi-like peak')
            
            plt.legend()
            
            # Plot frequency domain analysis in the second subplot
            plt.subplot(2, 1, 2)
            
            # Get FFT of the signal
            fft_vals = np.abs(np.fft.rfft(signal))
            fft_freqs = np.fft.rfftfreq(len(signal))
            
            plt.plot(fft_freqs, fft_vals)
            plt.title("Frequency Domain Analysis")
            plt.xlabel("Normalized Frequency")
            plt.ylabel("Magnitude")
            
            # Save the figure
            filename = f"signal_analysis_{category}_{i}_{feature_method}_{timestamp}.png"
            plt.tight_layout()
            plt.savefig(filename, dpi=300)
            plt.close()
            
            print(f"Signal visualization saved to {filename}")


def analyze_feature_importance(results, feature_method):
    """Analyze and visualize feature importance for tree-based models"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for name, result in results.items():
        model = result['model']
        
        # Different models have different ways to access feature importance
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif name == "XGBoost":
            importances = model.get_booster().get_score(importance_type='gain')
            # Convert to array format
            if isinstance(importances, dict):
                importances = np.array([importances.get(f"f{i}", 0) for i in range(len(importances))])
        elif name == "LightGBM" and hasattr(model, 'booster_'):
            importances = model.booster_.feature_importance(importance_type='gain')
        else:
            continue  # Skip if no feature importance available
        
        # Create feature names based on the feature extraction method
        if feature_method == 'spectral_shape':
            # Basic features
            feature_names = [
                'Num Peaks', 'Mean 3dB Width', 'Std 3dB Width', 
                'Mean 6dB Width', 'Std 6dB Width', 'Mean 10dB Width', 'Std 10dB Width',
                'BT-like Peaks', 'WiFi-like Peaks', 'Wide Peaks', 
                'Mean Width/Height', 'Max Width/Height'
            ]
            
            # Width histogram bins
            feature_names.extend([f'Width Hist {i}' for i in range(7)])
            
            # Top peak widths
            feature_names.extend([f'Top Peak Width {i}' for i in range(3)])
            
            # Energy features
            feature_names.extend([
                'Max WiFi Energy', 'Std WiFi Energy', 
                'Max BT Energy', 'Std BT Energy', 
                'WiFi/BT Energy Ratio'
            ])
            
            # Correlation features
            feature_names.extend([
                'Max WiFi Corr', 'Mean WiFi Corr',
                'Max BT Corr', 'Mean BT Corr'
            ])
            
            # Statistical features
            feature_names.extend([
                'Mean', 'Std', 'Max', 'Min', 'Median', 'P25', 'P75'
            ])
            
        elif feature_method == 'bandwidth':
            # Each segment has 5 features
            segments = 8
            segment_features = ['BT Count', 'WiFi Count', 'Avg Width', 'H/W Ratio', 'Energy Ratio']
            feature_names = []
            
            for i in range(segments):
                for f in segment_features:
                    feature_names.append(f'Segment {i+1} {f}')
                    
        elif feature_method == 'wavelet':
            # For wavelet features
            feature_names = []
            # 5 statistics for each of 6 coefficient levels
            for i in range(6):
                for stat in ['Mean', 'Std', 'Max', 'Energy', 'L1 Norm']:
                    feature_names.append(f'Wavelet L{i} {stat}')
            
            # Basic statistical features
            feature_names.extend([
                'Mean', 'Std', 'Max', 'Mean Diff', 'Std Diff', 'P25', 'P75'
            ])
            
        elif feature_method == 'combined':
            # For combined features, concat the feature names
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
            # Generic names for other methods
            feature_names = [f'Feature {i}' for i in range(len(importances))]
        
        # Ensure we have the right number of names
        if len(feature_names) > len(importances):
            feature_names = feature_names[:len(importances)]
        elif len(feature_names) < len(importances):
            feature_names.extend([f'Feature {i}' for i in range(len(feature_names), len(importances))])
        
        # Sort features by importance
        indices = np.argsort(importances)[::-1]
        
        # Plot top 20 features
        plt.figure(figsize=(12, 8))
        plt.title(f'Feature Importance - {name} - {feature_method}')
        plt.bar(range(min(20, len(indices))), 
                importances[indices[:20]], 
                align='center')
        plt.xticks(range(min(20, len(indices))), 
                  [feature_names[i] for i in indices[:20]], 
                  rotation=90)
        plt.tight_layout()
        
        # Save figure
        filename = f"feature_importance_{name}_{feature_method}_{timestamp}.png"
        plt.savefig(filename, dpi=300)
        plt.close()
        
        print(f"Feature importance for {name} saved to {filename}")
        
        # Print top 10 most important features
        print(f"\nTop 10 most important features for {name}:")
        for i, idx in enumerate(indices[:10]):
            print(f"{i+1}. {feature_names[idx]}: {importances[idx]:.4f}")


def save_best_model(results, feature_method):
    """Save the best performing model based on accuracy"""
    # Find the best model
    best_model_name = max(results, key=lambda x: results[x]['accuracy'])
    best_model = results[best_model_name]['model']
    accuracy = results[best_model_name]['accuracy']
    
    # Create filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"best_model_{best_model_name}_{feature_method}_{timestamp}.joblib"
    
    # Save the model
    joblib.dump(best_model, filename)
    print(f"\nBest model: {best_model_name} with accuracy {accuracy:.2f}%")
    print(f"Model saved to {filename}")
    
    return filename, best_model_name


# ------------------- Deployable Classifier -------------------
class RFClassifier:
    """RF Signal Classifier for WiFi vs Bluetooth with multiprocessing"""
    
    def __init__(self, model, feature_method='spectral_shape'):
        self.model = model
        self.feature_method = feature_method
        # Define expected signal characteristics
        self.wifi_width = 410  # ~20MHz in bins
        self.bt_width = 20     # ~1MHz in bins
    
    def extract_features(self, X):
        """Extract features from raw PSD data"""
        # For real-time/single sample processing, we don't need multiprocessing
        if X.shape[0] <= 10:  # For small batches, use direct processing
            if self.feature_method == 'spectral_shape':
                return self._extract_spectral_shape_features_direct(X)
            elif self.feature_method == 'bandwidth':
                return self._extract_bandwidth_features_direct(X)
            elif self.feature_method == 'combined':
                spectral = self._extract_spectral_shape_features_direct(X)
                bandwidth = self._extract_bandwidth_features_direct(X)
                return np.hstack((spectral, bandwidth))
            else:
                return self._extract_spectral_shape_features_direct(X)
        else:
            # For larger batches, use the multiprocessing version
            if self.feature_method == 'spectral_shape':
                return extract_spectral_shape_features(X)
            elif self.feature_method == 'bandwidth':
                return extract_segmented_bandwidth_features(X)
            elif self.feature_method == 'combined':
                spectral = extract_spectral_shape_features(X)
                bandwidth = extract_segmented_bandwidth_features(X)
                return np.hstack((spectral, bandwidth))
            else:
                return extract_spectral_shape_features(X)
    
    def _extract_spectral_shape_features_direct(self, X):
        """Direct feature extraction for small batches or single samples"""
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
        """Direct bandwidth feature extraction for small batches"""
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
        """Predict WiFi (0) or Bluetooth (1) from raw PSD data"""
        # Replace -inf values with noise floor - 10
        X_clean = X.copy()
        X_clean[X_clean == -np.inf] = noise_floor - 10
        
        # Extract features
        X_features = self.extract_features(X_clean)
        
        # Make prediction
        return self.model.predict(X_features)
    
    def predict_proba(self, X, noise_floor=-190.0):
        """Predict class probabilities"""
        # Replace -inf values with noise floor - 10
        X_clean = X.copy()
        X_clean[X_clean == -np.inf] = noise_floor - 10
        
        # Extract features
        X_features = self.extract_features(X_clean)
        
        # Make prediction
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X_features)
        else:
            # For models without predict_proba, create a simple 2-class output
            preds = self.model.predict(X_features)
            proba = np.zeros((len(preds), 2))
            proba[np.arange(len(preds)), preds.astype(int)] = 1
            return proba


def create_deployable_classifier(best_model, feature_method):
    """Create and save a deployable classifier"""
    # Create the classifier instance
    classifier = RFClassifier(best_model, feature_method)
    
    # Save the classifier
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"rf_classifier_{feature_method}_{timestamp}.joblib"
    joblib.dump(classifier, filename)
    
    print(f"Deployable classifier saved to {filename}")
    return filename, classifier


# ------------------- Main Function -------------------
def run_rf_classification(train_data_path, test_data_path, feature_methods=None):
    """Main function to run the RF classification pipeline"""
    if feature_methods is None:
        feature_methods = ['spectral_shape', 'bandwidth', 'combined']
    
    print("=" * 80)
    print("RF Signal Classifier - WiFi vs Bluetooth Spectral Shape Analysis")
    print("=" * 80)
    
    # Load data
    X_train, y_train = load_data(train_data_path)
    X_test, y_test = load_data(test_data_path)
    
    best_results = {}
    
    # Try different feature extraction methods
    for feature_method in feature_methods:
        print("\n" + "=" * 80)
        print(f"Feature Extraction Method: {feature_method}")
        print("=" * 80)
        
        # Train and evaluate models
        results = train_and_evaluate(
            None,  # Models will be created inside based on feature dimensions
            X_train, y_train, X_test, y_test, feature_method
        )
        
        # Visualize results
        visualize_results(results, feature_method)
        
        # Analyze feature importance for applicable models
        analyze_feature_importance(results, feature_method)
        
        # Visualize sample signals
        best_model_name = max(results, key=lambda x: results[x]['accuracy'])
        visualize_sample_signals(X_test, y_test, feature_method, best_model_name, results)
        
        # Save best model
        model_path, best_model_name = save_best_model(results, feature_method)
        
        # Create deployable classifier
        classifier_path, _ = create_deployable_classifier(
            results[best_model_name]['model'], 
            feature_method
        )
        
        best_results[feature_method] = {
            'model_name': best_model_name,
            'accuracy': results[best_model_name]['accuracy'],
            'model_path': model_path,
            'classifier_path': classifier_path
        }
    
    # Print overall best method
    print("\n" + "=" * 80)
    print("Overall Results Summary")
    print("=" * 80)
    
    for method, result in best_results.items():
        print(f"Method: {method} - Best Model: {result['model_name']} - Accuracy: {result['accuracy']:.2f}%")
    
    # Find overall best
    best_method = max(best_results, key=lambda x: best_results[x]['accuracy'])
    print(f"\nOverall best approach: {best_method} feature extraction with {best_results[best_method]['model_name']}")
    print(f"Model saved at: {best_results[best_method]['model_path']}")
    print(f"Deployable classifier saved at: {best_results[best_method]['classifier_path']}")
    
    return best_results


if __name__ == "__main__":
    # Define paths
    train_data_path = "train.h5"
    test_data_path = "test.h5"
    
    # Feature extraction methods to try - focused on spectral characteristics
    feature_methods = ['spectral_shape', 'bandwidth', 'combined']
    
    # Run classification
    run_rf_classification(train_data_path, test_data_path, feature_methods)
