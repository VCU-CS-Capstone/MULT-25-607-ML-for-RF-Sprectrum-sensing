"""
mlrf.classifier

Provides event detection and RFClassifier for WiFi/Bluetooth classification.
"""

import numpy as np
from scipy import signal as scipy_signal

def event_detector(psd, threshold_factor=1.05**-1):
    """
    Detect events in PSD data where the signal exceeds a dynamic threshold.

    Args:
        psd (np.ndarray): Power Spectral Density data (1D array).
        threshold_factor (float): Multiplier for dynamic threshold.

    Returns:
        list of tuple: List of (start_idx, end_idx) for detected events.
    """
    window_size = 64
    event_ranges = []
    in_event = False
    start_idx = 0
    signal_mean = np.mean(psd)
    dynamic_threshold = signal_mean * threshold_factor
    i = 0
    while i < len(psd) - window_size + 1:
        window = psd[i : i + window_size]
        window_average = round(np.mean(window), 2)
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
        event_ranges.append((start_idx, len(psd) - 1))
    elif not event_ranges:
        event_ranges.append((0, 0))
    return event_ranges

class RFClassifier:
    """
    RF Signal Classifier for WiFi vs Bluetooth.

    Uses event detection and feature extraction to classify signals.

    Args:
        model: Trained scikit-learn model with predict_proba.
        feature_method (str): Feature extraction method ("combined", "spectral_shape", "bandwidth").
        width_threshold (int): Threshold for distinguishing WiFi/Bluetooth by width (bins).
    """

    def __init__(self, model, feature_method="combined", width_threshold=350):
        self.model = model
        self.feature_method = feature_method
        self.width_threshold = width_threshold
        self.wifi_width = 410
        self.bt_width = 20

    def preprocess_with_event_detection(self, X, noise_floor=-190.0):
        """
        Preprocess data with event detection to isolate signal regions.

        Args:
            X (np.ndarray): Input PSD data (1D or 2D).
            noise_floor (float): Value to use for masked (non-event) regions.

        Returns:
            tuple: (processed_X, event_widths)
        """
        X_processed = X.copy()
        event_widths = []
        if X.ndim == 1:
            X = X.reshape(1, -1)
            X_processed = X_processed.reshape(1, -1)
        for i in range(X.shape[0]):
            events = event_detector(X[i])
            signal_mask = np.ones(X.shape[1]) * noise_floor
            total_width = 0
            for start, end in events:
                if start != 0 or end != 0:
                    signal_mask[start : end + 1] = X[i, start : end + 1]
                    total_width += end - start + 1
            event_widths.append(total_width)
            X_processed[i] = signal_mask
        return X_processed, event_widths

    def extract_features(self, X):
        """
        Extract features from raw PSD data.

        Args:
            X (np.ndarray): Input PSD data (1D or 2D).

        Returns:
            pandas.DataFrame: Feature matrix.
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)
        import pandas as pd
        if self.feature_method == "spectral_shape":
            features = self._extract_spectral_shape_features_direct(X)
            feature_names = [f'spectral_{i}' for i in range(features.shape[1])]
            return pd.DataFrame(features, columns=feature_names)
        elif self.feature_method == "bandwidth":
            features = self._extract_bandwidth_features_direct(X)
            feature_names = [f'bandwidth_{i}' for i in range(features.shape[1])]
            return pd.DataFrame(features, columns=feature_names)
        elif self.feature_method == "combined":
            spectral = self._extract_spectral_shape_features_direct(X)
            bandwidth = self._extract_bandwidth_features_direct(X)
            features = np.hstack((spectral, bandwidth))
            feature_names = [f'combined_{i}' for i in range(features.shape[1])]
            return pd.DataFrame(features, columns=feature_names)
        else:
            spectral = self._extract_spectral_shape_features_direct(X)
            bandwidth = self._extract_bandwidth_features_direct(X)
            features = np.hstack((spectral, bandwidth))
            feature_names = [f'combined_{i}' for i in range(features.shape[1])]
            return pd.DataFrame(features, columns=feature_names)

    def _extract_spectral_shape_features_direct(self, X):
        """
        Extract spectral shape features for each sample.

        Args:
            X (np.ndarray): Input PSD data (2D).

        Returns:
            np.ndarray: Feature matrix.
        """
        features = []
        for i in range(X.shape[0]):
            signal = X[i]
            feature_vector = self._extract_single_spectral_shape(signal)
            features.append(feature_vector)
        return np.array(features)

    def _extract_single_spectral_shape(self, signal):
        """
        Extract spectral shape features for a single PSD sample.

        Args:
            signal (np.ndarray): 1D PSD data.

        Returns:
            list: Feature vector.
        """
        feature_vector = []
        peaks, properties = scipy_signal.find_peaks(
            signal, height=np.median(signal), distance=self.bt_width // 2
        )
        if len(peaks) == 0:
            return [0] * 12 + [0] * 7 + [0] * 3 + [0] * 5 + [0] * 4 + [0] * 7
        widths_3db = scipy_signal.peak_widths(signal, peaks, rel_height=0.5)[0]
        widths_6db = scipy_signal.peak_widths(signal, peaks, rel_height=0.75)[0]
        widths_10db = scipy_signal.peak_widths(signal, peaks, rel_height=0.9)[0]
        peak_heights = properties["peak_heights"]
        peak_prominences = scipy_signal.peak_prominences(signal, peaks)[0]
        bt_like_peaks = np.sum((widths_3db > 5) & (widths_3db < self.bt_width * 2))
        wifi_like_peaks = np.sum(
            (widths_3db > self.bt_width * 2) & (widths_3db < self.wifi_width * 1.5)
        )
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
        width_bins = [
            0,
            self.bt_width / 2,
            self.bt_width,
            self.bt_width * 2,
            self.wifi_width / 2,
            self.wifi_width,
            self.wifi_width * 1.5,
            2048,
        ]
        width_hist, _ = np.histogram(widths_3db, bins=width_bins)
        feature_vector.extend(width_hist)
        peak_indices = np.argsort(peak_prominences)[-3:]
        top_widths = [
            widths_3db[i] if i < len(widths_3db) else 0 for i in peak_indices
        ]
        feature_vector.extend(top_widths)
        bt_energy, wifi_energy = self._energy_distribution(signal)
        feature_vector.extend([
            np.max(wifi_energy) if wifi_energy else 0,
            np.std(wifi_energy) if wifi_energy else 0,
            np.max(bt_energy) if bt_energy else 0,
            np.std(bt_energy) if bt_energy else 0,
            (np.max(wifi_energy) / np.max(bt_energy)) if wifi_energy and bt_energy and np.max(bt_energy) > 0 else 0,
        ])
        wifi_corr, bt_corr = self._template_correlation(signal)
        feature_vector.extend([
            np.max(wifi_corr) if wifi_corr else 0,
            np.mean(wifi_corr) if wifi_corr else 0,
            np.max(bt_corr) if bt_corr else 0,
            np.mean(bt_corr) if bt_corr else 0,
        ])
        feature_vector.extend([
            np.mean(signal),
            np.std(signal),
            np.max(signal),
            np.min(signal),
            np.median(signal),
            np.percentile(signal, 25),
            np.percentile(signal, 75),
        ])
        return feature_vector

    def _energy_distribution(self, signal):
        """Compute energy in WiFi and Bluetooth windows."""
        bt_energy = []
        wifi_energy = []
        for start in range(0, len(signal) - self.wifi_width, self.wifi_width // 4):
            if start + self.wifi_width <= len(signal):
                wifi_window = signal[start : start + self.wifi_width]
                wifi_energy.append(np.sum(10 ** (wifi_window / 10)))
        for start in range(0, len(signal) - self.bt_width, self.bt_width // 2):
            if start + self.bt_width <= len(signal):
                bt_window = signal[start : start + self.bt_width]
                bt_energy.append(np.sum(10 ** (bt_window / 10)))
        return bt_energy, wifi_energy

    def _template_correlation(self, signal):
        """Compute correlation with WiFi and Bluetooth templates."""
        x_wifi = np.linspace(-10, 10, self.wifi_width)
        wifi_template = np.exp(-0.5 * x_wifi**2 / 9)
        x_bt = np.linspace(-10, 10, self.bt_width)
        bt_template = np.exp(-0.5 * x_bt**2)
        wifi_corr = []
        bt_corr = []
        for start in range(0, len(signal) - self.wifi_width, self.wifi_width // 2):
            if start + self.wifi_width <= len(signal):
                window_signal = signal[start : start + self.wifi_width]
                if np.std(window_signal) > 0:
                    corr = np.corrcoef(window_signal, wifi_template)[0, 1]
                    wifi_corr.append(corr)
        for start in range(0, len(signal) - self.bt_width, self.bt_width // 2):
            if start + self.bt_width <= len(signal):
                window_signal = signal[start : start + self.bt_width]
                if np.std(window_signal) > 0:
                    corr = np.corrcoef(window_signal, bt_template)[0, 1]
                    bt_corr.append(corr)
        return wifi_corr, bt_corr

    def _extract_bandwidth_features_direct(self, X):
        """
        Extract bandwidth features for each sample.

        Args:
            X (np.ndarray): Input PSD data (2D).

        Returns:
            np.ndarray: Feature matrix.
        """
        features = []
        for i in range(X.shape[0]):
            signal = X[i]
            feature_vector = self._extract_single_bandwidth(signal)
            features.append(feature_vector)
        return np.array(features)

    def _extract_single_bandwidth(self, signal):
        """
        Extract bandwidth features for a single PSD sample.

        Args:
            signal (np.ndarray): 1D PSD data.

        Returns:
            list: Feature vector.
        """
        feature_vector = []
        num_segments = 8
        segment_size = len(signal) // num_segments
        for j in range(num_segments):
            start = j * segment_size
            end = (j + 1) * segment_size
            segment = signal[start:end]
            peaks, properties = scipy_signal.find_peaks(
                segment, height=np.median(segment), distance=self.bt_width // 2
            )
            if len(peaks) == 0:
                feature_vector.extend([0, 0, 0, 0, 0])
                continue
            widths = scipy_signal.peak_widths(segment, peaks, rel_height=0.5)[0]
            bt_count = np.sum((widths > 5) & (widths < self.bt_width * 1.5))
            wifi_count = np.sum(
                (widths > self.bt_width * 2) & (widths < self.wifi_width * 1.2)
            )
            avg_width = np.mean(widths) if len(widths) > 0 else 0
            heights = properties["peak_heights"]
            hw_ratio = np.mean(heights / widths) if len(widths) > 0 else 0
            peak_energy = 0
            for p, w in zip(peaks, widths):
                left = max(0, int(p - w / 2))
                right = min(len(segment), int(p + w / 2))
                peak_region = segment[left:right]
                peak_energy += np.sum(10 ** (peak_region / 10))
            total_energy = np.sum(10 ** (segment / 10))
            energy_ratio = peak_energy / total_energy if total_energy > 0 else 0
            feature_vector.extend(
                [bt_count, wifi_count, avg_width, hw_ratio, energy_ratio]
            )
        return feature_vector

    def predict(self, X, noise_floor=-190.0):
        """
        Predict WiFi (0) or Bluetooth (1) from raw PSD data.

        Args:
            X (np.ndarray): Input PSD data.
            noise_floor (float): Value to use for masked (non-event) regions.

        Returns:
            np.ndarray: Predicted class labels (0=WiFi, 1=Bluetooth).
        """
        X_clean = X.copy()
        X_clean[X_clean == -np.inf] = noise_floor - 10
        X_processed, event_widths = self.preprocess_with_event_detection(
            X_clean, noise_floor
        )
        X_features = self.extract_features(X_processed)
        y_proba = self.model.predict_proba(X_features)[:, 1]
        y_pred = (y_proba > 0.5).astype(int)
        for i in range(len(y_pred)):
            width = event_widths[i]
            confidence = max(y_proba[i], 1 - y_proba[i])
            if confidence < 0.85:
                if width < self.width_threshold:
                    y_pred[i] = 1
                else:
                    y_pred[i] = 0
        return y_pred

    def predict_proba(self, X, noise_floor=-190.0):
        """
        Predict class probabilities with event detection preprocessing.

        Args:
            X (np.ndarray): Input PSD data.
            noise_floor (float): Value to use for masked (non-event) regions.

        Returns:
            np.ndarray: Class probabilities.
        """
        X_clean = X.copy()
        X_clean[X_clean == -np.inf] = noise_floor - 10
        X_processed, _ = self.preprocess_with_event_detection(X_clean, noise_floor)
        X_features = self.extract_features(X_processed)
        return self.model.predict_proba(X_features)
