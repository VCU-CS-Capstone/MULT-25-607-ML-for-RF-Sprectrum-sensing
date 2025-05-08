
import numpy as np
from scipy import signal as scipy_signal
from .utils import event_detector


class RFClassifier:
    """RF Signal Classifier for WiFi vs Bluetooth using LightGBM"""

    def __init__(self, model, feature_method="combined", width_threshold=200):
        self.model = model
        self.feature_method = feature_method
        self.width_threshold = width_threshold  # ~10 MHz in bins
        self.wifi_width = 410  # ~20MHz in bins
        self.bt_width = 20  # ~1MHz in bins

    def preprocess_signal(self, X, noise_floor):
        """Apply event detection and noise floor replacement"""
        X_processed = X.copy()
        event_widths = []

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

    def _extract_spectral_shape_features_direct(self, X):
        """Direct spectral shape feature extraction for model prediction"""
        features = []
        for i in range(X.shape[0]):
            signal = X[i]
            feature_vector = []
            peaks, properties = scipy_signal.find_peaks(
                signal, height=np.median(signal), distance=self.bt_width // 2
            )

            if len(peaks) == 0:
                peak_heights = [0] # Ensure defined even if no peaks
                peak_prominences = [0] # Ensure defined
                widths_3db = [0] # Ensure defined
            else:
                widths_3db = scipy_signal.peak_widths(
                    signal, peaks, rel_height=0.5
                )[0]
                widths_6db = scipy_signal.peak_widths(
                    signal, peaks, rel_height=0.75
                )[0]
                widths_10db = scipy_signal.peak_widths(
                    signal, peaks, rel_height=0.9
                )[0]
                peak_heights = properties["peak_heights"]
                peak_prominences = scipy_signal.peak_prominences(
                    signal, peaks
                )[0]

            if len(peaks) > 0:
                bt_like_peaks = np.sum(
                    (widths_3db > 5) & (widths_3db < self.bt_width * 2)
                )
                wifi_like_peaks = np.sum(
                    (widths_3db > self.bt_width * 2)
                    & (widths_3db < self.wifi_width * 1.5)
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
                    bt_like_peaks, wifi_like_peaks, wide_peaks,
                    (
                        np.mean(width_height_ratios)
                        if len(width_height_ratios) > 0
                        else 0
                    ),
                    (
                        np.max(width_height_ratios)
                        if len(width_height_ratios) > 0
                        else 0
                    ),
                ])
                width_bins = [
                    0, self.bt_width / 2, self.bt_width, self.bt_width * 2,
                    self.wifi_width / 2, self.wifi_width, self.wifi_width * 1.5, 2048,
                ]
                width_hist, _ = np.histogram(widths_3db, bins=width_bins)
                feature_vector.extend(width_hist)
                peak_indices = np.argsort(peak_prominences)[-3:]
                top_widths = [
                    widths_3db[i] if i < len(widths_3db) else 0
                    for i in peak_indices
                ]
                feature_vector.extend(top_widths)
            else:
                feature_vector.extend([0] * 12)
                feature_vector.extend([0] * 7)
                feature_vector.extend([0] * 3)

            bt_energy, wifi_energy = [], []
            for start in range(
                0, len(signal) - self.wifi_width, self.wifi_width // 4
            ):
                if start + self.wifi_width <= len(signal):
                    wifi_window = signal[start : start + self.wifi_width]
                    wifi_energy.append(np.sum(10 ** (wifi_window / 10)))
            for start in range(
                0, len(signal) - self.bt_width, self.bt_width // 2
            ):
                if start + self.bt_width <= len(signal):
                    bt_window = signal[start : start + self.bt_width]
                    bt_energy.append(np.sum(10 ** (bt_window / 10)))
            feature_vector.extend([
                np.max(wifi_energy) if wifi_energy else 0,
                np.std(wifi_energy) if wifi_energy else 0,
                np.max(bt_energy) if bt_energy else 0,
                np.std(bt_energy) if bt_energy else 0,
                (
                    np.max(wifi_energy) / np.max(bt_energy)
                    if wifi_energy and bt_energy and np.max(bt_energy) > 0
                    else 0
                ),
            ])

            x_wifi = np.linspace(-10, 10, self.wifi_width)
            wifi_template = np.exp(-0.5 * x_wifi**2 / 9)
            x_bt = np.linspace(-10, 10, self.bt_width)
            bt_template = np.exp(-0.5 * x_bt**2)
            wifi_corr, bt_corr = [], []
            for start in range(
                0, len(signal) - self.wifi_width, self.wifi_width // 2
            ):
                if start + self.wifi_width <= len(signal):
                    window_signal = signal[start : start + self.wifi_width]
                    if np.std(window_signal) > 0:
                        corr = np.corrcoef(window_signal, wifi_template)[0, 1]
                        wifi_corr.append(corr)
            for start in range(
                0, len(signal) - self.bt_width, self.bt_width // 2
            ):
                if start + self.bt_width <= len(signal):
                    window_signal = signal[start : start + self.bt_width]
                    if np.std(window_signal) > 0:
                        corr = np.corrcoef(window_signal, bt_template)[0, 1]
                        bt_corr.append(corr)
            feature_vector.extend([
                np.max(wifi_corr) if wifi_corr else 0,
                np.mean(wifi_corr) if wifi_corr else 0,
                np.max(bt_corr) if bt_corr else 0,
                np.mean(bt_corr) if bt_corr else 0,
            ])
            feature_vector.extend([
                np.mean(signal), np.std(signal), np.max(signal), np.min(signal),
                np.median(signal), np.percentile(signal, 25), np.percentile(signal, 75),
            ])
            features.append(feature_vector)
        return np.array(features)

    def _extract_bandwidth_features_direct(self, X):
        """Direct bandwidth feature extraction for model prediction"""
        features = []
        for i in range(X.shape[0]):
            signal = X[i]
            feature_vector = []
            num_segments = 8
            segment_size = len(signal) // num_segments
            for j in range(num_segments):
                start = j * segment_size
                end = (j + 1) * segment_size
                segment = signal[start:end]
                peaks, properties = scipy_signal.find_peaks(
                    segment,
                    height=np.median(segment),
                    distance=self.bt_width // 2,
                )
                if len(peaks) == 0:
                    feature_vector.extend([0, 0, 0, 0, 0])
                    continue
                widths = scipy_signal.peak_widths(
                    segment, peaks, rel_height=0.5
                )[0]
                bt_count = np.sum(
                    (widths > 5) & (widths < self.bt_width * 1.5)
                )
                wifi_count = np.sum(
                    (widths > self.bt_width * 2)
                    & (widths < self.wifi_width * 1.2)
                )
                avg_width = np.mean(widths) if len(widths) > 0 else 0
                heights = properties["peak_heights"]
                hw_ratio = (
                    np.mean(heights / widths)
                    if len(widths) > 0 and np.all(widths != 0) # Avoid division by zero
                    else 0
                )
                peak_energy = 0
                for p, w in zip(peaks, widths):
                    left = max(0, int(p - w / 2))
                    right = min(len(segment), int(p + w / 2))
                    peak_region = segment[left:right]
                    peak_energy += np.sum(10 ** (peak_region / 10))
                total_energy = np.sum(10 ** (segment / 10))
                energy_ratio = (
                    peak_energy / total_energy if total_energy > 0 else 0
                )
                feature_vector.extend([
                    bt_count, wifi_count, avg_width, hw_ratio, energy_ratio,
                ])
            features.append(feature_vector)
        return np.array(features)

    def extract_features(self, X):
        """Extract features for model prediction using direct methods"""
        if self.feature_method == "spectral_shape":
            return self._extract_spectral_shape_features_direct(X)
        elif self.feature_method == "bandwidth":
            return self._extract_bandwidth_features_direct(X)
        elif self.feature_method == "combined":
            spectral = self._extract_spectral_shape_features_direct(X)
            bandwidth = self._extract_bandwidth_features_direct(X)
            return np.hstack((spectral, bandwidth))
        else: # Default to combined
            spectral = self._extract_spectral_shape_features_direct(X)
            bandwidth = self._extract_bandwidth_features_direct(X)
            return np.hstack((spectral, bandwidth))

    def predict(self, X, noise_floor=-190.0):
        """Predict WiFi (0) or Bluetooth (1) from raw PSD data"""
        X_clean = X.copy()
        X_clean[X_clean == -np.inf] = noise_floor - 10
        X_processed, event_widths = self.preprocess_signal(
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
                    y_pred[i] = 1  # Bluetooth
                else:
                    y_pred[i] = 0  # WiFi
        return y_pred

    def predict_proba(self, X, noise_floor=-190.0):
        """Predict class probabilities with event detection preprocessing"""
        X_clean = X.copy()
        X_clean[X_clean == -np.inf] = noise_floor - 10
        X_processed, _ = self.preprocess_signal(X_clean, noise_floor)
        X_features = self.extract_features(X_processed)
        return self.model.predict_proba(X_features)
