
import numpy as np
from scipy import signal as scipy_signal


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
            signal, height=np.median(signal), distance=bt_width // 2
        )

        if len(peaks) == 0:
            # Handle case with no peaks
            peak_widths = [0]
            peak_heights = [0]
            peak_prominences = [0]
        else:
            # Calculate widths at different relative heights
            widths_3db = scipy_signal.peak_widths(
                signal, peaks, rel_height=0.5
            )[0]
            widths_6db = scipy_signal.peak_widths(
                signal, peaks, rel_height=0.75
            )[0]
            widths_10db = scipy_signal.peak_widths(
                signal, peaks, rel_height=0.9
            )[0]

            # Get peak heights and prominences
            peak_heights = properties["peak_heights"]
            peak_prominences = scipy_signal.peak_prominences(signal, peaks)[0]

        # 2. Peak width statistics
        if len(peaks) > 0:
            bt_like_peaks = np.sum(
                (widths_3db > 5) & (widths_3db < bt_width * 2)
            )
            wifi_like_peaks = np.sum(
                (widths_3db > bt_width * 2) & (widths_3db < wifi_width * 1.5)
            )
            wide_peaks = np.sum(widths_3db > wifi_width)

            # Width/height ratio features
            width_height_ratios = widths_3db / peak_heights

            # Add basic width statistics
            feature_vector.extend([
                len(peaks),  # Number of peaks
                np.mean(widths_3db) if len(widths_3db) > 0 else 0,
                np.std(widths_3db) if len(widths_3db) > 0 else 0,
                np.mean(widths_6db) if len(widths_6db) > 0 else 0,
                np.std(widths_6db) if len(widths_6db) > 0 else 0,
                np.mean(widths_10db) if len(widths_10db) > 0 else 0,
                np.std(widths_10db) if len(widths_10db) > 0 else 0,
                bt_like_peaks,  # Number of Bluetooth-like peaks
                wifi_like_peaks,  # Number of WiFi-like peaks
                wide_peaks,  # Number of very wide peaks
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

            # Calculate peak width distributions
            width_bins = [
                0,
                bt_width / 2,
                bt_width,
                bt_width * 2,
                wifi_width / 2,
                wifi_width,
                wifi_width * 1.5,
                2048,
            ]
            width_hist, _ = np.histogram(widths_3db, bins=width_bins)
            feature_vector.extend(width_hist)

            # Top 3 peak widths (if available)
            peak_indices = np.argsort(peak_prominences)[-3:]
            top_widths = [
                widths_3db[i] if i < len(widths_3db) else 0
                for i in peak_indices
            ]
            feature_vector.extend(top_widths)
        else:
            # No peaks found - add placeholder values
            feature_vector.extend([0] * 12)  # Basic width statistics
            feature_vector.extend([0] * 7)  # Width histogram
            feature_vector.extend([0] * 3)  # Top peak widths

        # 3. Energy distribution in different bandwidths
        bt_energy = []
        wifi_energy = []

        for start in range(0, len(signal) - wifi_width, wifi_width // 4):
            if start + wifi_width <= len(signal):
                wifi_window = signal[start : start + wifi_width]
                wifi_energy.append(np.sum(10 ** (wifi_window / 10)))

        for start in range(0, len(signal) - bt_width, bt_width // 2):
            if start + bt_width <= len(signal):
                bt_window = signal[start : start + bt_width]
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

        # 4. Spectral shape template correlation
        x = np.linspace(-10, 10, wifi_width)
        wifi_template = np.exp(-0.5 * x**2 / 9)

        x = np.linspace(-10, 10, bt_width)
        bt_template = np.exp(-0.5 * x**2)

        wifi_corr = []
        bt_corr = []

        for start in range(0, len(signal) - wifi_width, wifi_width // 2):
            if start + wifi_width <= len(signal):
                window_signal = signal[start : start + wifi_width]
                if np.std(window_signal) > 0:
                    corr = np.corrcoef(window_signal, wifi_template)[0, 1]
                    wifi_corr.append(corr)

        for start in range(0, len(signal) - bt_width, bt_width // 2):
            if start + bt_width <= len(signal):
                window_signal = signal[start : start + bt_width]
                if np.std(window_signal) > 0:
                    corr = np.corrcoef(window_signal, bt_template)[0, 1]
                    bt_corr.append(corr)

        feature_vector.extend([
            np.max(wifi_corr) if wifi_corr else 0,
            np.mean(wifi_corr) if wifi_corr else 0,
            np.max(bt_corr) if bt_corr else 0,
            np.mean(bt_corr) if bt_corr else 0,
        ])

        # 5. Additional statistical features
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

        num_segments = 8
        segment_size = len(signal) // num_segments

        for j in range(num_segments):
            start = j * segment_size
            end = (j + 1) * segment_size
            segment = signal[start:end]

            peaks, properties = scipy_signal.find_peaks(
                segment, height=np.median(segment), distance=bt_width // 2
            )

            if len(peaks) == 0:
                feature_vector.extend([0, 0, 0, 0, 0])
                continue

            widths = scipy_signal.peak_widths(segment, peaks, rel_height=0.5)[
                0
            ]

            bt_count = np.sum((widths > 5) & (widths < bt_width * 1.5))
            wifi_count = np.sum(
                (widths > bt_width * 2) & (widths < wifi_width * 1.2)
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

        features.append(feature_vector)

    return features
