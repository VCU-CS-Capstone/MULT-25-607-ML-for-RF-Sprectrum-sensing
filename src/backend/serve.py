import asyncio
import json
import re
import numpy as np
import torch
import websockets.asyncio.server
import websockets.exceptions
from MLRF.sources import H5DataSource
from websockets.asyncio.server import serve
from dotenv import load_dotenv
import os
import argparse
import joblib
from scipy import signal as scipy_signal
import sys

# Load environment variables from .env file
load_dotenv()


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


class RFClassifier:
    """RF Signal Classifier for WiFi vs Bluetooth with event detection and width-based classification"""

    def __init__(self, model, feature_method="combined", width_threshold=200):
        self.model = model
        self.feature_method = feature_method
        self.width_threshold = (
            width_threshold  # ~10 MHz in bins (threshold for BT vs WiFi)
        )
        # Define expected signal characteristics
        self.wifi_width = 410  # ~20MHz in bins
        self.bt_width = 20  # ~1MHz in bins

    def preprocess_with_event_detection(self, X, noise_floor=-190.0):
        """Preprocess data with event detection to isolate signal regions"""
        X_processed = X.copy()
        event_widths = []  # Store the widths of detected events for each sample

        if X.ndim == 1:
            X = X.reshape(1, -1)
            X_processed = X_processed.reshape(1, -1)

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

    def extract_features(self, X):
        """Extract features from raw PSD data"""
        # For real-time/single sample processing
        if X.ndim == 1:
            X = X.reshape(1, -1)

        if self.feature_method == "spectral_shape":
            return self._extract_spectral_shape_features_direct(X)
        elif self.feature_method == "bandwidth":
            return self._extract_bandwidth_features_direct(X)
        elif self.feature_method == "combined":
            spectral = self._extract_spectral_shape_features_direct(X)
            bandwidth = self._extract_bandwidth_features_direct(X)
            return np.hstack((spectral, bandwidth))
        else:
            # Default to combined features
            spectral = self._extract_spectral_shape_features_direct(X)
            bandwidth = self._extract_bandwidth_features_direct(X)
            return np.hstack((spectral, bandwidth))

    def _extract_spectral_shape_features_direct(self, X):
        """Direct feature extraction for small batches or single samples"""
        features = []

        for i in range(X.shape[0]):
            signal = X[i]
            feature_vector = []

            # Find peaks
            peaks, properties = scipy_signal.find_peaks(
                signal, height=np.median(signal), distance=self.bt_width // 2
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

                peak_heights = properties["peak_heights"]
                peak_prominences = scipy_signal.peak_prominences(signal, peaks)[0]

            # Calculate features
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

                feature_vector.extend(
                    [
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
                    ]
                )

                # Width histogram
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

                # Top 3 peak widths
                peak_indices = np.argsort(peak_prominences)[-3:]
                top_widths = [
                    widths_3db[i] if i < len(widths_3db) else 0 for i in peak_indices
                ]
                feature_vector.extend(top_widths)
            else:
                feature_vector.extend([0] * 12)  # Basic width statistics
                feature_vector.extend([0] * 7)  # Width histogram
                feature_vector.extend([0] * 3)  # Top peak widths

            # Energy distribution
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

            feature_vector.extend(
                [
                    np.max(wifi_energy) if wifi_energy else 0,
                    np.std(wifi_energy) if wifi_energy else 0,
                    np.max(bt_energy) if bt_energy else 0,
                    np.std(bt_energy) if bt_energy else 0,
                    (
                        np.max(wifi_energy) / np.max(bt_energy)
                        if wifi_energy and bt_energy and np.max(bt_energy) > 0
                        else 0
                    ),
                ]
            )

            # Template correlation
            x = np.linspace(-10, 10, self.wifi_width)
            wifi_template = np.exp(-0.5 * x**2 / 9)

            x = np.linspace(-10, 10, self.bt_width)
            bt_template = np.exp(-0.5 * x**2)

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

            feature_vector.extend(
                [
                    np.max(wifi_corr) if wifi_corr else 0,
                    np.mean(wifi_corr) if wifi_corr else 0,
                    np.max(bt_corr) if bt_corr else 0,
                    np.mean(bt_corr) if bt_corr else 0,
                ]
            )

            # Statistical features
            feature_vector.extend(
                [
                    np.mean(signal),
                    np.std(signal),
                    np.max(signal),
                    np.min(signal),
                    np.median(signal),
                    np.percentile(signal, 25),
                    np.percentile(signal, 75),
                ]
            )

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
                    segment, height=np.median(segment), distance=self.bt_width // 2
                )

                if len(peaks) == 0:
                    # No peaks in this segment
                    feature_vector.extend([0, 0, 0, 0, 0])
                    continue

                # Calculate peak widths
                widths = scipy_signal.peak_widths(segment, peaks, rel_height=0.5)[0]

                # Calculate bandwidth features for this segment
                bt_count = np.sum((widths > 5) & (widths < self.bt_width * 1.5))
                wifi_count = np.sum(
                    (widths > self.bt_width * 2) & (widths < self.wifi_width * 1.2)
                )

                # Average width
                avg_width = np.mean(widths) if len(widths) > 0 else 0

                # Peak height to width ratio
                heights = properties["peak_heights"]
                hw_ratio = np.mean(heights / widths) if len(widths) > 0 else 0

                # Energy in peak regions vs total energy
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

        return np.array(features)

    def predict(self, X, noise_floor=-190.0):
        """Predict WiFi (0) or Bluetooth (1) from raw PSD data with width-based adjustment"""
        # Replace -inf values with noise floor - 10
        X_clean = X.copy()
        X_clean[X_clean == -np.inf] = noise_floor - 10

        # Apply event detection preprocessing
        X_processed, event_widths = self.preprocess_with_event_detection(
            X_clean, noise_floor
        )

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
        X_processed, _ = self.preprocess_with_event_detection(X_clean, noise_floor)

        # Extract features
        X_features = self.extract_features(X_processed)

        # Make prediction
        return self.model.predict_proba(X_features)


# Data source factory
def create_data_source(source_type, data_path=None):
    """Create appropriate data source based on type"""
    if source_type.lower() == "h5":
        if not data_path:
            raise ValueError("H5 data source requires a data_path")
        return H5DataSource(data_path)
    elif source_type.lower() == "sdr":
        # Import only if SDR source is requested
        try:
            from MLRF.sources import USRPDataSource

            return USRPDataSource()
        except ImportError:
            print(
                "Error: USRPDataSource not available. Make sure USRP libraries are installed."
            )
            sys.exit(1)
    else:
        raise ValueError(f"Unknown source type: {source_type}. Use 'h5' or 'sdr'")


def run_inference(model, data: np.ndarray) -> int:
    """
    Run inference with ML classifier

    Args:
        model: Loaded ML model (RFClassifier)
        data: Input spectrum data (raw values)

    Returns:
        Classification result: 0 for WiFi, 1 for Bluetooth
    """
    data = np.array(data, dtype=np.float32)

    if os.getenv("DEBUG") == "True":
        print(f"Data range before inference: {np.min(data):.2f} to {np.max(data):.2f}")

    # Run inference with the ML model
    result = model.predict(data)[0]

    if os.getenv("DEBUG") == "True":
        print(f"Classification result: {'Bluetooth' if result == 1 else 'WiFi'}")

    return int(result)


async def handle_client(websocket, model, target_hz, source_type, data_path):
    """Handle WebSocket client connection"""
    target_period = 1.0 / target_hz  # Calculate period in seconds
    print(f"Client connected. Target frequency: {target_hz} Hz")
    if websocket.request and websocket.request.headers:
        print(f"Origin: {websocket.request.headers.get('Origin')}")

    # Initialize data source
    data_source = create_data_source(source_type, data_path)

    # Pre-allocate arrays for better performance
    detection_metrics = np.zeros(3, dtype=np.float32)

    # Track performance metrics
    iteration_count = 0
    frequency_check_interval = 5.0  # Check frequency every 5 seconds
    last_check_time = asyncio.get_event_loop().time()

    try:
        while True:
            loop_start_time = asyncio.get_event_loop().time()

            # Get the next data point
            data = data_source.get_next_data()

            # Process detections
            detections = event_detector(data)
            detection = 0
            if detections != [(0, 0)]:
                # Run inference
                detection = run_inference(model, data)
                detection += 1

                # Optional debug info
                if os.getenv("DEBUG") == "True":
                    signal_type = "WiFi" if detection == 1 else "Bluetooth"
                    print(
                        f"Detection: {signal_type}, Range: {np.min(data):.2f} to {np.max(data):.2f}"
                    )

            # Prepare and send data
            detection_metrics[0] = detection
            detection_metrics[1:] = detections[0]

            # Use numpy's concatenation for better performance
            data_with_detection = np.concatenate((data, detection_metrics))

            # Send as JSON
            await websocket.send(json.dumps(data_with_detection.tolist()))

            # Update metrics
            iteration_count += 1
            current_time = asyncio.get_event_loop().time()
            elapsed_since_check = current_time - last_check_time

            # Print actual frequency every few seconds
            if elapsed_since_check >= frequency_check_interval:
                actual_frequency = iteration_count / elapsed_since_check
                print(
                    f"Actual frequency: {actual_frequency:.2f} Hz (target: {target_hz} Hz)"
                )
                # Reset counters
                iteration_count = 0
                last_check_time = current_time

            # Sleep for the remaining time to maintain target frequency
            elapsed = asyncio.get_event_loop().time() - loop_start_time
            sleep_time = max(0, target_period - elapsed)
            await asyncio.sleep(sleep_time)

    except (
        websockets.exceptions.ConnectionClosedOK,
        websockets.exceptions.ConnectionClosedError,
    ):
        print("Client disconnected normally.")
    except asyncio.CancelledError:
        print("Connection cancelled, shutting down.")
    except Exception as e:
        print(f"Error handling client: {e}")
    finally:
        data_source.close()
        print("Closing WebSocket connection.")


async def run_server(host, port, model, target_hz, source_type, data_path):
    """Run WebSocket server with the given configuration"""
    trusted_origins_str = os.getenv("TRUSTED_ORIGINS", ".*")
    trusted_origins = re.compile(trusted_origins_str)

    async with serve(
        lambda ws: handle_client(ws, model, target_hz, source_type, data_path),
        host,
        port,
        origins=[trusted_origins],
    ):
        print(f"Server started on {host}:{port}")
        print(f"Using source: {source_type}")
        if source_type.lower() == "h5":
            print(f"Data path: {data_path}")
        print(f"Target frequency: {target_hz} Hz")
        print("Press Ctrl+C to stop.")
        await asyncio.Future()


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="RF Signal Classification Server")
    parser.add_argument("--port", type=int, default=3030, help="WebSocket server port")
    parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="WebSocket server host"
    )
    parser.add_argument(
        "--model", type=str, required=True, help="Path to saved model file"
    )
    parser.add_argument(
        "--hz", type=int, default=30, help="Target update frequency in Hz"
    )
    parser.add_argument(
        "--source",
        type=str,
        choices=["h5", "sdr"],
        default="h5",
        help="Data source type (h5 or sdr)",
    )
    parser.add_argument("--data_path", type=str, help="Path to H5 file for h5 source")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    return parser.parse_args()


def main():
    """Main entry point with command line argument handling"""
    args = parse_args()

    # Set DEBUG environment variable if --debug was provided
    if args.debug:
        os.environ["DEBUG"] = "True"

    # Validate args
    if args.source.lower() == "h5" and not args.data_path:
        print("Error: --data_path is required when using h5 source")
        sys.exit(1)

    # Load model
    print(f"Loading model from: {args.model}")
    try:
        # Check if model is a standalone model or a RFClassifier
        loaded_obj = joblib.load(args.model)

        if isinstance(loaded_obj, RFClassifier):
            model = loaded_obj
            print("Loaded RFClassifier model")
        else:
            # Create RFClassifier with the loaded model
            model = RFClassifier(loaded_obj, feature_method="combined")
            print("Loaded ML model and wrapped in RFClassifier")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    # Run the server
    try:
        asyncio.run(
            run_server(
                args.host, args.port, model, args.hz, args.source, args.data_path
            )
        )
    except KeyboardInterrupt:
        print("KeyboardInterrupt received: shutting down.")


if __name__ == "__main__":
    main()
