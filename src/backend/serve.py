import asyncio
import json
import re
import uhd
import h5py
import numpy as np
import torch
import torch.nn as nn
import websockets.asyncio.server
import websockets.exceptions
from websockets.asyncio.server import serve


class USRPDataSource:
    def __init__(
        self,
        center_freq=2.45e9,
        num_samps=1024,
        sample_rate=50e6,
        bandwidth=50e6,
        gain=80,
        buffer_size=0.1,
    ):
        """
        Initialize the USRP data source with configurable parameters

        Args:
            center_freq: Center frequency in Hz
            num_samps: Number of samples to collect
            sample_rate: Sample rate in Hz (reduced default)
            bandwidth: Bandwidth in Hz (reduced default)
            gain: Receiver gain (reduced default)
            buffer_size: Buffer size in seconds
        """
        # Store parameters as instance attributes

        self.center_freq = center_freq
        self.num_samps = num_samps
        self.sample_rate = sample_rate
        self.bandwidth = bandwidth
        self.gain = gain
        try:
            import uhd

            self.uhd = uhd  # Store module reference
            self.usrp = uhd.usrp.MultiUSRP()

            # Use lower sample rate to prevent overflows
            self.usrp.set_rx_rate(sample_rate, 0)
            self.usrp.set_rx_freq(uhd.libpyuhd.types.tune_request(center_freq), 0)
            self.usrp.set_rx_bandwidth(bandwidth, 0)
            self.usrp.set_rx_agc(True, 0)
            # self.usrp.set_rx_gain(gain, 0)

            # Set up the stream with larger buffer
            self.st_args = uhd.usrp.StreamArgs("fc32", "sc16")
            self.st_args.channels = [0]

            # Calculate buffer size in samples
            self.buffer_size_samps = int(sample_rate * buffer_size)

            # Set stream args with appropriate buffer size
            self.st_args.args = f"num_recv_frames={max(8, self.buffer_size_samps // 1024)},recv_frame_size=1024"
            self.streamer = self.usrp.get_rx_stream(self.st_args)

            # Maximum chunk size to read from USRP at once
            self.chunk_size = min(1024, self.num_samps)

            self.initialized = True
        except ImportError:
            print("Warning: UHD module not found. USRP functionality will be limited.")
            self.uhd = None
            self.usrp = None
            self.initialized = False

    def set_center_freq(self, center_freq):
        """Update the center frequency of the SDR"""
        if self.initialized:
            self.center_freq = center_freq
            self.usrp.set_rx_freq(self.uhd.libpyuhd.types.tune_request(center_freq), 0)
            return True
        return False

    def receive_iq_data(self, center_freq=None):
        """
        Receive IQ data from the USRP with overflow protection

        Args:
            center_freq: Optional center frequency override
        """
        if center_freq is not None:
            self.set_center_freq(center_freq)

        if not self.initialized:
            return np.zeros(self.num_samps, dtype=np.complex64)

        # Create metadata and receive buffer
        metadata = self.uhd.types.RXMetadata()
        recv_buffer = np.zeros((1, self.chunk_size), dtype=np.complex64)
        samples = np.zeros(self.num_samps, dtype=np.complex64)

        # Configure stream command for a single burst
        stream_cmd = self.uhd.types.StreamCMD(self.uhd.types.StreamMode.num_done)
        stream_cmd.num_samps = self.num_samps
        stream_cmd.stream_now = True
        self.streamer.issue_stream_cmd(stream_cmd)

        # Receive Samples with proper error handling
        num_rx_samps = 0
        timeout = 3.0  # seconds

        while num_rx_samps < self.num_samps:
            samples_to_recv = min(self.chunk_size, self.num_samps - num_rx_samps)

            try:
                rx_samps = self.streamer.recv(recv_buffer, metadata, timeout)

                if metadata.error_code != self.uhd.types.RXMetadataErrorCode.none:
                    # Handle the error based on the error code
                    if (
                        metadata.error_code
                        == self.uhd.types.RXMetadataErrorCode.overflow
                    ):
                        print("O", end="", flush=True)  # Indicate overflow
                    else:
                        print(f"Error: {metadata.error_code}")
                    continue

                if rx_samps == 0:
                    print("Timeout")
                    break

                samples[num_rx_samps : num_rx_samps + rx_samps] = recv_buffer[0][
                    :rx_samps
                ]
                num_rx_samps += rx_samps

            except Exception as e:
                print(f"Error receiving samples: {e}")
                break

        return samples

    @staticmethod
    def calculate_psd(x, center_freq):
        N = 2048
        Fs = 50e6
        x = x * np.hamming(len(x))  # apply a Hamming window
        PSD = np.abs(np.fft.fft(x)) ** 2 / (N * Fs)
        PSD_log = 10.0 * np.log10(PSD)  # Add small constant to prevent log(0)
        PSD_shifted = np.fft.fftshift(PSD_log)

        f = np.arange(
            Fs / -1, Fs, Fs / (N / 2)
        )  # start, stop, step.  centered around 0 Hz
        f += center_freq  # now add center frequency
        return (PSD_shifted, f)

    def get_next_data(self):
        """Get the next data point from USRP"""
        # Implementation for getting data from USRP would go here
        # This is a placeholder
        iq_data = self.receive_iq_data(center_freq=2.425e9)
        iq_data_2 = self.receive_iq_data(center_freq=2.475e9)

        iq_data_total = np.concatenate((iq_data, iq_data_2))
        psd_data, _ = self.calculate_psd(iq_data_total, center_freq=2.45e9)
        return psd_data

    def reset(self):
        """Reset the data source"""
        pass

    def close(self):
        """Close the USRP connection"""
        self.usrp = None


class H5DataSource:
    def __init__(self, filename="data.h5"):
        self.filename = filename
        self.current_idx = 0

        # Initialize H5 file
        self.h5_file = h5py.File(self.filename, "r")

        # Get dataset keys and sort them numerically
        dataset_keys = [
            k for k in self.h5_file.keys() if isinstance(self.h5_file[k], h5py.Dataset)
        ]
        self.keys = sorted(dataset_keys, key=lambda x: int(x))
        self.total_keys = len(self.keys)
        self.initialized = True

    def get_next_data(self):
        """Get the next data point from the H5 file"""
        if not self.total_keys:
            return np.array([])

        current_key = self.keys[self.current_idx]
        data = self.h5_file[current_key][()]

        # Move to next dataset circularly
        self.current_idx = (self.current_idx + 1) % self.total_keys

        return data

    def reset(self):
        """Reset the data source"""
        self.current_idx = 0

    def close(self):
        """Close the H5 file"""
        if self.h5_file:
            self.h5_file.close()
            self.h5_file = None


class SpectrumClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=51, padding=25),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(64, 128, kernel_size=25, padding=12),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(128, 256, kernel_size=11, padding=5),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.5), nn.Linear(128, 1)
        )
        self._init_weights()

    def _init_weights(self):
        """Initialize weights for Conv1d and Linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


torch.serialization.add_safe_globals([SpectrumClassifier])


def get_torch_device():
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    return device


device = get_torch_device()
model = torch.load("MLRF_1.1.pth", map_location=device, weights_only=False)
model.eval()


def run_inference(data: np.ndarray, min, max) -> int:
    PSD_MIN = min
    PSD_MAX = max
    data = np.array(data, dtype=np.float32)
    data = np.clip(data, PSD_MIN, PSD_MAX)
    data = (data - PSD_MIN) / (PSD_MAX - PSD_MIN)
    tensor = torch.tensor(data, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    tensor = tensor.to(device)

    with torch.no_grad():
        output = model(tensor)
        prob = torch.sigmoid(output).item()
    return 1 if prob > 0.5 else 0


async def handle_client(websocket: websockets.asyncio.server.ServerConnection) -> None:
    target_hz = 30  # Set your desired frequency here (e.g., 30 Hz)
    target_period = 1.0 / target_hz  # Calculate period in seconds
    print(f"Client connected. Target frequency: {target_hz} Hz")
    if websocket.request and websocket.request.headers:
        print(f"Origin: {websocket.request.headers.get('Origin')}")

    # Initialize data source
    data_source = USRPDataSource()

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
                detection = run_inference(data, np.min(data), np.max(data))
                detection += 1
                if detection == 1:
                    print("wifi")
                elif detection == 2:
                    print("bluetooth")

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


def event_detector(PSD, threshold_factor=1.05**-1):
    """
    Detect events in PSD data where signal exceeds a threshold relative to the signal mean

    Args:
        PSD: Power Spectral Density data array
        threshold_factor: Factor multiplied by mean to determine threshold (e.g., 1.5 = 50% above mean)

    Returns:
        List of tuples containing start and end indices of detected events
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


async def main():
    trusted_origins = re.compile(r".*")
    async with serve(
        handle_client,
        "0.0.0.0",
        3030,
        origins=[trusted_origins],
    ):
        print("Server started. Press Ctrl+C to stop.")
        print(f"Using device: {get_torch_device()}")
        await asyncio.Future()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("KeyboardInterrupt received: shutting down.")
