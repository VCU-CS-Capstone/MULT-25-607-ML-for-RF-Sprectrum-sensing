import asyncio
import json
import re

import h5py
import numpy as np
import torch
import torch.nn as nn
import websockets.asyncio.server
import websockets.exceptions
from websockets.asyncio.server import serve


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


def run_inference(data: np.ndarray) -> int:
    PSD_MIN = -180
    PSD_MAX = -40
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
    sleep_time = 0.03
    print("Client connected.")
    if websocket.request and websocket.request.headers:
        print(f"Origin: {websocket.request.headers.get('Origin')}")

    with h5py.File("data.h5", "r") as h5_file:
        # Get dataset keys and sort them numerically
        dataset_keys = [
            k for k in h5_file.keys() if isinstance(h5_file[k], h5py.Dataset)
        ]
        keys = sorted(dataset_keys, key=lambda x: int(x))

        idx = 0
        try:
            while True:
                # Get the dataset and its data
                current_key = keys[idx]
                data = h5_file.get(current_key)[()]

                # Process detections
                detections = event_detector(data, -95)[0]
                detection = 0
                if detections != (0, 0):
                    detection = run_inference(data)
                    detection += 1

                # Prepare and send data
                data_array = np.array(data)
                data_with_detection = np.append(
                    data_array, [detection, detections[0], detections[1]]
                )

                await websocket.send(json.dumps(data_with_detection.tolist()))

                # Move to next dataset circularly
                idx = (idx + 1) % len(keys)
                await asyncio.sleep(sleep_time)

        except (
            websockets.exceptions.ConnectionClosedOK,
            websockets.exceptions.ConnectionClosedError,
        ):
            print("Client disconnected normally.")
        except asyncio.CancelledError:
            print("Connection cancelled, shutting down.")
        finally:
            print("Closing WebSocket connection.")


def detect_peaks(x, num_train, num_guard, rate_fa):
    num_cells = x.size
    num_train_half = round(num_train / 2)
    num_guard_half = round(num_guard / 2)
    num_side = num_train_half + num_guard_half

    alpha = num_train * (rate_fa ** (-1 / num_train) - 1)

    is_peak = np.zeros(num_cells, dtype=bool)

    for i in range(num_side, num_cells - num_side):
        if i != i - num_side + np.argmax(x[i - num_side : i + num_side + 1]):
            continue

        sum1 = np.sum(x[i - num_side : i + num_side + 1])
        sum2 = np.sum(x[i - num_guard_half : i + num_guard_half + 1])
        p_noise = (sum1 - sum2) / num_train
        threshold = alpha * p_noise

        if x[i] > threshold:
            is_peak[i] = True

    peak_ranges = []
    in_peak = False
    start_idx = 0

    for i in range(num_cells):
        if is_peak[i] and not in_peak:
            start_idx = i
            in_peak = True
        elif not is_peak[i] and in_peak:
            peak_ranges.append((start_idx, i - 1))
            in_peak = False

    if in_peak:
        peak_ranges.append((start_idx, num_cells - 1))

    return peak_ranges


def event_detector(PSD, threshold):
    window_size = 64
    event_ranges = []
    in_event = False
    start_idx = 0

    i = 0
    while i < len(PSD) - window_size + 1:
        window = PSD[i : i + window_size]
        window_average = round(sum(window) / window_size, 2)

        if window_average > threshold and not in_event:
            start_idx = i
            in_event = True
            i += window_size
        elif window_average <= threshold and in_event:
            event_ranges.append((start_idx, i - 1))
            in_event = False
            i += window_size
        else:
            i += window_size

    if in_event:
        event_ranges.append((start_idx, len(PSD) - 1))
    else:
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
        await asyncio.Future()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("KeyboardInterrupt received: shutting down.")
