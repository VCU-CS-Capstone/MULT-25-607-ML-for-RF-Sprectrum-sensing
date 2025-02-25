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

PSD_MIN = -180
PSD_MAX = -40


class SpectrumClassifier(nn.Module):
    """CNN Architecture for Full Spectrum Analysis."""

    def __init__(self):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            # First layer looks for 2.5 MHz; (100/2048) ≈ 48.8 kHz resolution
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
    # print(f"Client \"{websocket.request.headers.get('Origin')}\" connected.")

    with h5py.File("data.h5", "r") as h5_file:
        dataset_keys = [
            k for k in h5_file.keys() if isinstance(h5_file[k], h5py.Dataset)
        ]
        keys = sorted(dataset_keys, key=lambda x: int(x))
        idx = 0
        try:
            while True:
                dataset = h5_file[keys[idx]]  # type: ignore[index]
                data = dataset[:]  # type: ignore[index]
                detection = run_inference(data)  # type: ignore[index]
                #
                data_with_detection = np.append(data, [detection + 1, 900 / 2, 1200 / 2])  # type: ignore[index]
                message = json.dumps(data_with_detection.tolist())
                await websocket.send(message)
                idx = (idx + 1) % len(keys)
                await asyncio.sleep(0.04)
        except (
            websockets.exceptions.ConnectionClosedOK,
            websockets.exceptions.ConnectionClosedError,
        ):
            print("Client disconnected normally.")
        except asyncio.CancelledError:
            print("Connection cancelled, shutting down.")
        finally:
            print("Closing WebSocket connection.")


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
