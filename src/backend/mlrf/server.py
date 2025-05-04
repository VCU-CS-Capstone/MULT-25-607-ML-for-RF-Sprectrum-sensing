"""
mlrf.server

Implements the WebSocket server and data source selection logic for RF classification.
"""

import asyncio
import json
import numpy as np
import os
import re
import sys
import websockets.exceptions
from .sources import H5DataSource, UHD_AVAILABLE
from .classifier import event_detector
from . import logger

def create_data_source(source_type, data_path=None):
    """
    Factory for creating a data source.

    Args:
        source_type (str): "h5" or "sdr"
        data_path (str): Path to H5 file (for h5 source)

    Returns:
        DataSource instance

    Raises:
        ValueError: If source_type is invalid or required data_path is missing.
    """
    if source_type.lower() == "h5":
        if not data_path:
            raise ValueError("H5 data source requires a data_path")
        logger.info(f"Creating H5 data source: {data_path}")
        return H5DataSource(data_path)
    elif source_type.lower() == "sdr":
        if not UHD_AVAILABLE:
            logger.error("UHD/USRP libraries not available. SDR source cannot be used.")
            sys.exit(1)
        from .sources import USRPDataSource
        logger.info("Creating SDR data source.")
        return USRPDataSource()
    else:
        raise ValueError(f"Unknown source type: {source_type}. Use 'h5' or 'sdr'")

def run_inference(model, data: np.ndarray) -> int:
    """
    Run inference with ML classifier.

    Args:
        model: Loaded ML model (RFClassifier)
        data: Input spectrum data (raw values)

    Returns:
        int: Classification result (0 for WiFi, 1 for Bluetooth)
    """
    data = np.array(data, dtype=np.float32)
    if os.getenv("DEBUG") == "True":
        logger.debug(f"Data range before inference: {np.min(data):.2f} to {np.max(data):.2f}")
    result = model.predict(data)[0]
    if os.getenv("DEBUG") == "True":
        logger.debug(f"Classification result: {'Bluetooth' if result == 1 else 'WiFi'}")
    return int(result)

async def handle_client(websocket, model, target_hz, source_type, data_path):
    """
    Handle a WebSocket client connection.

    Args:
        websocket: WebSocket connection object.
        model: RFClassifier instance.
        target_hz (float): Target update frequency.
        source_type (str): Data source type ("h5" or "sdr").
        data_path (str): Path to H5 file (if using h5 source).
    """
    target_period = 1.0 / target_hz
    logger.info(f"Client connected. Target frequency: {target_hz} Hz")
    if websocket.request and websocket.request.headers:
        logger.info(f"Origin: {websocket.request.headers.get('Origin')}")
    data_source = create_data_source(source_type, data_path)
    detection_metrics = np.zeros(3, dtype=np.float32)
    iteration_count = 0
    frequency_check_interval = 5.0
    last_check_time = asyncio.get_event_loop().time()
    
    # Add timing adjustment variables
    timing_adjustment = 0.0
    
    try:
        while True:
            loop_start_time = asyncio.get_event_loop().time()
            data = data_source.get_next_data()
            detections = event_detector(data)
            detection = 0
            if detections != [(0, 0)]:
                detection = run_inference(model, data)
                detection += 1
                if os.getenv("DEBUG") == "True":
                    signal_type = "WiFi" if detection == 1 else "Bluetooth"
                    logger.debug(
                        f"Detection: {signal_type}, Range: {np.min(data):.2f} to {np.max(data):.2f}"
                    )
            detection_metrics[0] = detection
            detection_metrics[1:] = detections[0]
            data_with_detection = np.concatenate((data, detection_metrics))
            await websocket.send(json.dumps(data_with_detection.tolist()))
            iteration_count += 1
            current_time = asyncio.get_event_loop().time()
            elapsed_since_check = current_time - last_check_time
            if elapsed_since_check >= frequency_check_interval:
                actual_frequency = iteration_count / elapsed_since_check
                logger.info(
                    f"Actual frequency: {actual_frequency:.2f} Hz (target: {target_hz} Hz)"
                )
                
                # Dynamic timing adjustment based on actual vs target frequency
                if actual_frequency < target_hz * 0.98:  # Too slow
                    timing_adjustment -= 0.001  # Reduce sleep time
                elif actual_frequency > target_hz * 1.02:  # Too fast
                    timing_adjustment += 0.001  # Increase sleep time
                
                logger.info(f"Timing adjustment: {timing_adjustment:.6f} seconds")
                
                iteration_count = 0
                last_check_time = current_time
            
            elapsed = asyncio.get_event_loop().time() - loop_start_time
            sleep_time = max(0, target_period - elapsed + timing_adjustment)
            await asyncio.sleep(sleep_time)
    except (
        websockets.exceptions.ConnectionClosedOK,
        websockets.exceptions.ConnectionClosedError,
    ):
        logger.info("Client disconnected normally.")
    except asyncio.CancelledError:
        logger.info("Connection cancelled, shutting down.")
    except Exception as e:
        logger.error(f"Error handling client: {e}")
    finally:
        data_source.close()
        logger.info("Closing WebSocket connection.")

async def run_server(host, port, model, target_hz, source_type, data_path):
    """
    Run the WebSocket server.

    Args:
        host (str): Host address.
        port (int): Port number.
        model: RFClassifier instance.
        target_hz (float): Target update frequency.
        source_type (str): Data source type ("h5" or "sdr").
        data_path (str): Path to H5 file (if using h5 source).
    """
    trusted_origins_str = os.getenv("TRUSTED_ORIGINS", ".*")
    trusted_origins = re.compile(trusted_origins_str)
    from websockets.asyncio.server import serve
    async with serve(
        lambda ws: handle_client(ws, model, target_hz, source_type, data_path),
        host,
        port,
        origins=[trusted_origins],
    ):
        logger.info(f"Server started on {host}:{port}")
        logger.info(f"Using source: {source_type}")
        if source_type.lower() == "h5":
            logger.info(f"Data path: {data_path}")
        logger.info(f"Target frequency: {target_hz} Hz")
        logger.info("Press Ctrl+C to stop.")
        await asyncio.Future()
