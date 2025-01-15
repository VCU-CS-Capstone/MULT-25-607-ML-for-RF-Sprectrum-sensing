import numpy as np
import asyncio
import json
import logging
from app.utils.data_processing import normalize_psd
from app.utils.websocket_manager import WebSocketManager
from app.configs.settings import settings
from app.models.model_loader import load_model  # Ensure correct import

# Configure logging
logger = logging.getLogger(__name__)

try:
    if settings.use_mock_sdr:
        from app.sdr.mock_sdr import MockRtlSdr as RtlSdr
        logger.info("Using MockRtlSdr for simulation.")
    elif settings.use_file_sdr:
        from app.sdr.file_sdr import FileRtlSdr as RtlSdr
        logger.info("Using FileRtlSdr for simulation.")
    elif settings.use_psd_simulator:
        from app.sdr.psd_simulator import PSDSimulator as RtlSdr
        logger.info("Using PSDSimulator for simulation.")
    else:
        from pyrtlsdr import RtlSdr
        logger.info("Using real RtlSdr hardware.")
except ImportError as e:
    if settings.use_mock_sdr:
        from app.sdr.mock_sdr import MockRtlSdr as RtlSdr
        logger.error("Failed to import pyrtlsdr. Falling back to MockRtlSdr.")
    elif settings.use_file_sdr:
        from app.sdr.file_sdr import FileRtlSdr as RtlSdr
        logger.error("Failed to import pyrtlsdr. Falling back to FileRtlSdr.")
    elif settings.use_psd_simulator:
        from app.sdr.psd_simulator import PSDSimulator as RtlSdr
        logger.error("Failed to import pyrtlsdr. Falling back to PSDSimulator.")
    else:
        logger.critical("Failed to import pyrtlsdr and no simulation mode is enabled.")
        raise e

import torch

class SDRStream:
    def __init__(self, model, ws_manager: WebSocketManager, settings):
        self.model = model
        self.ws_manager = ws_manager
        self.settings = settings
        try:
            if settings.use_file_sdr:
                self.sdr = RtlSdr(sample_directory=settings.file_sdr_directory, sample_rate=settings.sdr_sample_rate)
            elif settings.use_psd_simulator:
                self.sdr = RtlSdr(sample_directory=settings.psd_simulator_directory, sample_rate=settings.sdr_sample_rate)
            else:
                self.sdr = RtlSdr()
            self.configure_sdr()
            logger.info("Initialized SDRStream.")
        except Exception as e:
            logger.error(f"Failed to initialize SDRStream: {e}")
            raise

    def configure_sdr(self):
        if not (self.settings.use_mock_sdr or self.settings.use_file_sdr or self.settings.use_psd_simulator):
            try:
                self.sdr.sample_rate = self.settings.sdr_sample_rate
                self.sdr.center_freq = self.settings.sdr_center_freq
                self.sdr.freq_correction = self.settings.sdr_freq_correction
                self.sdr.gain = self.settings.sdr_gain
                logger.info("Configured real RtlSdr settings.")
            except Exception as e:
                logger.error(f"Failed to configure SDR settings: {e}")
                raise

    async def start_stream(self):
        logger.info("Starting SDR stream.")
        while self.sdr.running:
            try:
                samples = self.sdr.read_samples(512)
                logger.debug(f"Read {len(samples)} samples from SDR.")
                if samples.size == 0:
                    logger.info("No more samples to read. Stopping stream.")
                    break  # No more samples to read

                # Pad IQ data to 8192 samples
                if len(samples) < 8192:
                    iq_padded = np.pad(samples, (0, 8192 - len(samples)), 'constant')
                    logger.debug(f"Padded IQ data from {len(samples)} to {len(iq_padded)} samples.")
                else:
                    iq_padded = samples[:8192]
                    logger.debug(f"Truncated IQ data to {len(iq_padded)} samples.")

                # Compute PSD
                psd_db = self.compute_psd(iq_padded)
                logger.debug(f"Computed PSD with shape: {psd_db.shape}")

                # Normalize PSD
                psd_norm = normalize_psd(psd_db)
                logger.debug(f"Normalized PSD with shape: {psd_norm.shape}")

                # Classify signal
                classification = self.classify_signal(psd_norm)
                logger.debug(f"Classification result: {classification}")

                # Prepare data packet
                data_packet = {
                    "psd": psd_norm.tolist(),
                    "classification": classification
                }

                # Broadcast to WebSocket clients
                await self.ws_manager.broadcast(json.dumps(data_packet))
                logger.debug("Broadcasted data packet to WebSocket clients.")

                await asyncio.sleep(0.1)  # Adjust as needed
            except Exception as e:
                logger.error(f"Error in SDR stream: {e}")
                break  # Exit the streaming loop on error

        logger.info("SDR stream ended.")

    def compute_psd(self, iq_data):
        fft_size = 8192  # Desired FFT size
        original_length = len(iq_data)
        window = np.hamming(original_length)
        logger.debug(f"Original IQ data length: {original_length}")
        logger.debug(f"Window length: {len(window)}")

        # Padding or truncating IQ data to match fft_size
        if original_length < fft_size:
            iq_padded = np.pad(iq_data, (0, fft_size - original_length), 'constant')
            logger.debug(f"Padded IQ data from {original_length} to {len(iq_padded)} samples.")
        elif original_length > fft_size:
            iq_padded = iq_data[:fft_size]
            logger.debug(f"Truncated IQ data from {original_length} to {len(iq_padded)} samples.")
        else:
            iq_padded = iq_data
            logger.debug("No padding or truncation needed for IQ data.")

        # Re-initialize window after padding/truncation
        window = np.hamming(len(iq_padded))
        logger.debug(f"New window length after padding/truncation: {len(window)}")

        # Apply window
        try:
            windowed_iq = iq_padded * window
            logger.debug(f"Windowed IQ data shape: {windowed_iq.shape}")
        except ValueError as e:
            logger.error(f"Error applying window: {e}")
            raise

        # Compute FFT and PSD
        fft_result = np.fft.fftshift(np.fft.fft(windowed_iq))
        psd = np.abs(fft_result) ** 2
        psd_db = 10 * np.log10(psd + 1e-10)  # dB scale
        logger.debug(f"PSD shape: {psd_db.shape}")
        return psd_db

    def classify_signal(self, psd_norm):
        if len(psd_norm) != 8192:
            logger.error(f"Expected PSD with 8192 features, but got {len(psd_norm)}")
            raise ValueError(f"Expected PSD with 8192 features, but got {len(psd_norm)}")
        try:
            input_tensor = torch.tensor(psd_norm, dtype=torch.float32).unsqueeze(0)  # [1, 8192]
            logger.debug(f"Input tensor shape: {input_tensor.shape}")
            with torch.no_grad():
                output = self.model(input_tensor)
                _, predicted = torch.max(output.data, 1)
                logger.debug(f"Model output: {output}")
                return "Bluetooth" if predicted.item() == 1 else "WiFi"
        except Exception as e:
            logger.error(f"Error during signal classification: {e}")
            raise

    def close(self):
        try:
            self.sdr.close()
            logger.info("Closed SDRStream.")
        except Exception as e:
            logger.error(f"Error closing SDRStream: {e}")
