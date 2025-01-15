import numpy as np
import logging

logger = logging.getLogger(__name__)

class MockRtlSdr:
    def __init__(self):
        self.sample_rate = 2.4e6  # Hz
        self.center_freq = 2.45e9  # Hz
        self.freq_correction = 60    # PPM
        self.gain = 'auto'
        self.running = True
        logger.info("Initialized MockRtlSdr.")

    def read_samples(self, num_samples=512):
        # Generate synthetic IQ samples (e.g., sine wave with noise)
        t = np.arange(num_samples) / self.sample_rate
        freq = 1e5  # 100 kHz
        iq_samples = (np.cos(2 * np.pi * freq * t) + 1j * np.sin(2 * np.pi * freq * t)) + \
                     (np.random.normal(0, 0.1, num_samples) + 1j * np.random.normal(0, 0.1, num_samples))
        logger.debug(f"Generated {num_samples} mock IQ samples.")
        return iq_samples.astype(np.complex64)

    def close(self):
        self.running = False
        logger.info("Closed MockRtlSdr.")