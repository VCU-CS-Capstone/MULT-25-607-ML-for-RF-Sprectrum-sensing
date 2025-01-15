import numpy as np
import os
import logging

logger = logging.getLogger(__name__)

class PSDSimulator:
    def __init__(self, sample_directory, sample_rate=2.4e6):
        self.sample_directory = sample_directory
        self.sample_rate = sample_rate
        self.files = sorted([f for f in os.listdir(sample_directory) if f.endswith('.npy')])
        self.current_file = 0
        self.running = True
        logger.info(f"Initialized PSDSimulator with directory: {sample_directory}")

    def read_samples(self, num_samples=512):
        if self.current_file >= len(self.files):
            self.current_file = 0  # Loop back or stop
            self.running = False  # Or set to True to loop indefinitely
            logger.info("Reached end of PSD sample files. Stopping PSDSimulator.")
            return np.array([], dtype=np.float32)

        if self.running:
            file_path = os.path.join(self.sample_directory, self.files[self.current_file])
            psd = np.load(file_path)
            self.current_file += 1
            logger.debug(f"Read PSD from {file_path}")
            return psd[:num_samples].astype(np.float32)
        else:
            return np.array([], dtype=np.float32)

    def close(self):
        self.running = False
        logger.info("Closed PSDSimulator.")