import stat
from typing import List, Tuple
import h5py
import numpy as np

class h5kit:
    def __init__(self, h5file: str) -> None:
        self.h5file: str = h5file
        
    def read(self, key: str | list | None = None) -> np.ndarray | dict:
        with h5py.File(self.h5file, 'r') as f:
            if isinstance(key, list):
                return np.array([f[k][:] for k in key])
            if key is None:
                return np.array([f[k][:] for k in f.keys()])
            return f[key][:]

    def write(self, key: str, data: np.ndarray) -> None:
        with h5py.File(self.h5file, 'a') as f:
            if key in f:
                del f[key]
            f.create_dataset(name=key, data=data, dtype=type(data[0]))

    def keys(self) -> List[str]:
        with h5py.File(self.h5file, 'r') as f:
            return list(f.keys())

    def shape(self, key: str) -> Tuple[int, ...]:
        with h5py.File(self.h5file, 'r') as f:
            return f[key].shape

    def close(self) -> None:
        pass

class psdkit:
    @staticmethod
    def _apply_window(data: np.ndarray) -> np.ndarray:
        """Apply a Hanning window to the data."""
        return data * np.hanning(len(data))

    @staticmethod
    def _compute_fft(windowed_data: np.ndarray, sample_rate: float) -> np.ndarray:
        """Compute the Power Spectral Density (PSD) using FFT."""
        length = len(windowed_data)
        return np.abs(np.fft.fft(windowed_data)) ** 2 / (length * sample_rate)
    @staticmethod
    def iq_to_psd(data: np.ndarray, sample_rate: float, center_frequency: float) -> np.ndarray:
        """
        Convert I/Q data to a Power Spectral Density (PSD) representation.

        Parameters:
            data: The I/Q data as a NumPy array (complex64).
            sample_rate: The sample rate of the signal.
            center_frequency: The center frequency of the signal.
            precision: Precision level (16, 32, or 64) for floating-point calculations.

        Returns:
            A NumPy array representing the PSD in dB.
        """
            
        windowed_data = psdkit._apply_window(data)
        psd = psdkit._compute_fft(windowed_data, sample_rate)
        return np.fft.fftshift(10.0 * np.log10(psd))

    @staticmethod
    def normalize(data: np.ndarray) -> np.ndarray:
        """Normalize data by dividing by 32768."""
        return data / 32768

    @staticmethod
    def deinterlace_iq(data: np.ndarray) -> np.ndarray:
        """Deinterlace I/Q data."""
        return data[0::2] + 1j * data[1::2]
