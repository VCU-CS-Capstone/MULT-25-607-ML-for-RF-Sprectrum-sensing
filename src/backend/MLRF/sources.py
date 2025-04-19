# import uhd
import h5py
import numpy as np


# class USRPDataSource:
#     """
#     A data source that streams data from a Universal Software Radio Peripheral (USRP).
#
#     This class configures the USRP with specified parameters such as center frequency,
#     sample rate, and gain, and provides methods to receive IQ data, calculate the
#     Power Spectral Density (PSD), and manage the USRP connection.
#
#     Args:
#         center_freq (float): Center frequency in Hz.  Default: 2.45e9.
#         num_samps (int): Number of samples to collect per data point. Default: 1024.
#         sample_rate (float): Sample rate in Hz. Default: 50e6.
#         bandwidth (float): Bandwidth in Hz. Default: 50e6.
#         gain (int): Receiver gain in dB. Default: 80.
#         buffer_size (float): Buffer size in seconds.  Default: 0.1.
#     """
#
#     def __init__(
#         self,
#         center_freq=2.45e9,
#         num_samps=1024,
#         sample_rate=50e6,
#         bandwidth=50e6,
#         gain=80,
#         buffer_size=0.1,
#     ):
#         self.center_freq = center_freq
#         self.num_samps = num_samps
#         self.sample_rate = sample_rate
#         self.bandwidth = bandwidth
#         self.gain = gain
#
#         self.uhd = uhd
#         self.usrp = uhd.usrp.MultiUSRP()
#
#         self.usrp.set_rx_rate(sample_rate, 0)
#         self.usrp.set_rx_freq(uhd.libpyuhd.types.tune_request(center_freq), 0)
#         self.usrp.set_rx_bandwidth(bandwidth, 0)
#         self.usrp.set_rx_agc(True, 0)
#
#         self.st_args = uhd.usrp.StreamArgs("fc32", "sc16")
#         self.st_args.channels = [0]
#
#         self.buffer_size_samps = int(sample_rate * buffer_size)
#
#         self.st_args.args = (
#             f"num_recv_frames={max(8, self.buffer_size_samps // 1024)},"
#             "recv_frame_size=1024"
#         )
#         self.streamer = self.usrp.get_rx_stream(self.st_args)
#
#         self.chunk_size = min(1024, self.num_samps)
#         self.initialized = True
#
#     def set_center_freq(self, center_freq):
#         """Update the center frequency of the SDR."""
#         if self.initialized:
#             self.center_freq = center_freq
#             self.usrp.set_rx_freq(self.uhd.libpyuhd.types.tune_request(center_freq), 0)
#             return True
#         return False
#
#     def receive_iq_data(self, center_freq=None):
#         """
#         Receive IQ data from the USRP with overflow protection.
#
#         Args:
#             center_freq (float, optional):  Override the default center frequency.
#                 Defaults to None.
#
#         Returns:
#             numpy.ndarray: An array of complex64 IQ samples.
#         """
#         if center_freq is not None:
#             self.set_center_freq(center_freq)
#
#         if not self.initialized:
#             return np.zeros(self.num_samps, dtype=np.complex64)
#
#         # Create metadata and receive buffer
#         metadata = self.uhd.types.RXMetadata()
#         recv_buffer = np.zeros((1, self.chunk_size), dtype=np.complex64)
#         samples = np.zeros(self.num_samps, dtype=np.complex64)
#
#         # Configure stream command for a single burst
#         stream_cmd = self.uhd.types.StreamCMD(self.uhd.types.StreamMode.num_done)
#         stream_cmd.num_samps = self.num_samps
#         stream_cmd.stream_now = True
#         self.streamer.issue_stream_cmd(stream_cmd)
#
#         # Receive Samples with proper error handling
#         num_rx_samps = 0
#         timeout = 3.0  # seconds
#
#         while num_rx_samps < self.num_samps:
#             samples_to_recv = min(self.chunk_size, self.num_samps - num_rx_samps)
#
#             try:
#                 rx_samps = self.streamer.recv(recv_buffer, metadata, timeout)
#
#                 if metadata.error_code != self.uhd.types.RXMetadataErrorCode.none:
#                     # Handle the error based on the error code
#                     if (
#                         metadata.error_code
#                         == self.uhd.types.RXMetadataErrorCode.overflow
#                     ):
#                         print("O", end="", flush=True)  # Indicate overflow
#                     else:
#                         print(f"Error: {metadata.error_code}")
#                     continue
#
#                 if rx_samps == 0:
#                     print("Timeout")
#                     break
#
#                 samples[num_rx_samps : num_rx_samps + rx_samps] = recv_buffer[0][
#                     :rx_samps
#                 ]
#                 num_rx_samps += rx_samps
#
#             except Exception as e:
#                 print(f"Error receiving samples: {e}")
#                 break
#
#         return samples
#
#     @staticmethod
#     def calculate_psd(x, center_freq):
#         """
#         Calculate the Power Spectral Density (PSD) of a signal.
#
#         Args:
#             x (numpy.ndarray): Input signal.
#             center_freq (float): Center frequency of the signal.
#
#         Returns:
#             tuple: A tuple containing the shifted PSD in dB and the corresponding
#             frequency vector.
#         """
#         N = 2048
#         Fs = 50e6
#         x = x * np.hamming(len(x))  # apply a Hamming window
#         PSD = np.abs(np.fft.fft(x)) ** 2 / (N * Fs)
#         PSD_log = 10.0 * np.log10(PSD)  # Add small constant to prevent log(0)
#         PSD_shifted = np.fft.fftshift(PSD_log)
#
#         # Frequency vector centered around 0 Hz
#         f = np.linspace(-Fs / 2, Fs / 2, N, endpoint=False)
#         f += center_freq  # now add center frequency
#         return (PSD_shifted, f)
#
#     def get_next_data(self):
#         """
#         Get the next data point from USRP, combining data from two center frequencies.
#
#         Returns:
#             numpy.ndarray: PSD data calculated from combined IQ data.
#         """
#         iq_data = self.receive_iq_data(center_freq=2.425e9)
#         iq_data_2 = self.receive_iq_data(center_freq=2.475e9)
#
#         iq_data_total = np.concatenate((iq_data, iq_data_2))
#         psd_data, _ = self.calculate_psd(iq_data_total, center_freq=2.45e9)
#         return psd_data
#
#     def reset(self):
#         """Reset the data source (not implemented)."""
#         pass
#
#     def close(self):
#         """Close the USRP connection."""
#         self.usrp = None
#
#
class H5DataSource:
    """
    A data source that reads data from an H5 file.

    This class opens an H5 file and provides methods to iterate through the datasets
    within the file. It reads datasets sequentially, wrapping around to the beginning
    of the file after reaching the end.

    Args:
        filename (str):  The name of the H5 file to read.  Defaults to "data.h5".
    """

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
        """
        Get the next data point from the H5 file.

        Returns:
            numpy.ndarray: The next data point from the H5 file. Returns an empty
            array if the file is empty.
        """
        if not self.total_keys:
            return np.array([])

        current_key = self.keys[self.current_idx]
        data = self.h5_file[current_key][()]

        # Move to next dataset circularly
        self.current_idx = (self.current_idx + 1) % self.total_keys

        return data

    def reset(self):
        """Reset the data source to the beginning of the H5 file."""
        self.current_idx = 0

    def close(self):
        """Close the H5 file."""
        if self.h5_file:
            self.h5_file.close()
            self.h5_file = None
