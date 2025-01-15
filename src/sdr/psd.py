import uhd
import numpy as np
import time
from scipy.signal import welch
import matplotlib.pyplot as plt

sdr = uhd.usrp.MultiUSRP() 

num_samps = 1024 # number of samples received
#center_freq = 2425e6 # Hz
sample_rate = 40e6 # Hz
gain = 60 # dB
Fs = 50e6
N = 1024
collects=1000
threshold=-80
bandwidth=50e6

def configure_sdr(center_freq=2.4e9):
    sdr.set_rx_rate(sample_rate, 0)
    sdr.set_rx_freq(uhd.libpyuhd.types.tune_request(center_freq), 0)
    sdr.set_rx_gain(gain, 0)
    sdr.set_rx_bandwidth(bandwidth, 0)

def receive_iq_data(num_samples=1024, duration=1.0):
    """
    Receives IQ data from the SDR.

    :param sdr: UHD USRP source object.
    :param num_samples: Number of samples to receive per buffer.
    :param duration: Duration of reception in seconds.
    :return: IQ data as a NumPy array.
    """
    stream_args = uhd.usrp.StreamArgs("fc32", "sc16")
    rx_streamer = sdr.get_rx_stream(stream_args)

    samples = np.zeros((num_samples,), dtype=np.complex64)
    metadata = uhd.types.RXMetadata()

    rx_streamer.issue_stream_cmd(
        uhd.types.StreamCMD(uhd.types.StreamMode.start_cont)
    )

    print("Receiving IQ data...")
    iq_data = []

    start_time = time.time()
    while time.time() - start_time < duration:
        num_received = rx_streamer.recv(samples, metadata)
        if metadata.error_code != uhd.types.RXMetadataErrorCode.none:
            print(f"Error receiving data: {metadata.error_code}")
            continue

        iq_data.append(samples[:num_received].copy())

    rx_streamer.issue_stream_cmd(
        uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont)
    )

    print("Reception complete.")
    return np.concatenate(iq_data)

def calculate_psd(iq_data, sample_rate):
    """
    Calculate the Power Spectral Density (PSD) from IQ data.

    :param iq_data: IQ samples as a NumPy array.
    :param sample_rate: Sampling rate in Hz.
    :return: Frequencies and PSD values.
    """
    frequencies, psd = welch(
        iq_data,
        fs=sample_rate*2,
        nperseg=256,
        return_onesided=False,
        scaling='density'
    )
    plt.semilogy(frequencies, psd)
    #plt.ylim([0.5e-3, 1])
    plt.xlabel('frequency [Hz]')
    plt.ylabel('PSD [V**2/Hz]')
    plt.show()

    return frequencies, psd

def plot_psd(frequencies, psd, center_freq):
    """
    Plot the Power Spectral Density (PSD).

    :param frequencies: Frequencies array.
    :param psd: PSD values.
    :param center_freq: Center frequency in Hz.
    """
    # Adjust frequencies to account for center frequency
    adjusted_frequencies = frequencies + center_freq

    plt.figure(figsize=(10, 6))
    plt.plot(adjusted_frequencies, 10 * np.log10(psd), label='psd')
    plt.title("Power Spectral Density (PSD)")
    plt.xlabel("Frequency (MHz)")
    plt.ylabel("Power Density (dB/Hz)")
    plt.grid(True)
    plt.legend()
    plt.show()

def main():
	try:
		PSD= []
		#for i in range(collects):
		# Create a USRP source
		sdr = uhd.usrp.MultiUSRP()
		
		# Configure the SDR
		sample_rate = 50e6
		center_freq = 2.425e9
		configure_sdr(center_freq=center_freq)

		# Receive IQ data for 1 second
		duration = 1.0
		iq_data = receive_iq_data(num_samples=1024, duration=duration)
		
		
		#Repeat at higher frequency
		
		# Configure the SDR
		sample_rate = 50e6
		center_freq = 2.475e9
		configure_sdr(center_freq=center_freq)
		
		# Receive IQ data for 1 second
		duration = 1.0
		iq_data_2 = receive_iq_data(num_samples=1024, duration=duration)

		# Save the IQ data to a file
		#np.save("iq_data.npy", iq_data)
		#print(f"IQ data saved to 'iq_data.npy'")
		
		iq_data_total=np.concatenate((iq_data,iq_data_2))

		# Calculate and plot PSD
		frequencies, psd = calculate_psd(iq_data_total, sample_rate)
		#plot_psd(frequencies, psd, center_freq=2.45e9)

	except Exception as e:
		print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
