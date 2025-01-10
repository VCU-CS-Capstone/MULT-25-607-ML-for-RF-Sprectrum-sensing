import uhd
import numpy as np
import numpy.typing as npt
import time
from scipy.signal import welch
import matplotlib.pyplot as plt
import keyboard
import threading

usrp = uhd.usrp.MultiUSRP()

num_samps = 1024 # number of samples received
#center_freq = 2425e6 # Hz
sample_rate = 40e6 # Hz
gain = 76 # dB

N = 1024
collects=100
threshold=-75
bandwidth=30e6
Fs = bandwidth


def receive_iq_data(center_freq):
    usrp.set_rx_rate(sample_rate, 0)
    usrp.set_rx_freq(uhd.libpyuhd.types.tune_request(center_freq), 0)
    usrp.set_rx_gain(gain, 0)
    usrp.set_rx_bandwidth(bandwidth, 0)

    # Set up the stream and receive buffer
    st_args = uhd.usrp.StreamArgs("fc32", "sc16")
    st_args.channels = [0]
    metadata = uhd.types.RXMetadata()
    streamer = usrp.get_rx_stream(st_args)
    recv_buffer = np.zeros((1, 1024), dtype=np.complex64)

    # Start Stream
    stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.start_cont)
    stream_cmd.stream_now = True
    streamer.issue_stream_cmd(stream_cmd)

    # Receive Samples
    samples = np.zeros(num_samps, dtype=np.complex64)
    for i in range(num_samps//1024):
        streamer.recv(recv_buffer, metadata)
        samples[i*1024:(i+1)*1024] = recv_buffer[0]

    # Stop Stream
    stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont)
    streamer.issue_stream_cmd(stream_cmd)

    #print(len(samples))
    #print(samples[0:10])
    return(samples)


def calculate_psd(x,center_freq):
    x = x * np.hamming(len(x)) # apply a Hamming window
    PSD = np.abs(np.fft.fft(x))**2 / (N*Fs)
    PSD_log = 10.0*np.log10(PSD)
    PSD_shifted = np.fft.fftshift(PSD_log)
    #center_freq = 2.45e9 # frequency we tuned our SDR to
    f = np.arange(Fs/-2.0, Fs/2.0, Fs/(N)) # start, stop, step.  centered around 0 Hz
    f += center_freq # now add center frequency
    return(PSD_shifted, f)
    
    
def event_detector(PSD_shifted):
    for i in range(len(PSD_shifted)):
                if PSD_shifted[i] >= threshold:
                    print('Pass')
                    #print(PSD_shifted[i])
                    return True
                
                    
    

def main():
    try:
        PSD= []
        for i in range(collects):
            iq_data = receive_iq_data(center_freq=2.4125e9)
            event, f = calculate_psd(iq_data, center_freq=2.4125e9)
            x=event
            frequency=f
            
            #print(len(frequency))
            #print(frequency)
            iq_data = receive_iq_data(center_freq=2.4375e9)
            event, f = calculate_psd(iq_data, center_freq=2.4375e9)
            x = np.concatenate((x,event))
            frequency = np.concatenate((frequency, f))
            
            iq_data = receive_iq_data(center_freq=2.4625e9)
            event, f = calculate_psd(iq_data, center_freq=2.4625e9)
            x = np.concatenate((x,event))
            frequency = np.concatenate((frequency, f))
            
            iq_data = receive_iq_data(center_freq=2.4875e9)
            event, f = calculate_psd(iq_data, center_freq=2.4875e9)
            x = np.concatenate((x,event))
            frequency = np.concatenate((frequency, f))
            #print(len(frequency))
            #print(frequency)
            
            if (event_detector(x)==True):
                 #print(len(frequency))
                 #print(len(x))
                 PSD.append(x)
                 plt.plot(frequency, x)
                 plt.show()
            
        np.save("psd_wifi_ch1.npy", PSD)

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
