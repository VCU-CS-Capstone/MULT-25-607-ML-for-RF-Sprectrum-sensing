import argparse
import sys
import numpy as np
from tqdm import tqdm

def parse_arguments():
    parser = argparse.ArgumentParser(description='IQ Data Split and Save Tool')
    parser.add_argument('-i', '--input', type=str, required=True, help='Input file path')
    parser.add_argument('-o', '--output', type=str, default='output', help='Output directory')
    parser.add_argument('-p', '--prefix', type=str, default='data', help='Output file prefix')
    parser.add_argument('-f', '--center_freq', type=float, default=0.0, help='Center frequency')
    return parser.parse_args()

def calculate_psd(input_data, center_freq):
    input_data = input_data * np.hanning(len(input_data))
    length = len(input_data)
    sample_rate = 30000000 
    
    psd = np.abs(np.fft.fft(input_data))**2 / (length*sample_rate)
    psd_log = 10.0*np.log10(psd)
    psd_shifted = np.fft.fftshift(psd_log)

    # f = np.arange(Fs/-2.0, Fs/2, Fs/(N)) # start, stop, step.  centered around 0 Hz
    # f += center_freq # now add center frequency

    return psd_shifted

def main():
    args = parse_arguments()
    data = np.load(args.input)

    num_samples = len(data) // 2048
    for idx in tqdm(range(num_samples), desc="Saving data", colour="green"):
        start_idx = idx * 2048
        end_idx = start_idx + 2048
        segment = data[start_idx:end_idx]
        output = calculate_psd(segment, center_freq=args.center_freq)
        np.save(f"{args.output}/{args.prefix}_{idx}.npy", output.astype(np.float32))
    print(f"Total samples processed: {num_samples}")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)
