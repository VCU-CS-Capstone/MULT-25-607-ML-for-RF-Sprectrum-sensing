import os
import argparse
import sys
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

def parse_arguments():
    parser = argparse.ArgumentParser(description='IQ Data Split and Save Tool')
    parser.add_argument('-i', '--input', type=str, nargs='+', required=True, help='Input file paths')
    parser.add_argument('-o', '--output', type=str, default='output', help='Output directory')
    parser.add_argument('-p', '--prefix', type=str, default='data', help='Output file prefix')
    parser.add_argument('-f', '--center_freq', type=float, default=0.0, help='Center frequency')
    parser.add_argument('-t', '--threads', type=int, default=4, help='Number of processes')
    return parser.parse_args()

def apply_window(data):
    return data * np.hanning(len(data))

def compute_fft(windowed_data, sample_rate=30000000):
    length = len(windowed_data)
    return np.abs(np.fft.fft(windowed_data))**2 / (length * sample_rate)

def calculate_psd(input_data, center_freq):
    windowed_data = apply_window(input_data)
    psd = compute_fft(windowed_data)
    psd_log = 10.0 * np.log10(psd)
    return np.fft.fftshift(psd_log)

def create_output_directory(file_path, output_dir):
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    file_output_dir = os.path.join(output_dir, file_name)
    os.makedirs(file_output_dir, exist_ok=True)
    return file_output_dir

def save_segment(output_path, data):
    np.save(output_path, data.astype(np.float32))

def process_segments(data, file_path, prefix, file_output_dir, center_freq):
    """
    Removed tqdm here to avoid spamming from multiple processes.
    Just process each segment in a regular for loop.
    """
    num_samples = len(data) // 2048
    for idx in range(num_samples):
        start_idx = idx * 2048
        end_idx = start_idx + 2048
        segment = data[start_idx:end_idx]
        output = calculate_psd(segment, center_freq)
        output_path = os.path.join(file_output_dir, f"{prefix}_{idx}.npy")
        save_segment(output_path, output)
    return num_samples

def process_file(file_path, prefix, output_dir, center_freq):
    if prefix not in file_path:
        return f"Skipping {file_path}, does not match prefix '{prefix}'"
    try:
        file_output_dir = create_output_directory(file_path, output_dir)
        data = np.load(file_path)
        num_segments = process_segments(data, file_path, prefix, file_output_dir, center_freq)
        return f"Finished {file_path}, total segments: {num_segments}"
    except Exception as e:
        return f"Error processing {file_path}: {str(e)}"

def process_files_parallel(input_files, prefix, output_dir, center_freq, num_procs):
    results = []
    with ProcessPoolExecutor(max_workers=num_procs) as executor:
        future_to_file = {
            executor.submit(process_file, f, prefix, output_dir, center_freq): f
            for f in input_files
        }
        for future in tqdm(as_completed(future_to_file), total=len(input_files), desc="Overall Progress", unit="Files"):
            results.append(future.result())
    return results

def main():
    args = parse_arguments()
    results = process_files_parallel(
        args.input, args.prefix, args.output, args.center_freq, args.threads
    )
    for res in results:
        print(res)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)
