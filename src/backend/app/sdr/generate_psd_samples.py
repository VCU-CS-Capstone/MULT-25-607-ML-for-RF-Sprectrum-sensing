import numpy as np
import os

def generate_psd_samples(directory, num_samples=100, sample_length=512):
    """
    Generates synthetic PSD data and saves them as .npy files.

    Parameters:
    - directory (str): Path to save the PSD samples.
    - num_samples (int): Number of PSD samples to generate.
    - sample_length (int): Number of frequency bins in each PSD sample.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

    for i in range(num_samples):
        # Simulate PSD data
        # Generate a base noise PSD
        noise = np.random.normal(loc=0.0, scale=1.0, size=sample_length)

        # Simulate signal presence
        if i % 2 == 0:
            # Even-indexed samples simulate WiFi signals
            # Add a peak at a specific frequency bin
            peak_bin = np.random.randint(100, 200)
            noise[peak_bin:peak_bin+5] += 5  # Add a sharp peak
            label = "WiFi"
        else:
            # Odd-indexed samples simulate Bluetooth signals
            # Add a peak at a different frequency bin
            peak_bin = np.random.randint(300, 400)
            noise[peak_bin:peak_bin+5] += 5  # Add a sharp peak
            label = "Bluetooth"

        # Ensure PSD is positive
        psd = np.abs(noise)

        # Save PSD data as .npy file
        file_path = os.path.join(directory, f'psd_{i}.npy')
        np.save(file_path, psd)

        print(f"Generated {label} PSD sample: {file_path}")

if __name__ == "__main__":
    generate_psd_samples(directory='sample_data', num_samples=100, sample_length=512)
    print("PSD sample generation complete.")
