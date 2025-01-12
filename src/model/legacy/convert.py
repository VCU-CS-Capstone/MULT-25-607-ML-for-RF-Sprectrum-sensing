import h5py
import csv
import scipy.signal as signal
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

def hdf5_to_dataset(hdf5_file):
    """
    Load the IQ data from the HDF5 file
    """
    source_data = h5py.File(hdf5_file, "r")
    source_keys = list(source_data.keys())
    return source_data, source_keys


def parse_iq():
    hdf5_file = "./data/source_data/2_4ghz_bluetooth.h5"
    output = "./data/convert"
    data, keys = hdf5_to_dataset(hdf5_file)
    for idx, key in enumerate(keys, start=143):
        source_data = data[key][()]
        source_data = np.array(source_data)
        np.save(f"{output}/data_{idx}.npy", source_data)
        print(f"Saved data_{idx}.npy")
    csv_output = f"{output}/data_index.csv"
    with open(csv_output, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["idx", "wifi"])
        for idx, key in enumerate(keys, start=143):
            writer.writerow([idx, "bluetooth"])
    print(f"Saved data_index.csv")

class psd_dataset(Dataset):
    def __init__(self):
        self.data_path = "./data/training_data"
        self.output = "./data/training_data_psd"
    def __len__(self):
        return 284

    def __getitem__(self, idx):

        """
        # Compute the PSD of the input IQ data
        """
        data = np.load(f"{self.data_path}/data_{idx}.npy")
        freq, psd = signal.welch(
        data, fs=30000000, nperseg=2048, return_onesided=False
        )
        np.save(f"{self.output}/data_{idx}.npy", psd)
        print(f"Saved data_{idx}.npy")
        return torch.tensor([1])



def iq_to_psd():
    data_path = "./data/testing_data"
    output = "./data/testing_data_psd"
    """
    # Compute the PSD of the input IQ data
    """
    for i in range(0, 284):
        data = np.load(f"{data_path}/data_{i}.npy")
        freq, psd = signal.welch(
        data, fs=30000000, nperseg=2048, return_onesided=False
        )
        np.save(f"{output}/data_{i}.npy", psd)
        print(f"Saved data_{i}.npy")


if __name__ == "__main__":
    dataset = psd_dataset()
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    for i, _ in enumerate(loader):
        pass
