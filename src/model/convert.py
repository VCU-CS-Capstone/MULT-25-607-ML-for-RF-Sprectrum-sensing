from MLRF.datatools import h5kit
import os
import numpy as np
from tqdm import tqdm

path = './data/over_air'

files = os.listdir(path)

dataset = h5kit('overair.h5')


for file in tqdm(files):
    dataset.write(file, np.load(f"{path}/{file}"))