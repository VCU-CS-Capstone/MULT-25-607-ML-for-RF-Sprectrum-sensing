import argparse
import numpy as np
import os
from tqdm import tqdm
import sys

#!/usr/bin/env python3

def main():
    parser = argparse.ArgumentParser(description="Split .npy file by rows.")
    parser.add_argument('-i', '--input', type=str, required=True, help='Input file path')
    parser.add_argument('-o', '--output', type=str, default='output', help='Output directory')
    parser.add_argument('-p', '--prefix', type=str, default='data', help='Output file prefix')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    data = np.load(args.input)
    for i, row in tqdm(enumerate(data), desc="Saving data", colour="green"):
        out_file = os.path.join(args.output, f"{args.prefix}_{i}.npy")
        np.save(out_file, row.astype(np.float32))

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)
