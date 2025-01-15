import argparse
import numpy as np
import os
from tqdm import tqdm
import sys

#!/usr/bin/env python3

def main():
    parser = argparse.ArgumentParser(description="Split .npy file by rows.")
    
    # Using nargs='+' to accept multiple files
    parser.add_argument('-i', '--input', type=str, required=True, nargs='+', help='List of input files')
    parser.add_argument('-o', '--output', type=str, default='output', help='Output directory')
    parser.add_argument('-p', '--prefix', type=str, default='data', help='Output file prefix')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    
    input_files = args.input
    
    if not input_files:
        print(f"No valid files provided in the input argument.")
        sys.exit(1)
    
    for input_file in input_files:
        if not os.path.isfile(input_file):
            print(f"Skipping invalid file: {input_file}")
            continue
        
        print(f"Processing file: {input_file}")
        data = np.load(input_file)
        
        for i, row in tqdm(enumerate(data), desc="Saving data", colour="green"):
            out_file = os.path.join(args.output, f"{args.prefix}_{os.path.basename(input_file)}_{i}.npy")
            np.save(out_file, row.astype(np.float32))
    
    print(f"Processing complete. Output saved to {args.output}")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)
