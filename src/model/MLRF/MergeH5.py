from time import time_ns
from datatools import h5kit
import argparse
from tqdm import tqdm
import sys
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

def read_file(file, prefix):
    h5_file = h5kit(file)
    results = []
    for key in h5_file.keys():
        if prefix:
            new_key = f"{prefix}_{time_ns()}"
        else:
            new_key = key
        data = h5_file.read(key)
        results.append((new_key, data))
    # Use file size (in bytes) as an estimate for memory usage.
    file_size = os.path.getsize(file)
    return results, file_size

def main():
    parser = argparse.ArgumentParser(description="Combines h5 files concurrently in batches.")
    parser.add_argument('--input', type=str, required=True, nargs='+')
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--prefix', type=str, required=False)
    parser.add_argument('--batch_size_mb', type=int, default=500,
                        help="Maximum total file size in MB to load per batch.")
    parser.add_argument('--max_workers', type=int, default=4, help="Number of processes for reading.")
    args = parser.parse_args()

    threshold = args.batch_size_mb * 1024 * 1024  # convert MB to bytes
    input_files = args.input
    output_h5 = h5kit(args.output)
    pbar = tqdm(input_files, colour="green", desc="Merging Files")
    
    batch_tasks = []
    batch_mem = 0

    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        for file in pbar:
            try:
                file_size = os.path.getsize(file)
            except Exception as e:
                print(f"Error reading file size of {file}: {e}")
                continue

            future = executor.submit(read_file, file, args.prefix)
            batch_tasks.append(future)
            batch_mem += file_size

            # If current batch exceeds threshold, process this batch.
            if batch_mem >= threshold:
                for fut in as_completed(batch_tasks):
                    try:
                        results, _ = fut.result()
                        for new_key, data in results:
                            output_h5.write(new_key, data)
                    except Exception as e:
                        print(f"Error processing a file: {e}")
                batch_tasks = []
                batch_mem = 0

        # Process remaining tasks.
        for fut in as_completed(batch_tasks):
            try:
                results, _ = fut.result()
                for new_key, data in results:
                    output_h5.write(new_key, data)
            except Exception as e:
                print(f"Error processing a file: {e}")

    print("Done!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)