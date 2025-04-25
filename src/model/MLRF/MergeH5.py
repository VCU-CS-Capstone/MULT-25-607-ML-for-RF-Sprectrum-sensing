"""
Merges multiple HDF5 files into a single output file using the correct
h5py File.copy() method for efficient copying.

All datasets from source files are copied directly to the root level
of the output file, with keys renamed to format 'wifi_uuid' or 'bluetooth_uuid'.
"""

import argparse
import h5py
import os
import sys
import time
import uuid  # For generating unique identifiers


def merge_hdf5_files(input_files, output_file, verbose=False):
    """
    Merges multiple HDF5 files into one, keeping all keys at the root level
    and renaming them to use UUIDs.

    Args:
        input_files (list): A list of paths to input HDF5 files.
        output_file (str): The path to the output HDF5 file.
        verbose (bool): If True, print detailed progress messages.
    """
    # Prevent overwriting an input file
    abs_input_paths = {os.path.abspath(f) for f in input_files}
    abs_output_path = os.path.abspath(output_file)
    if abs_output_path in abs_input_paths:
        print(
            f"Error: Output file '{output_file}' cannot be one of the input files.",
            file=sys.stderr,
        )
        sys.exit(1)

    if os.path.exists(output_file):
        print(
            f"Warning: Output file '{output_file}' already exists and will be overwritten.",
            file=sys.stderr,
        )
        # h5py 'w' mode truncates existing files

    start_time = time.time()
    print(f"Starting merge process into '{output_file}'...")

    try:
        # Open the destination file in write mode ('w')
        # This creates the file if it doesn't exist or truncates it if it does.
        with h5py.File(output_file, "w") as f_dest:
            print(f"Created/Opened output file: '{output_file}'")

            merged_keys = set()
            total_keys_merged = 0
            key_mappings = {}  # To track original key -> new key mappings

            for i, source_path in enumerate(input_files):
                file_start_time = time.time()
                abs_source_path = os.path.abspath(source_path)

                if not os.path.exists(abs_source_path):
                    print(
                        f"Warning: Input file '{source_path}' not found. Skipping.",
                        file=sys.stderr,
                    )
                    continue

                if not h5py.is_hdf5(abs_source_path):
                    print(
                        f"Warning: Input file '{source_path}' is not a valid HDF5 file. Skipping.",
                        file=sys.stderr,
                    )
                    continue

                try:
                    print(f"Processing file {i+1}/{len(input_files)}: '{source_path}'")
                    # Open the source file in read mode ('r')
                    with h5py.File(source_path, "r") as f_source:
                        keys_in_file = 0
                        # Iterate through all top-level keys in the source file
                        for key in f_source.keys():
                            # Parse the key to extract prefix (wifi or bluetooth)
                            if "_" in key:
                                prefix = key.split("_")[0]
                                if prefix in ["wifi", "bluetooth"]:
                                    # Generate a new UUID
                                    new_key = f"{prefix}_{uuid.uuid4()}"

                                    if verbose:
                                        print(f"  Renaming '{key}' to '{new_key}'")
                                else:
                                    # Not a wifi/bluetooth key, generate generic key
                                    new_key = f"item_{uuid.uuid4()}"

                                    if verbose:
                                        print(
                                            f"  Non-standard key '{key}' renamed to '{new_key}'"
                                        )
                            else:
                                # Key doesn't have an underscore, generate generic key
                                new_key = f"item_{uuid.uuid4()}"

                                if verbose:
                                    print(
                                        f"  Non-standard key '{key}' renamed to '{new_key}'"
                                    )

                            # Store the mapping
                            key_mappings[key] = new_key

                            # Copy the data with the new key name
                            f_source.copy(key, f_dest, name=new_key)
                            merged_keys.add(new_key)
                            keys_in_file += 1

                        total_keys_merged += keys_in_file
                        if verbose:
                            print(f"  Added {keys_in_file} keys from '{source_path}'")

                except Exception as e:
                    print(
                        f"Error processing file '{source_path}': {e}",
                        file=sys.stderr,
                    )
                    # Abort on error during copy to prevent partial/corrupt merge
                    raise  # Re-raise the exception to trigger the outer cleanup

                file_end_time = time.time()
                if verbose:
                    print(
                        f"  Finished processing '{source_path}' in {file_end_time - file_start_time:.2f} seconds."
                    )

            # Add attributes to the root of the merged file indicating completion
            f_dest.attrs["merge_timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S %Z")
            f_dest.attrs["merged_files"] = [
                os.path.basename(f)
                for f in input_files
                if os.path.exists(f) and h5py.is_hdf5(f)
            ]
            f_dest.attrs["total_keys_merged"] = total_keys_merged

            # Optionally save key mappings as an attribute
            if verbose:
                # Convert mappings dict to list of strings for storage
                mapping_strings = [f"{old}→{new}" for old, new in key_mappings.items()]
                try:
                    f_dest.attrs["key_mappings"] = mapping_strings
                except TypeError:
                    # Handle case where attributes can't store large string lists
                    print("Could not store key mappings as attribute (too large)")

    except Exception as e:
        print(f"\nAn error occurred during the merge process: {e}", file=sys.stderr)
        # Clean up potentially partially written output file if it exists
        if os.path.exists(output_file):
            try:
                os.remove(output_file)
                print(
                    f"Removed partially created or corrupt output file '{output_file}'."
                )
            except OSError as remove_err:
                print(
                    f"Error trying to remove incomplete output file '{output_file}': {remove_err}",
                    file=sys.stderr,
                )
        sys.exit(1)  # Exit with error status

    end_time = time.time()
    print(
        f"\nSuccessfully merged {total_keys_merged} keys from {len([f for f in input_files if os.path.exists(f) and h5py.is_hdf5(f)])} valid HDF5 file(s) into '{output_file}'"
    )
    print(f"Total time taken: {end_time - start_time:.2f} seconds.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge multiple HDF5 files into a single output file. "
        "All datasets from source files are copied directly to the root level "
        "of the output file with keys renamed to format 'wifi_uuid' or 'bluetooth_uuid'.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "input_files",
        metavar="INPUT_FILE",
        nargs="+",
        help="One or more input HDF5 files to merge.",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        dest="output_file",
        help="Path to the output HDF5 file. Will be overwritten if it exists.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Increase output verbosity.",
    )

    args = parser.parse_args()

    # Basic validation before starting
    if not args.output_file.endswith((".h5", ".hdf5", ".hdf", ".he5")):
        print(
            f"Warning: Output file '{args.output_file}' does not have a standard HDF5 extension.",
            file=sys.stderr,
        )

    merge_hdf5_files(args.input_files, args.output_file, args.verbose)

