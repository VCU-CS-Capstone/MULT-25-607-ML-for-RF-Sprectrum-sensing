import argparse
import sys
import random
from datatools import h5kit
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="Split an .h5 dataset into training and testing sets.")
    parser.add_argument("--input", required=True, help="Input .h5 file")
    parser.add_argument("--train_output", required=True, help="Output training .h5 file")
    parser.add_argument("--test_output", required=True, help="Output testing .h5 file")
    parser.add_argument("--test_pct", type=int, default=20, help="Percentage of data for test set")
    args = parser.parse_args()

    input_h5 = h5kit(args.input)
    keys = list(input_h5.keys())
    random.shuffle(keys)

    test_size = int(len(keys) * args.test_pct / 100)
    test_keys = keys[:test_size]
    train_keys = keys[test_size:]

    train_h5 = h5kit(args.train_output)
    test_h5 = h5kit(args.test_output)

    for key in tqdm(test_keys):
        test_h5.write(key, input_h5.read(key))
    for key in tqdm(train_keys):
        train_h5.write(key, input_h5.read(key))

    print(f"Training set: {len(train_keys)} items, Testing set: {len(test_keys)} items.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)
 