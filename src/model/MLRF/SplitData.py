import argparse
import os
import random
import shutil

def move_folders_by_pattern(source_folder, target_folder, pattern, percentage):
    folders = [
        d for d in os.listdir(source_folder)
        if d.startswith(pattern) and os.path.isdir(os.path.join(source_folder, d))
    ]
    if not folders:
        print(f"No folders found for pattern: {pattern}")
        return

    total_folders = len(folders)
    folders_to_move = int(total_folders * percentage / 100)
    selected_folders = random.sample(folders, folders_to_move)

    for folder in selected_folders:
        source_path = os.path.join(source_folder, folder)
        target_path = os.path.join(target_folder, folder)
        shutil.move(source_path, target_path)
        print(f"Moved folder: {folder}")

def main():
    parser = argparse.ArgumentParser(description='Move folders by pattern.')
    parser.add_argument('--source', default='./training_psd', help='Source folder')
    parser.add_argument('--target', default='./testing_psd', help='Target folder')
    parser.add_argument('--wifi_pct', type=int, default=30, help='Percentage of wifi folders to move')
    parser.add_argument('--bluetooth_pct', type=int, default=30, help='Percentage of bluetooth folders to move')
    args = parser.parse_args()

    os.makedirs(args.target, exist_ok=True)

    move_folders_by_pattern(args.source, args.target, 'wifi_', args.wifi_pct)
    move_folders_by_pattern(args.source, args.target, 'bluetooth_', args.bluetooth_pct)

if __name__ == '__main__':
    main()
