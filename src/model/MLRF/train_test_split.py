#!/usr/bin/env python3

import argparse
import os
import random
import shutil

def move_files_by_pattern(source_folder, target_folder, pattern, percentage):
    files = [f for f in os.listdir(source_folder) 
             if f.startswith(pattern) and f.endswith('.npy')]
    if not files:
        print(f"No files found for pattern: {pattern}")
        return
    
    total_files = len(files)
    files_to_move = int(total_files * percentage / 100)
    selected_files = random.sample(files, files_to_move)

    for file in selected_files:
        source_path = os.path.join(source_folder, file)
        target_path = os.path.join(target_folder, file)
        shutil.move(source_path, target_path)
        print(f"Moved: {file}")

def main():
    parser = argparse.ArgumentParser(description='Move files by pattern.')
    parser.add_argument('--source', default='./training_psd', help='Source folder')
    parser.add_argument('--target', default='./testing_psd', help='Target folder')
    parser.add_argument('--wifi_pct', type=int, default=30, help='Percentage of wifi files to move')
    parser.add_argument('--bluetooth_pct', type=int, default=30, help='Percentage of bluetooth files to move')
    args = parser.parse_args()

    os.makedirs(args.target, exist_ok=True)

    move_files_by_pattern(args.source, args.target, 'wifi_', args.wifi_pct)
    move_files_by_pattern(args.source, args.target, 'bluetooth_', args.bluetooth_pct)

if __name__ == '__main__':
    main()
