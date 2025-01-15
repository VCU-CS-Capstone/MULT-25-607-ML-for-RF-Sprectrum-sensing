#!/usr/bin/fish
uv run iq_split_save.py -i ./data/open_data/training_data/bluetooth_0.npy -o ./data/open_data/training_psd/ -p bluetooth
uv run iq_split_save.py -i ./data/open_data/training_data/wifi_0.npy -o ./data/open_data/training_psd/ -p wifi 
uv run train_test_split.py --source ./data/open_data/training_psd/ --target ./data/open_data/testing_psd/ 