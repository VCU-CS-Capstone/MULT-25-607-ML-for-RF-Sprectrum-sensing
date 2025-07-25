#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status.

# Define variables for common paths to improve readability and maintainability
UHD_DIR="$HOME/Desktop/UHD"
MLRF_DIR="$HOME/Desktop/MLRF"
UV_TARGET_BIN_DIR="$HOME/.local/bin" # The user-requested location for the uv executable, which is also its default install location

echo "--- Starting script execution ---"

echo "--- Installing system dependencies ---"
sudo apt-get update
sudo apt-get install -y autoconf automake build-essential ccache cmake cpufrequtils doxygen ethtool \
g++ git inetutils-tools libboost-all-dev libncurses-dev libusb-1.0-0 libusb-1.0-0-dev \
libusb-dev curl

echo "--- Installing uv for the current user ---"
curl -LsSf https://astral.sh/uv/install.sh | sh

mkdir -p "$UV_TARGET_BIN_DIR"
export PATH="$UV_TARGET_BIN_DIR:$PATH"
echo "PATH updated for current session: $PATH"

echo "--- Cloning repositories ---"
if [ ! -d "$UHD_DIR" ]; then
    echo "Cloning EttusResearch/uhd into $UHD_DIR"
    git clone https://github.com/EttusResearch/uhd "$UHD_DIR"
else
    echo "UHD directory already exists: $UHD_DIR. Skipping clone."
fi

if [ ! -d "$MLRF_DIR" ]; then
    echo "Cloning VCU-CS-Capstone/MULT-25-607-ML-for-RF-Sprectrum-sensing into $MLRF_DIR"
    git clone https://github.com/VCU-CS-Capstone/MULT-25-607-ML-for-RF-Sprectrum-sensing "$MLRF_DIR"
else
    echo "MLRF directory already exists: $MLRF_DIR. Skipping clone."
fi

echo "--- Setting up MLRF backend with uv ---"
cd "$MLRF_DIR/src/backend/"
uv sync

if [ -f ".venv/bin/activate" ]; then
    echo "Activating Python virtual environment."
    source .venv/bin/activate
else
    echo "Error: .venv/bin/activate not found. The virtual environment might not have been created correctly by uv sync."
    exit 1
fi

echo "--- Building UHD ---"
cd "$UHD_DIR/host/"
mkdir -p build
cd build
cmake -DENABLE_TESTS=OFF -DENABLE_C_API=OFF -DENABLE_PYTHON_API=ON -DENABLE_MANUAL=OFF ..
sudo make -j6

echo "--- Copying UHD Python modules to MLRF virtual environment ---"
UHD_PYTHON_BUILD_DIR="$UHD_DIR/host/build/python"
MLRF_VENV_SITE_PACKAGES_DIR="$MLRF_DIR/src/backend/.venv/lib/python3.11/site-packages"

mkdir -p "$MLRF_VENV_SITE_PACKAGES_DIR"

if [ -d "$UHD_PYTHON_BUILD_DIR/uhd" ]; then
    echo "Copying uhd module to $MLRF_VENV_SITE_PACKAGES_DIR"
    cp -r "$UHD_PYTHON_BUILD_DIR/uhd" "$MLRF_VENV_SITE_PACKAGES_DIR/"
else
    echo "Warning: uhd module not found at $UHD_PYTHON_BUILD_DIR/uhd. Skipping copy."
fi

if [ -d "$UHD_PYTHON_BUILD_DIR/usrp_mpm" ]; then
    echo "Copying usrp_mpm module to $MLRF_VENV_SITE_PACKAGES_DIR"
    cp -r "$UHD_PYTHON_BUILD_DIR/usrp_mpm" "$MLRF_VENV_SITE_PACKAGES_DIR/"
else
    echo "Warning: usrp_mpm module not found at $UHD_PYTHON_BUILD_DIR/usrp_mpm. Skipping copy."
fi

echo "--- Installing UHD system-wide ---"
sudo make install
sudo ldconfig 
sudo uhd_images_downloader

echo "--- Setting up udev rules for USRP devices ---"
cd /usr/local/lib/uhd/utils
sudo cp uhd-usrp.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules
sudo udevadm trigger

echo "--- Script finished successfully ---"

