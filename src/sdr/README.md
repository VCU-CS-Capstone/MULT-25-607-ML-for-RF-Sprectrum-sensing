# SDR:Data Collection

The **Data Collection Notebook** is intended only for asynchronous data collection for training and testing of the machine learning model as part of the [MLRF](../README.md) (Machine Learning Radio Frequency) project.  

---

## Getting Started

- This script and all information in this guide is based around using the **USRP B210 SDR**

### Prerequisites

- Recommended to follow (https://pysdr.org/content/usrp.html) for installation and setup of SDR
- Any bluetooth and Wi-Fi sources (these instructions contain specific information for the GL-AR300M16-EXT Router used in this project)
- Signal attenuator and spectrum analyzer

---

## Usage

- For the GL-AR300M16-EXT Router follow (https://docs.gl-inet.com/router/en/3/tutorials/file_sharing/) to setup file sharing
- The SDR's frontend has a max input of -15dBm
- Connect either emitter to a spectrum analyzer and through attenuation ensure the signal is below this threshold
- Connect emitter and attenuation to SDR
- Name your h5 file at top of main as a test file and set a low number of collects in global variable
- Uncomment plots and run the script
- Ensure the plots look as expected and match results on spectrum analyzer
- Recomment plots as this speeds up data collection
- Name h5 file appropriately for source and set a high number of collects and run file

---

## Troubleshooting
- **No Devices Found:**  
  Simply unplug and replug SDR
- Multiple debug statements are setup throughout the code to be able to check data is properly formatted at each step as well as timing
- Most issues can be resolved by examining the plots to understand what is happening with incoming data

---

## Requirements

- Python3
- UHD
- Linux (Ubuntu 22.04)
- Spectrum Analyzer
- Attenuator
- USRP B210 SDR
- Wi-Fi and Bluetooth emitters

---

## Contact

For questions, email [daniel.hartman1@gmail.com](mailto:daniel.hartman1@gmail.com). :D
