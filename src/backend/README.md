# MLRF: Machine Learning for Radio Frequency Signal Classification

MLRF is a modular Python framework for **real-time RF signal classification** (e.g., WiFi vs Bluetooth) using machine learning.  
It supports both live SDR (USRP) and HDF5 file data sources, and serves results over a WebSocket API for easy integration with dashboards or other systems.

---

## Features

- **Real-time RF signal classification** (WiFi/Bluetooth) using ML
- Supports **USRP SDR** (via UHD) and **HDF5 file** data sources
- **WebSocket server** for streaming classification results
- **Event detection** and robust feature extraction

---

## Architecture

```
serve.py
  |
  |-- mlrf/
      |-- __init__.py      # Logging setup
      |-- sources.py       # USRPDataSource, H5DataSource
      |-- classifier.py    # Event detection, RFClassifier
      |-- server.py        # WebSocket server, data source selection
```

---

## Installation

1. **Clone the repository and install dependencies:**

   ```bash
   git clone https://github.com/yourusername/mlrf.git
   cd mlrf
   pip install -r requirements.txt
   ```

   - For SDR support, install UHD and USRP drivers (see [Ettus Research UHD](https://github.com/EttusResearch/uhd)).
   - For HDF5 support, `h5py` is required (included in requirements).
   - You can set environment variables:
     - `MLRF_LOG_LEVEL` (e.g., `DEBUG`, `INFO`)
     - `TRUSTED_ORIGINS` (regex for allowed WebSocket origins)

---

## Usage

2. **Start the Server**

   Using HDF5 data source:
   ```bash
   uv run serve.py --model models/MLRF_1.5.joblib --data_path data.h5
   ```

   Using SDR (USRP) data source:
   ```bash
   uv run serve.py --model models/MLRF_1.5.joblib --source sdr
   ```

   **Common options:**
   - `--port` (default: 3030)
   - `--host` (default: 0.0.0.0)
   - `--hz` (target update frequency, default: 30)
   - `--debug` (enable debug logging)

3. **Connect a WebSocket Client**

   The server streams JSON arrays:  
   `[psd_data..., detection, event_start, event_end]`

   - `detection`: 0 = no event, 1 = WiFi, 2 = Bluetooth
   - `event_start`, `event_end`: indices of detected event in the spectrum

---

## Logging

- Logging is configured in `mlrf/__init__.py`.
- Use `--debug` or set `MLRF_LOG_LEVEL=DEBUG` for verbose output.

---

## Requirements

- [UV](https://docs.astral.sh/uv/) 
- (Optional) UHD and USRP hardware for SDR support

---

## Troubleshooting

- **SDR not available:**  
  If UHD is not installed, only HDF5 data source is available.
- **Model loading errors:**  
  Ensure your model is compatible with scikit-learn and saved with `joblib`.
- **WebSocket connection issues:**  
  Check `TRUSTED_ORIGINS` and firewall settings.

---

## Contact

For questions, email [kushpatel169@gmail.com](mailto:kushpatel169@gmail.com). :D

