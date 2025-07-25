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

## Getting Started

### Prerequisites

- [UV](https://docs.astral.sh/uv/) (Package Manager, venv manager and more)
- For SDR support:
    - UHD (Universal Hardware Driver) and USRP drivers installed (see [Ettus Research UHD](https://github.com/EttusResearch/uhd)).
    - Compatible USRP hardware.
    - RUN THE SETUP.SH SCRIPT IN THE ROOT FIRST!!!

### Installation

1.  **Clone the repository:**
    ```bash
    # Assuming you have cloned the repository and navigated into its directory
    cd /path/to/mlrf
    ```

2.  **Install dependencies using UV:**
    ```bash
    uv sync
    ```
    *(This command installs dependencies based on `pyproject.toml` into a virtual environment managed by UV.)*

3.  **(Optional) Configure Environment Variables:**
    You can set environment variables before running:
    - `MLRF_LOG_LEVEL`: Controls logging verbosity (e.g., `DEBUG`, `INFO`). Defaults to `INFO`.
    - `TRUSTED_ORIGINS`: A regex pattern for allowed WebSocket origins (e.g., `http://localhost:3000`). Defaults to allowing any origin if not set.

---

## Usage

1.  **Start the Server using UV**

    Use `uv run` to execute the `serve.py` script within the managed environment.

    **Using HDF5 data source:**
    ```bash
    uv run serve.py --model models/MLRF_1.5.joblib --data_path data.h5
    ```
    *(Replace `data.h5` with your HDF5 file path)*

    **Using SDR (USRP) data source:**
    ```bash
    uv run serve.py --model models/MLRF_1.5.joblib --source sdr
    ```
    *(Requires UHD drivers and USRP hardware, see install guide)*

    **Common command-line options:**
    - `--model`: Path to the trained `.joblib` classification model (Required).
    - `--data_path`: Path to the HDF5 file (Required if `--source h5`).
    - `--source`: Data source type (`h5` or `sdr`, default: `h5`).
    - `--port`: WebSocket server port (default: 3030).
    - `--host`: Host address to bind the server to (default: `0.0.0.0`).
    - `--hz`: Target update frequency for processing and sending data (default: 30).
    - `--debug`: Enable debug logging (overrides `MLRF_LOG_LEVEL`). Logging is configured in `mlrf/__init__.py`.

2.  **Connect a WebSocket Client**

    Connect your frontend or WebSocket client to the address specified (e.g., `ws://localhost:3030`). The server streams JSON arrays containing:
    `[psd_data..., detection, event_start, event_end]`

    - `psd_data...`: Array of Power Spectral Density values.
    - `detection`: Integer representing the classification result (e.g., 0 = no event, 1 = WiFi, 2 = Bluetooth - defined in `classifier.py`).
    - `event_start`, `event_end`: Indices within the `psd_data` array indicating the start and end of the detected event, if any.

---

## Project Structure

```
serve.py             # Entry point (uv run serve.py ...)
  |
  |-- mlrf/
      |-- __init__.py      # Logging setup
      |-- sources.py       # USRPDataSource, H5DataSource
      |-- classifier.py    # Event detection, RFClassifier
      |-- server.py        # WebSocket server, data source selection
```
*(Note: `serve.py` is the top-level script executed by `uv run`)*

---

## Customization

-   **Server Configuration:** Use command-line arguments (`--port`, `--host`, `--hz`, etc.) or environment variables (`MLRF_LOG_LEVEL`, `TRUSTED_ORIGINS`).
-   **Data Sources:** Modify `mlrf/sources.py` to add support for different hardware or file formats.
-   **Classification Model:** Train and use your own scikit-learn compatible model saved with `joblib`. Specify the path using the `--model` argument.
-   **Detection Logic:** Adjust parameters or algorithms within `mlrf/classifier.py`.
-   **WebSocket Message Format:** Modify the data structure sent by the WebSocket in `mlrf/server.py`.

---

## Troubleshooting

-   **SDR Not Available / UHD Errors:**
    -   Ensure UHD drivers are correctly installed and accessible.
    -   Verify the USRP device is connected (`uhd_find_devices`).
    -   If UHD is not installed or intended, ensure you are using `--source h5`.
-   **Model Loading Errors:**
    -   Verify the `--model` path points to a valid `.joblib` file.
    -   Ensure the model was trained with compatible library versions.
-   **WebSocket Connection Issues:**
    -   Check if the server is running (`uv run ...`) and listening on the correct `--host` and `--port`.
    -   Verify firewall settings.
    -   Ensure the client uses the correct WebSocket URL.
    -   Check the `TRUSTED_ORIGINS` environment variable if connecting from a different origin.
-   **Dependency Problems:**
    -   Ensure `uv sync` completed successfully. Check the output for errors.
    -   Always use `uv run` to execute scripts, ensuring the correct environment is used.

---

## Contact

For questions, email [kushpatel169@gmail.com](mailto:kushpatel169@gmail.com). :D

---

**Happy hacking!**
