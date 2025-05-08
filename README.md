# MLRF: Machine Learning for Radio Frequency Signal Classification

**MLRF** is a modular Python framework for **real-time RF signal classification** (e.g., WiFi vs Bluetooth) using machine learning.  
It supports both live SDR (USRP) and HDF5 file data sources, and serves results over a WebSocket API for easy integration with dashboards or other systems.

This repository is organized to be fully self-contained, following the [Linux Foundation OMP Mentorship program](https://github.com/lf-omp) guidelines.  
It includes not only code, but also documentation, research, deliverables, and project management resources to support current and future contributors.

---

## Project Structure

```
.
├── Documentation/
├── Project Deliverables/
├── Research/
├── Status Reports/
├── src/
│   ├── backend/
│   │   ├── mlrf/
│   │   ├── models/
│   ├── frontend/
│   │   ├── public/
│   │   ├── src/
│   ├── model/
│   │   ├── data/
│   │   ├── MLRF/
│   └── sdr/
│       ├── 1_17_testing/
└── README.md
```

- **Documentation/**: Architecture, design, installation, and configuration guides.
- **Project Deliverables/**: Final PDF versions of all major deliverables.
- **Research/**: Research notes, references, and background material.
- **Status Reports/**: Weekly reports, milestones, and project management docs.
- **src/**: All source code and related resources.
    - **backend/**: Backend server and RF classification logic.
    - **frontend/**: React-based dashboard and visualization code.
    - **model/**: Model training, evaluation, and metadata.
    - **sdr/**: SDR data collection and processing scripts.

---

## Features

- **Real-time RF signal classification** (WiFi/Bluetooth) using ML
- Supports **USRP SDR** (via UHD) and **HDF5 file** data sources
- **WebSocket server** for streaming classification results
- **Event detection** and robust feature extraction
- **Frontend dashboard** for real-time visualization and control (React + Material-UI)
- **Comprehensive documentation** and project management resources

---

## Getting Started

### Prerequisites

- **Backend:**
  - Python 3.11
  - [UV](https://docs.astral.sh/uv/) (Python package & venv manager)
  - For SDR support:
    - UHD (Universal Hardware Driver) and USRP drivers installed (see [Ettus Research UHD](https://github.com/EttusResearch/uhd))
    - Compatible USRP hardware

- **Frontend:**
  - [Node.js](https://nodejs.org/) (v16 or newer recommended)
  - [pnpm](https://pnpm.io/) (install with `npm install -g pnpm`)

---

## Installation

### Backend

1. **Install backend dependencies using UV:**
    ```bash
    cd src/backend
    uv sync
    ```

2. **(Optional) Configure Environment Variables:**
    - `MLRF_LOG_LEVEL`: Controls logging verbosity (e.g., `DEBUG`, `INFO`). Defaults to `INFO`.
    - `TRUSTED_ORIGINS`: Regex pattern for allowed WebSocket origins (e.g., `http://localhost:3000`). Defaults to allowing any origin if not set.

### Frontend

1. **Install frontend dependencies using pnpm:**
    ```bash
    cd src/frontend
    pnpm install
    ```

---

## Usage

### Backend (RF Classification Server)

1. **Start the Server using UV**

    Use `uv run` to execute the `serve.py` script within the managed environment.

    **Using HDF5 data source:**
    ```bash
    uv run serve.py --model models/MLRF_1.5.joblib --data_path models/data.h5
    ```

    **Using SDR (USRP) data source:**
    ```bash
    uv run serve.py --model models/MLRF_1.5.joblib --source sdr
    ```

    **Common command-line options:**
    - `--model`: Path to the trained `.joblib` classification model (Required).
    - `--data_path`: Path to the HDF5 file (Required if `--source h5`).
    - `--source`: Data source type (`h5` or `sdr`, default: `h5`).
    - `--port`: WebSocket server port (default: 3030).
    - `--host`: Host address to bind the server to (default: `0.0.0.0`).
    - `--hz`: Target update frequency for processing and sending data (default: 30).
    - `--debug`: Enable debug logging (overrides `MLRF_LOG_LEVEL`).

2. **WebSocket Output**

    The backend streams JSON arrays to connected clients:
    ```
    [psd_data..., detection, event_start, event_end]
    ```
    - `psd_data...`: Array of Power Spectral Density values.
    - `detection`: Integer representing the classification result (e.g., 0 = no event, 1 = WiFi, 2 = Bluetooth).
    - `event_start`, `event_end`: Indices within the `psd_data` array indicating the start and end of the detected event, if any.

---

### Frontend (Dashboard)

1. **Start the frontend development server:**
    ```bash
    cd src/frontend
    pnpm start
    ```
    The app will open at [http://localhost:3000](http://localhost:3000).

2. **Usage Notes:**
    - The frontend expects a WebSocket backend running at `ws://localhost:3030`.
    - Use the **Connect** button to start receiving data.
    - Adjust the **Signal Range** and **Pixel Size** sliders to tune the visualization.
    - Use the **Clear** button to reset the spectrogram.
    - Click the **Fullscreen** button for an immersive view; controls will appear as an overlay in fullscreen.
    - Error messages and connection status are displayed in the UI.

---

## Customization

- **Backend:**
  - Server configuration via command-line or environment variables.
  - Data sources: extend in `src/backend/mlrf/sources.py`.
  - Classification model: use your own scikit-learn compatible `.joblib` model.
  - Detection logic: adjust in `src/backend/mlrf/classifier.py`.
  - WebSocket message format: modify in `src/backend/mlrf/server.py`.

- **Frontend:**
  - **WebSocket URL:** Change in the UI or code if your backend runs elsewhere.
  - **Styling:** Uses [Material-UI](https://mui.com/) for UI components and theming.
  - **Detection Types:** Extend to support more signal types or custom overlays.
  - **Spectrogram Parameters:** Adjust color maps, time window, or frequency range in the relevant components or utility files.

---

## Troubleshooting

- **WebSocket Connection Error:**  
  Make sure your backend server is running and accessible at `ws://localhost:3030` or your configured endpoint.
- **Port Conflicts:**  
  If port 3000 is in use, you can specify another port with `PORT=3001 pnpm start`.
- **Data Not Displaying:**  
  Ensure the backend is streaming data in the expected format:  
  `[psd_data..., detection, event_start, event_end]`
- **SDR Not Available / UHD Errors:**
    - Ensure UHD drivers are correctly installed and accessible.
    - Verify the USRP device is connected (`uhd_find_devices`).
    - If UHD is not installed or intended, ensure you are using `--source h5`.
- **Model Loading Errors:**
    - Verify the `--model` path points to a valid `.joblib` file.
    - Ensure the model was trained with compatible library versions.
- **Dependency Problems:**
    - Ensure `uv sync` and `pnpm install` completed successfully. Check the output for errors.
    - Always use `uv run` and `pnpm start` to execute scripts, ensuring the correct environment is used.

---

## Project Team

- **Riley Stuart**  - *V2X Vectrus* - Mentor
- **Chandler Barfield** - *V2X Vectrus* - Mentor
- **John Robie**  - *V2X Vectrus* - Mentor
- **Tamer Nadeem** - *Computer Science* - Faculty Advisor
- **Yanxiao Zhao** - *Electrical Engineering* - Faculty Advisor
- **Kush Patel** - *Computer Science* - Student Team Member
- **Shane Simes** - *Computer Science* - Student Team Member
- **Daniel Hartman** - *Electrical Engineering* - Student Team Member
- **Baaba Jeffrey** - *Computer Engineering* - Student Team Member

---

**Happy hacking!**

---

*This project is part of the Linux Foundation OMP Mentorship program and follows its best practices for open source project organization and documentation.*
