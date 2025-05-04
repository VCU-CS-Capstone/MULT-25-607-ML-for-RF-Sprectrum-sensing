# MLRF: Machine Learning for Radio Frequency Signal Classification

The **MLRF Frontend** is a React-based web application for real-time visualization and control of RF signal classification, as part of the [MLRF](../README.md) (Machine Learning Radio Frequency) project.  
It provides a responsive, real-time spectrogram and detection overlay for WiFi and Bluetooth signals, connecting to the MLRF backend via WebSocket.

---

## Features

- **Real-time Spectrogram:** Visualizes incoming PSD (Power Spectral Density) data as a color spectrogram.
- **Detection Overlay:** Highlights detected WiFi (green) and Bluetooth (red) regions.
- **Interactive Controls:** Adjust signal range, pixel size, and connection state.
- **WebSocket Integration:** Connects to a backend server for live data streaming.
- **Error Handling:** User-friendly error messages and connection status indicators.
- **Fullscreen Mode:** Immersive analysis with adaptive controls.

---

## Getting Started

### Prerequisites

- [Node.js](https://nodejs.org/) (v16 or newer recommended)
- [pnpm](https://pnpm.io/) (install with `npm install -g pnpm`)

### Installation

1. **Install dependencies:**
   ```bash
   pnpm install
   ```

2. **Start the development server:**
   ```bash
   pnpm start
   ```
   The app will open at [http://localhost:3000](http://localhost:3000).

---

## Usage

- The frontend expects a WebSocket backend running at `ws://localhost:3030`.
- Use the **Connect** button to start receiving data.
- Adjust the **Signal Range** and **Pixel Size** sliders to tune the visualization.
- Use the **Clear** button to reset the spectrogram.
- Click the **Fullscreen** button (top right of the spectrogram) for an immersive view; controls will appear as an overlay in fullscreen.

---

## Project Structure

```
src/
 |
 |-- components/
 |   |-- Spectrogram.js         # Main spectrogram display component
 |   |-- SpectrogramCanvas.js   # Canvas rendering logic
 |   |-- ControlButtons.js      # Play/Pause/etc. buttons
 |   |-- ConnectionStatus.js    # WebSocket connection indicator
 |   |-- DetectionInfo.js       # Displays classification results
 |   |-- PixelSizeSlider.js     # Slider for pixel size control
 |   |-- SignalRangeSlider.js   # Slider for signal range control
 |   |-- ErrorModal.js          # Modal for displaying errors
 |   |-- ServerAddressModal.js  # Modal for entering server address
 |
 |-- utils/
 |   |-- classify.js            # Frontend classification enum
 |   |-- colorMap.js            # Spectrogram color mapping
 |   |-- drawingUtils.js        # Canvas drawing helpers
 |   |-- psdUtils.js            # Power Spectral Density calculations/helpers
 |   |-- useWindowSize.js       # Hook for tracking window size
 |   |-- websocketHandler.js    # WebSocket connection management
 |
 |-- App.js                     # Main application component
 |-- index.js                   # Application entry point
 |-- index.css                  # Global styles (Empty)
```

- **components/**: UI and visualization components.
- **utils/**: Utility functions for signal processing, drawing, and WebSocket handling.
- **App.js**: Main application logic and state management.

---

## Customization

- **WebSocket URL:**  
  Change the URL in `Spectrogram.js` or make it configurable if your backend runs elsewhere.
- **Styling:**  
  Uses [Material-UI](https://mui.com/) for UI components and theming. You can customize the theme or component styles as needed.
- **Detection Types:**  
  Extend `classify.js` and `drawingUtils.js` to support more signal types or custom overlays.
- **Spectrogram Parameters:**  
  Adjust color maps, time window, or frequency range in the relevant components or utility files.

---

## Troubleshooting

- **WebSocket Connection Error:**  
  Make sure your backend server is running and accessible at `ws://localhost:3030` or your configured endpoint.
- **Port Conflicts:**  
  If port 3000 is in use, you can specify another port with `PORT=3001 pnpm start`.
- **Data Not Displaying:**  
  Ensure the backend is streaming data in the expected format:  
  `[psd_data..., detection, event_start, event_end]`

---

## Requirements

- Node.js 16+
- pnpm
- React
- Material-UI

---

## Contact

For questions, email [kushpatel169@gmail.com](mailto:kushpatel169@gmail.com). :D

---

**Happy hacking!**
