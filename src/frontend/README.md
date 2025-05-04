# MLRF Frontend

This is the **frontend** for the MLRF (Machine Learning Radio Frequency) project, built with [Create React App](https://create-react-app.dev/).  
It provides a responsive, real-time spectrogram visualization and control interface for RF signal detection, including WiFi and Bluetooth, via a WebSocket backend.

---

## Features

- **Real-time Spectrogram**: Visualizes incoming PSD (Power Spectral Density) data as a color spectrogram.
- **Detection Overlay**: Highlights detected WiFi (GREEN) and Bluetooth regions (RED).
- **Controls**: Adjust signal range, pixel size, and connection state.
- **WebSocket Integration**: Connects to a backend server for live data.
- **Error Handling**: User-friendly error messages and connection status.
- **Fullscreen Mode**: Canvas and controls adapt for immersive analysis.

---

## Getting Started

### Prerequisites

- [Node.js](https://nodejs.org/) (v16 or newer recommended)
- [npm](https://www.npmjs.com/) (comes with Node.js)

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
  components/
    Spectrogram.js
    SpectrogramCanvas.js
    ControlButtons.js
    ConnectionStatus.js
    DetectionInfo.js
    PixelSizeSlider.js
    SignalRangeSlider.js
    ErrorModal.js
  utils/
    classify.js
    colorMap.js
    drawingUtils.js
    psdUtils.js
    useWindowSize.js
    websocketHandler.js
  App.js
  index.js
  index.css
```

---

## Customization

- **WebSocket URL**: Change the URL in `Spectrogram.js` or make it configurable if your backend runs elsewhere.
- **Styling**: Uses [Material-UI](https://mui.com/) for UI components and theming.
- **Detection Types**: Extend `classify.js` and `drawingUtils.js` to support more signal types if needed.

---

## Troubleshooting

- **WebSocket Connection Error**:  
  Make sure your backend server is running and accessible at `ws://localhost:3030`.
- **Port Conflicts**:  
  If port 3000 is in use, you can specify another port with `PORT=3001 npm start`.

---

## License

This project is licensed under the MIT License.

---

## Acknowledgments

- [Create React App](https://create-react-app.dev/)
- [Material-UI](https://mui.com/)
- [WebSocket API](https://developer.mozilla.org/en-US/docs/Web/API/WebSockets_API)

---

**MLRF Frontend**  
_Real-time RF signal visualization and control._
