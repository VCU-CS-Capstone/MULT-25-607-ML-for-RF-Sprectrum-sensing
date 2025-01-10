import { useEffect, useRef, useState } from "react";
import { Button, Card, CardBody, CardFooter} from "@material-tailwind/react";

function Spectrogram({ width, height }) {
  // =============== State Management ===============
  const containerRef = useRef(null);
  const waterfallRef = useRef(null);
  const [bluetoothActive, setBluetoothActive] = useState(false);
  const [wifiActive, setWifiActive] = useState(false);
  const [liveData, setLiveData] = useState(false);
  const [wifiBand, setWifiBand] = useState(null);

  // =============== Color Mapping Functions ===============
  const createColorMap = () => {
    const colMap = [];
    for (let i = 0; i < 256; i++) {
      const r = Math.min(255, Math.max(0, i > 128 ? (i - 128) * 2 : 0));
      const g = Math.min(255, Math.max(0, i > 64 ? i * 2 : 0));
      const b = Math.min(255, Math.max(0, 255 - i * 2));
      colMap.push([r, g, b, 255]);
    }
    return colMap;
  };

  // =============== Canvas Drawing Functions ===============
  const initializeWaterfall = () => {
    const colMap = createColorMap();
    const canvas = document.createElement("canvas");
    const ctx = canvas.getContext("2d");
    canvas.width = width;
    canvas.height = height;

    if (containerRef.current) {
      containerRef.current.appendChild(canvas);
    }

    const drawLine = (buffer) => {
      if (!buffer || buffer.length === 0) return;
      const imgData = ctx.createImageData(width, 1);
      for (let i = 0; i < width; i++) {
        const intensity = buffer[i] || 0;
        const [r, g, b, a] = colMap[intensity];
        const index = i * 4;
        imgData.data[index] = r;
        imgData.data[index + 1] = g;
        imgData.data[index + 2] = b;
        imgData.data[index + 3] = a;
      }
      ctx.putImageData(imgData, 0, height-1);
      ctx.drawImage(canvas, 0, 1, width, height - 1, 0, 0, width, height - 1);
    };

    waterfallRef.current = {
      canvas,
      ctx,
      drawLine,
      clear: () => ctx.clearRect(0, 0, width, height),
    };

    return () => {
      if (containerRef.current && waterfallRef.current?.canvas) {
        containerRef.current.removeChild(waterfallRef.current.canvas);
      }
    };
  };

  // =============== Signal Simulation Functions ===============
  const simulateFFT = () => {
    const buffer = new Uint8Array(width).fill(0);
    const frequencies = {
      bluetooth: [2400, 2483],
      wifi: [[2400, 2500], [3600, 3700], [5000, 5800]],
    };

    const addSignal = (range, buffer) => {
      const randomFreq = (range[1] - range[0]) + range[0];
      const index = Math.floor(((randomFreq - 2400) / 5000) * buffer.length);
      
      // Width of the signal (increased for better fade effect)
      const width = 100;
      
      for (let i = -width; i <= width; i++) {
      const targetIndex = index + i;
      if (targetIndex >= 0 && targetIndex < buffer.length) {
        // Calculate fade factor based on distance from center
        const distance = Math.abs(i);
        const fadeFactor = 1 - (distance / width);
        const intensity = 255 * fadeFactor;
        buffer[targetIndex] = Math.max(buffer[targetIndex], intensity);
      }
      }
    };

    if (!liveData) {
      if (bluetoothActive) addSignal(frequencies.bluetooth, buffer);
      if (wifiActive && wifiBand) addSignal(wifiBand, buffer);

      // Add background noise
      for (let i = 0; i < buffer.length; i++) {
        // Box-Muller transform to generate Gaussian noise
        const u1 = Math.random();
        const u2 = Math.random();
        const gaussianNoise = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
        // Scale the noise and ensure it stays within 0-255 range
        buffer[i] = Math.max(buffer[i], Math.min(128, Math.max(0, 70 + gaussianNoise * 20)));
      }

      if (waterfallRef.current) {
        waterfallRef.current.drawLine(buffer);
      }
    }
  };

  // =============== Event Handlers ===============
  const handleClear = () => {
    waterfallRef.current?.clear();
  };

  const handleWifiToggle = () => {
    setWifiActive(!wifiActive);
    if (!wifiActive) {
      const wifiBands = [[2400, 2500], [3600, 3700], [5000, 5800]];
      setWifiBand(wifiBands[Math.floor(Math.random() * wifiBands.length)]);
    } else {
      setWifiBand(null);
    }
  };

  // =============== Effects ===============
  // Initialize waterfall display
  useEffect(() => {
    const cleanup = initializeWaterfall();
    return () => cleanup();
  }, [width, height]);

  // Handle signal simulation
  useEffect(() => {
    const interval = setInterval(simulateFFT, 1000 / 30);
    return () => clearInterval(interval);
  }, [bluetoothActive, wifiActive, wifiBand, liveData, width]);

  // =============== Render ===============
  return (
    <Card className="flex items-center w-fit">
      <div
        ref={containerRef}
        className="border border-gray-300 rounded"
        style={{ width: `${width}px`, height: `${height}px` }}
      />
      <CardFooter>
        <Button
          variant={bluetoothActive ? "filled" : "outlined"}
          color={bluetoothActive ? "red" : "blue"}
          onClick={() => setBluetoothActive(!bluetoothActive)}
        >
          {bluetoothActive ? "Disable Bluetooth" : "Enable Bluetooth"}
        </Button>
        <Button
          variant={wifiActive ? "filled" : "outlined"}
          color={wifiActive ? "red" : "green"}
          onClick={handleWifiToggle}
        >
          {wifiActive ? "Disable Wi-Fi" : "Enable Wi-Fi"}
        </Button>
        <Button
          variant={liveData ? "filled" : "outlined"}
          color={liveData ? "purple" : "deep-orange"}
          onClick={() => setLiveData(!liveData)}
        >
          {liveData ? "Switch to Demo Data" : "Switch to Live Data"}
        </Button>
        <Button variant="outlined" color="gray" onClick={handleClear}>
          Clear
        </Button>
      </CardFooter>
    </Card>
  );
}

export default Spectrogram;
