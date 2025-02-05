import React, { useEffect, useRef, useState, useCallback, useMemo } from "react";
import { Box, Button, Typography, CircularProgress, Alert, Divider } from "@mui/material";

const Spectrogram = () => {
  const canvasRef = useRef(null);
  const logContainerRef = useRef(null);
  const [classification, setClassification] = useState("Unknown");
  const [connectionStatus, setConnectionStatus] = useState("Idle");
  const [error, setError] = useState(null);
  const [logs, setLogs] = useState([]);
  const [simulate, setSimulate] = useState(false);
  const socketRef = useRef(null);
  const drawTimeout = useRef(null);

  // Precompute the color map only once
  const colorMap = useMemo(() => {
    const colMap = [];
    for (let i = 0; i < 256; i++) {
      let r, g, b;
      if (i < 64) {
        r = 0;
        g = i * 4;
        b = 255;
      } else if (i < 128) {
        r = 0;
        g = 255;
        b = 255 - (i - 64) * 4;
      } else if (i < 192) {
        r = (i - 128) * 4;
        g = 255;
        b = 0;
      } else {
        r = 255;
        g = 255 - (i - 192) * 4;
        b = 0;
      }
      colMap.push([r, g, b, 255]); // Ensure alpha is 255
    }
    // console.log("Color map created with", colMap.length, "entries.");
    return colMap;
  }, []);

  // Function to downsample PSD data
  const downsamplePsd = (psd, targetWidth) => {
    const step = Math.floor(psd.length / targetWidth);
    const downsampled = [];
    for (let i = 0; i < targetWidth; i++) {
      const segment = psd.slice(i * step, (i + 1) * step);
      const avg = segment.reduce((a, b) => a + b, 0) / step;
      downsampled.push(avg);
    }
    return downsampled;
  };

  // Spectrogram drawing logic
  const drawSpectrogram = useCallback((psd) => {
    try {
      // Removed console logs for speed.
      if (!canvasRef.current) return;

      if (
        !Array.isArray(psd) ||
        psd.length !== 8192 ||
        psd.some((val) => typeof val !== "number" || val < 0 || val > 1)
      ) {
        setError("Invalid PSD data format.");
        return;
      }

      const canvas = canvasRef.current;
      const ctx = canvas.getContext("2d");
      const width = canvas.width;
      const height = canvas.height;

      if (drawTimeout.current) return;

      drawTimeout.current = requestAnimationFrame(() => {
        const downsampled = downsamplePsd(psd, width);
        // Create ImageData for one line
        const imgData = ctx.createImageData(width, 1);

        for (let i = 0; i < width; i++) {
          const intensity = Math.min(255, Math.floor(downsampled[i] * 255));
          const [r, g, b, a] = colorMap[intensity];
          const index = i * 4;
          imgData.data[index] = r;
          imgData.data[index + 1] = g;
          imgData.data[index + 2] = b;
          imgData.data[index + 3] = a;
        }

        // Shift existing image up by 1 pixel
        try {
          const image = ctx.getImageData(0, 1, width, height - 1);
          ctx.putImageData(image, 0, 0);
        } catch (e) {
          console.error("Error shifting image data:", e);
        }

        // Draw the new line at the bottom
        ctx.putImageData(imgData, 0, height - 1);
        drawTimeout.current = null;
      });
    } catch (error) {
      console.error("Error in drawSpectrogram:", error);
      setError("Error drawing spectrogram.");
    }
  }, [colorMap]);

  // Simulation: simulate incoming signal messages every 1.5 seconds when toggled on
  useEffect(() => {
    if (!simulate) return;

    setConnectionStatus("Simulating");
    const interval = setInterval(() => {
      // Create a simulated PSD array (all values set to 0.5)
      const simulatedPSD = Array(8192).fill(0.5);
      // Randomly choose a classification
      const simulatedClassification = Math.random() < 0.5 ? "WiFi" : "Bluetooth";

      drawSpectrogram(simulatedPSD);
      setClassification(simulatedClassification);
      const timestamp = new Date().toLocaleTimeString();
      setLogs((prevLogs) => [
        { timestamp, classification: simulatedClassification },
        ...prevLogs,
      ]);
    }, 1500); // Faster interval (1.5 seconds)

    return () => {
      clearInterval(interval);
      setConnectionStatus("Idle");
    };
  }, [simulate, drawSpectrogram]);

  // Scroll to top of logs when a new entry is added
  useEffect(() => {
    if (logContainerRef.current) {
      logContainerRef.current.scrollTop = 0;
    }
  }, [logs]);

  // Function to handle clearing the spectrogram
  const handleClear = () => {
    if (!canvasRef.current) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    ctx.fillStyle = "#000000";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
  };

  // Function to send messages via WebSocket (kept for production use)
  const sendMessage = (message) => {
    if (socketRef.current && socketRef.current.readyState === WebSocket.OPEN) {
      socketRef.current.send(JSON.stringify(message));
    } else {
      setError("WebSocket is not open. Cannot send message.");
    }
  };

  // Determine classification badge color based on classification
  const getBadgeColor = () => {
    switch (classification) {
      case "WiFi":
        return "#1E88E5"; // Blue
      case "Bluetooth":
        return "#43A047"; // Green
      default:
        return "#757575"; // Gray
    }
  };

  return (
    <Box
      sx={{
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        padding: "20px",
        backgroundColor: "#121212",
        minHeight: "100vh",
        color: "#ffffff",
      }}
    >
      {/* Main container with spectrogram and logs side by side */}
      <Box
        sx={{
          display: "flex",
          flexDirection: { xs: "column", md: "row" },
          gap: "20px",
          width: "90%",
          maxWidth: "1200px",
          marginBottom: "20px",
        }}
      >
        {/* Spectrogram and controls container */}
        <Box
          sx={{
            flex: 2,
            display: "flex",
            flexDirection: "column",
          }}
        >
          <Box
            sx={{
              position: "relative",
              width: "100%",
              height: "400px",
              border: "2px solid #ffffff",
              borderRadius: "8px",
              overflow: "hidden",
              backgroundColor: "#000000",
              boxShadow: "0 4px 8px rgba(0, 0, 0, 0.3)",
              marginBottom: "20px",
            }}
          >
            <canvas ref={canvasRef} style={{ width: "100%", height: "100%" }} />
            {connectionStatus === "Idle" && (
              <Box
                sx={{
                  position: "absolute",
                  top: "50%",
                  left: "50%",
                  transform: "translate(-50%, -50%)",
                  display: "flex",
                  flexDirection: "column",
                  alignItems: "center",
                  backgroundColor: "rgba(0,0,0,0.5)",
                  padding: "10px",
                  borderRadius: "8px",
                }}
              >
                <CircularProgress color="inherit" />
                <Typography variant="body1" sx={{ mt: 2 }}>
                  Idle
                </Typography>
              </Box>
            )}
            {error && (
              <Alert
                severity="error"
                sx={{
                  position: "absolute",
                  top: 0,
                  left: 0,
                  width: "100%",
                  zIndex: 1,
                }}
              >
                {error}
              </Alert>
            )}
          </Box>

          <Box
            sx={{
              display: "flex",
              gap: "10px",
              marginBottom: "20px",
            }}
          >
            <Button variant="contained" color="secondary" onClick={handleClear}>
              Clear Spectrogram
            </Button>
            <Button
              variant="outlined"
              color="inherit"
              onClick={() => sendMessage({ type: "request_update" })}
            >
              Request Update
            </Button>
            <Button
              variant="outlined"
              color="primary"
              onClick={() => setSimulate((prev) => !prev)}
            >
              {simulate ? "Stop Simulation" : "Toggle Simulation"}
            </Button>
          </Box>

          <Box>
            <Typography variant="h6">
              Current Classification:{" "}
              <Box
                component="span"
                sx={{
                  padding: "8px 16px",
                  borderRadius: "16px",
                  backgroundColor: getBadgeColor(),
                  color: "#ffffff",
                  fontWeight: "bold",
                }}
              >
                {classification}
              </Box>
            </Typography>
          </Box>
        </Box>

        {/* Log panel container */}
        <Box
          sx={{
            flex: 1,
            display: "flex",
            flexDirection: "column",
            border: "2px solid #ffffff",
            borderRadius: "8px",
            height: "400px",
            overflowY: "auto",
            backgroundColor: "#1e1e1e",
            padding: "10px",
          }}
          ref={logContainerRef}
        >
          <Typography variant="h6" sx={{ textAlign: "center", mb: 1 }}>
            Signal Log
          </Typography>
          <Divider sx={{ backgroundColor: "#757575", mb: 1 }} />
          {logs.length === 0 ? (
            <Typography variant="body2" sx={{ textAlign: "center", color: "#757575" }}>
              No signals detected yet.
            </Typography>
          ) : (
            logs.map((log, index) => (
              <Box key={index} sx={{ mb: 1, padding: "4px", borderBottom: "1px solid #333" }}>
                <Typography variant="caption" color="secondary">
                  {log.timestamp}
                </Typography>
                <Typography variant="body2" sx={{ ml: 1, display: "inline" }}>
                  {log.classification}
                </Typography>
              </Box>
            ))
          )}
        </Box>
      </Box>
    </Box>
  );
};

export default Spectrogram;
