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
  const simulationIntervalRef = useRef(null);

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
    return colMap;
  }, []);

  // Function to downsample PSD data to match canvas width
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

  // Spectrogram drawing logic – draws a one-pixel-high line at the bottom
  const drawSpectrogram = useCallback((psd) => {
    try {
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

        try {
          const image = ctx.getImageData(0, 1, width, height - 1);
          ctx.putImageData(image, 0, 0);
        } catch (e) {
          console.error("Error shifting image data:", e);
        }
        ctx.putImageData(imgData, 0, height - 1);
        drawTimeout.current = null;
      });
    } catch (error) {
      console.error("Error in drawSpectrogram:", error);
      setError("Error drawing spectrogram.");
    }
  }, [colorMap]);

  // Simulation: continuously update the spectrogram background,
  // and with a small probability insert a simulated signal in the correct frequency band.
  useEffect(() => {
    if (!simulate) return;

    setConnectionStatus("Simulating");
    simulationIntervalRef.current = setInterval(() => {
      // Create a baseline PSD: all values set to 0.5
      const psd = Array(8192).fill(0.5);

      // With a 15% chance, simulate a signal
      if (Math.random() < 0.15) {
        // Randomly pick signal type
        const simulatedClassification = Math.random() < 0.5 ? "WiFi" : "Bluetooth";
        
        // Insert the signal bump in the appropriate band:
        if (simulatedClassification === "WiFi") {
          // For WiFi, use a fixed band (e.g., indices 3500–3700)
          const start = 3500;
          const widthBump = Math.floor(Math.random() * 100) + 100; // width between 100 and 200
          for (let i = start; i < start + widthBump && i < psd.length; i++) {
            psd[i] = 1.0;
          }
        } else {
          // For Bluetooth, use a different band (e.g., indices 1000–1100)
          const start = 1000;
          const widthBump = Math.floor(Math.random() * 50) + 50; // width between 50 and 100
          for (let i = start; i < start + widthBump && i < psd.length; i++) {
            psd[i] = 1.0;
          }
        }
        // Update classification and add a log entry
        setClassification(simulatedClassification);
        const timestamp = new Date().toLocaleTimeString();
        setLogs((prevLogs) => [
          { timestamp, classification: simulatedClassification },
          ...prevLogs,
        ]);
      } else {
        // No signal this tick; classification remains "Unknown"
        setClassification("Unknown");
      }

      // Draw the PSD line (with or without a signal bump)
      drawSpectrogram(psd);
    }, 200); // Update every 200ms

    return () => {
      clearInterval(simulationIntervalRef.current);
      setConnectionStatus("Idle");
    };
  }, [simulate, drawSpectrogram]);

  // Auto-scroll log to top when new entry is added
  useEffect(() => {
    if (logContainerRef.current) {
      logContainerRef.current.scrollTop = 0;
    }
  }, [logs]);

  const handleClear = () => {
    if (!canvasRef.current) return;
    const ctx = canvasRef.current.getContext("2d");
    ctx.fillStyle = "#000000";
    ctx.fillRect(0, 0, canvasRef.current.width, canvasRef.current.height);
  };

  const sendMessage = (message) => {
    if (socketRef.current && socketRef.current.readyState === WebSocket.OPEN) {
      socketRef.current.send(JSON.stringify(message));
    } else {
      setError("WebSocket is not open. Cannot send message.");
    }
  };

  const getBadgeColor = () => {
    switch (classification) {
      case "WiFi":
        return "#1E88E5";
      case "Bluetooth":
        return "#43A047";
      default:
        return "#757575";
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
            <canvas
              ref={canvasRef}
              width={800} // explicit drawing resolution
              height={400} // explicit drawing resolution
              style={{ width: "100%", height: "100%" }}
            />
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
