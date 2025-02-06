import React, { useEffect, useRef, useState, useCallback, useMemo } from "react";
import { Box, Button, Typography, CircularProgress, Alert, Divider } from "@mui/material";

const Spectrogram = () => {
  const canvasRef = useRef(null);
  const logContainerRef = useRef(null);
  const tailRef = useRef(null); // holds tail info for fading the detection
  const detectionEventRef = useRef(null); // holds a current detection event, if any
  const [classification, setClassification] = useState("Unknown");
  const [connectionStatus, setConnectionStatus] = useState("Idle");
  const [error, setError] = useState(null);
  const [logs, setLogs] = useState([]);
  const [simulate, setSimulate] = useState(false);
  const drawTimeout = useRef(null);
  const simulationIntervalRef = useRef(null);

  // Precompute the color map for 256 intensity levels.
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
      colMap.push([r, g, b, 255]);
    }
    return colMap;
  }, []);

  // Downsample the PSD (8192 points) to match canvas width.
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

  /**
   * Draws a one-pixel-high row on the spectrogram.
   *
   * @param {number[]} psd - Array of 8192 numbers (each in [0,1]).
   * @param {number} globalOffset - A uniform brightness offset added to every pixel.
   * @param {object|null} tailInfo - If provided, an object with { start, end, offset } that applies
   *        an extra offset only for canvas pixels whose corresponding PSD index falls in that region.
   */
  const drawSpectrogram = useCallback((psd, globalOffset = 0, tailInfo = null) => {
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
      const psdToCanvasFactor = 8192 / width;
      if (drawTimeout.current) return;
      drawTimeout.current = requestAnimationFrame(() => {
        const downsampled = downsamplePsd(psd, width);
        const imgData = ctx.createImageData(width, 1);
        for (let i = 0; i < width; i++) {
          const baseIntensity = Math.min(255, Math.floor(downsampled[i] * 255));
          let offset = globalOffset;
          if (tailInfo) {
            const psdIndex = i * psdToCanvasFactor;
            if (psdIndex >= tailInfo.start && psdIndex < tailInfo.end) {
              offset = Math.max(offset, tailInfo.offset);
            }
          }
          const clampedIntensity = Math.max(0, Math.min(255, Math.floor(baseIntensity + offset)));
          const colorEntry = colorMap[clampedIntensity] || [0, 0, 0, 255];
          const [r, g, b, a] = colorEntry;
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

  // Simulation effect for continuous spectrogram.
  // Update interval is 50ms per row.
  // Generates a noisy baseline, boosts rows when noise > threshold,
  // and (with 15% chance or if an event is ongoing) simulates a long detection event.
  useEffect(() => {
    if (!simulate) return;
    setConnectionStatus("Simulating");
    simulationIntervalRef.current = setInterval(() => {
      // Generate a noisy baseline (values around 0.5 ±0.4).
      const psd = Array.from({ length: 8192 }, () => 0.5 + (Math.random() - 0.5) * 0.4);
      let globalOffset = 0;
      let tailInfo = null;

      // Check if the maximum noise exceeds a threshold.
      const noiseThreshold = 0.65;
      const maxVal = Math.max(...psd);
      if (maxVal > noiseThreshold) {
        globalOffset = 30;
      }

      // If a detection event is already active, use it.
      if (detectionEventRef.current) {
        // Continue the ongoing event.
        const event = detectionEventRef.current;
        globalOffset = 0; // Do not boost the current row further.
        // Use the event's center and sigma.
        var simulatedClassification = event.type;
        var center = event.center;
        var sigma = event.sigma;
        event.duration--; // decrement duration
        if (event.duration <= 0) {
          detectionEventRef.current = null;
        }
      } else if (Math.random() < 0.15) {
        // Start a new detection event.
        const simulatedClassification = Math.random() < 0.5 ? "WiFi" : "Bluetooth";
        let center, sigma, duration;
        if (simulatedClassification === "WiFi") {
          center = 3600 + (Math.random() - 0.5) * 100; // around 3600 ±50
          sigma = 60; // wider bump
        } else {
          center = 1050 + (Math.random() - 0.5) * 40; // around 1050 ±20
          sigma = 30;
        }
        // Set duration between 10 and 20 ticks.
        duration = Math.floor(Math.random() * 10) + 10;
        detectionEventRef.current = { type: simulatedClassification, center, sigma, duration };
        // Log the event once when it starts.
        setClassification(simulatedClassification);
        const timestamp = new Date().toLocaleTimeString();
        setLogs((prevLogs) => [
          { timestamp, classification: simulatedClassification },
          ...prevLogs,
        ]);
      } else {
        setClassification("Unknown");
      }

      // If there is an active detection event, add its Gaussian bump to the PSD.
      if (detectionEventRef.current) {
        const event = detectionEventRef.current;
        const amplitude = 0.7; // bump amplitude
        for (let i = 0; i < 8192; i++) {
          const bump = amplitude * Math.exp(-Math.pow(i - event.center, 2) / (2 * Math.pow(event.sigma, 2)));
          // Add the bump to the baseline, capping at 1.
          psd[i] = Math.min(1, psd[i] + bump);
        }
        // Also, create a tail for subsequent rows.
        tailRef.current = { center: event.center, sigma: event.sigma, steps: 10, offset: 120 };
      }

      // If a tail exists, prepare tailInfo.
      if (tailRef.current) {
        tailInfo = { center: tailRef.current.center, sigma: tailRef.current.sigma, offset: tailRef.current.steps * 20 };
        tailRef.current.steps -= 1;
        if (tailRef.current.steps <= 0) {
          tailRef.current = null;
        }
      }

      // Draw the current row.
      drawSpectrogram(psd, globalOffset, tailInfo);
    }, 50); // update every 50ms for a continuous effect

    return () => {
      clearInterval(simulationIntervalRef.current);
      setConnectionStatus("Idle");
    };
  }, [simulate, drawSpectrogram]);

  // Auto-scroll the log panel when new entries are added.
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
    console.log("Message sent:", message);
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
            <Button variant="outlined" color="inherit" onClick={() => sendMessage({ type: "request_update" })}>
              Request Update
            </Button>
            <Button variant="outlined" color="primary" onClick={() => setSimulate((prev) => !prev)}>
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
