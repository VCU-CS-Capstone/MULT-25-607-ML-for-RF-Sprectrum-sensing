import React, { useEffect, useRef, useState, useCallback, useMemo } from "react";
import { Box, Button, Typography, CircularProgress, Alert, Divider } from "@mui/material";

const Spectrogram = () => {
  // Refs for canvases and logs.
  const canvasRef = useRef(null);
  const overlayCanvasRef = useRef(null);
  const logContainerRef = useRef(null);
  const tickCountRef = useRef(0); // counts how many rows have been drawn

  // Arrays for storing detection events and boxes.
  const detectionEventsRef = useRef([]);
  const detectionBoxesRef = useRef([]);

  const [classification, setClassification] = useState("Unknown");
  const [connectionStatus, setConnectionStatus] = useState("Idle");
  const [error, setError] = useState(null);
  const [logs, setLogs] = useState([]);
  const [simulate, setSimulate] = useState(false);

  const drawTimeout = useRef(null);
  const simulationIntervalRef = useRef(null);

  // Precompute a color map for 256 intensity levels (blue -> cyan -> green -> yellow -> red).
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

  // Downsample the PSD to match the canvas width.
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
   * Generate a PSD row with:
   * 1) Baseline noise
   * 2) Stable horizontal lines at specific frequencies
   * 3) A vertical band in the center (around freq=1024) that’s brighter
   * 4) Mild random flicker
   */
  const generateStructuredPsd = (timeIndex) => {
    const psd = new Array(2048).fill(0);

    // 1) Baseline noise (~0.2 + random)
    for (let i = 0; i < 2048; i++) {
      psd[i] = 0.2 + Math.random() * 0.1;
    }

    // 2) Stable horizontal lines (boosted amplitude)
    //    e.g., lines at [100, 200, 400, 800, 1200, 1600]
    const stableLines = [100, 200, 400, 800, 1200, 1600];
    stableLines.forEach((freqIndex) => {
      // Stronger boost so lines are easier to see
      psd[freqIndex] += 0.6 + Math.random() * 0.2; 
    });

    // 3) Vertical band in the center ([900..1140])
    for (let i = 900; i <= 1140; i++) {
      psd[i] += 0.3;
    }

    // 4) Mild random flicker/burst every ~15 ticks
    if (timeIndex % 15 === 0) {
      for (let i = 600; i <= 650; i++) {
        psd[i] += Math.random() * 0.3;
      }
    }

    // Clamp to [0,1]
    for (let i = 0; i < 2048; i++) {
      if (psd[i] > 1) psd[i] = 1;
    }
    return psd;
  };

  /**
   * Draws one new row on the spectrogram.
   */
  const drawSpectrogram = useCallback(
    (psd, globalOffset = 0, tailInfo = null) => {
      try {
        if (!canvasRef.current) return;
        if (
          !Array.isArray(psd) ||
          psd.length !== 2048 ||
          psd.some((val) => typeof val !== "number" || val < 0 || val > 1)
        ) {
          setError("Invalid PSD data format.");
          return;
        }
        const canvas = canvasRef.current;
        const ctx = canvas.getContext("2d", { willReadFrequently: true });
        const width = canvas.width;
        const height = canvas.height;
        const psdToCanvasFactor = psd.length / width;

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
            const clampedIntensity = Math.max(0, Math.min(255, baseIntensity + offset));
            const colorEntry = colorMap[clampedIntensity] || [0, 0, 0, 255];
            const [r, g, b, a] = colorEntry;
            const idx = i * 4;
            imgData.data[idx] = r;
            imgData.data[idx + 1] = g;
            imgData.data[idx + 2] = b;
            imgData.data[idx + 3] = a;
          }

          // Shift the image up one row
          try {
            const image = ctx.getImageData(0, 1, width, height - 1);
            ctx.putImageData(image, 0, 0);
          } catch (e) {
            console.error("Error shifting image data:", e);
          }

          // Draw the new row at the bottom
          ctx.putImageData(imgData, 0, height - 1);
          drawTimeout.current = null;
        });
      } catch (err) {
        console.error("Error in drawSpectrogram:", err);
        setError("Error drawing spectrogram.");
      }
    },
    [colorMap]
  );

  /**
   * Draws all detection boxes on the overlay canvas.
   * We compute an effective frequency-to-pixel scaling:
   *   step = floor(2048 / width)
   *   pixelX = freqIndex / step
   */
  const drawDetectionBoxes = useCallback(() => {
    if (!canvasRef.current || !overlayCanvasRef.current) return;
    const canvas = canvasRef.current;
    const overlayCanvas = overlayCanvasRef.current;
    const ctx = overlayCanvas.getContext("2d");
    const width = canvas.width;
    const height = canvas.height;
    ctx.clearRect(0, 0, width, height);

    const effectiveStep = Math.floor(2048 / width);
    const scaleX = 1 / effectiveStep;

    const newBoxes = [];
    detectionBoxesRef.current.forEach((box) => {
      const currentTick = tickCountRef.current;
      const y_top = (height - 1) - (currentTick - box.startTick);
      let y_bottom = y_top + box.totalTicks - 1;
      if (y_bottom < 0) return;

      const visibleTop = Math.max(0, y_top);
      const visibleBottom = Math.min(height - 1, y_bottom);
      const rectHeight = visibleBottom - visibleTop + 1;

      let x = (box.center - 3 * box.sigma) * scaleX;
      let rectWidth = (6 * box.sigma) * scaleX;
      if (x < 0) {
        rectWidth += x;
        x = 0;
      }
      if (x + rectWidth > width) {
        rectWidth = width - x;
      }

      let strokeColor;
      if (box.type === "Bluetooth") {
        strokeColor = "#2196F3"; // blue
      } else if (box.type === "WiFi") {
        strokeColor = "#F44336"; // red
      } else {
        return;
      }
      ctx.strokeStyle = strokeColor;
      ctx.lineWidth = 2;
      ctx.strokeRect(x, visibleTop, rectWidth, rectHeight);
      newBoxes.push(box);
    });
    detectionBoxesRef.current = newBoxes;
  }, []);

  /**
   * Simulation effect.
   * Every 50ms, we generate a structured PSD row and optionally add
   * a detection event (Gaussian bump + bounding box).
   */
  useEffect(() => {
    if (!simulate) return;
    setConnectionStatus("Simulating");

    simulationIntervalRef.current = setInterval(() => {
      // 1) Generate a structured baseline row
      const psd = generateStructuredPsd(tickCountRef.current);

      let globalOffset = 0;
      let tailInfo = null;

      // 2) Possibly add a new detection event (5% chance)
      if (Math.random() < 0.05) {
        const simulatedClassification = Math.random() < 0.5 ? "WiFi" : "Bluetooth";
        let eventCenter, eventSigma, eventDuration;

        // Increased sigma for bigger signals
        // WiFi: was ~6, now ~12
        // Bluetooth: was ~3, now ~8
        if (simulatedClassification === "WiFi") {
          eventCenter = 360 + (Math.random() - 0.5) * 10;
          eventSigma = 12; 
        } else {
          eventCenter = 105 + (Math.random() - 0.5) * 4;
          eventSigma = 8; 
        }

        eventDuration = Math.floor(Math.random() * 10) + 10; // in ticks

        // Create detection event
        detectionEventsRef.current.push({
          type: simulatedClassification,
          center: eventCenter,
          sigma: eventSigma,
          duration: eventDuration,
        });
        // Create detection box
        detectionBoxesRef.current.push({
          type: simulatedClassification,
          center: eventCenter,
          sigma: eventSigma,
          startTick: tickCountRef.current,
          totalTicks: eventDuration + 5,
        });

        setClassification(simulatedClassification);
        const timestamp = new Date().toLocaleTimeString();
        setLogs((prevLogs) => [{ timestamp, classification: simulatedClassification }, ...prevLogs]);
      } else {
        setClassification("Unknown");
      }

      // 3) Add Gaussian bumps for each active detection event
      // Increased amplitude from 0.7 -> 1.0 for stronger signals
      detectionEventsRef.current.forEach((event) => {
        const amplitude = 1.0;
        for (let i = 0; i < 2048; i++) {
          const bump = amplitude * Math.exp(-Math.pow(i - event.center, 2) / (2 * Math.pow(event.sigma, 2)));
          psd[i] = Math.min(1, psd[i] + bump);
        }
        // Decrement event duration
        event.duration--;
      });
      // Remove finished events
      detectionEventsRef.current = detectionEventsRef.current.filter((ev) => ev.duration > 0);

      // 4) Draw the spectrogram row
      drawSpectrogram(psd, globalOffset, tailInfo);
      tickCountRef.current++;
      drawDetectionBoxes();
    }, 50);

    return () => {
      clearInterval(simulationIntervalRef.current);
      setConnectionStatus("Idle");
      if (overlayCanvasRef.current) {
        const ctx = overlayCanvasRef.current.getContext("2d");
        ctx.clearRect(0, 0, overlayCanvasRef.current.width, overlayCanvasRef.current.height);
      }
    };
  }, [simulate, drawSpectrogram, drawDetectionBoxes]);

  // WebSocket effect (only if simulation is off).
  useEffect(() => {
    if (simulate) return;

    // Replace with your actual WebSocket server URL.
    const ws = new WebSocket("ws://localhost:8080");

    ws.onopen = () => {
      setConnectionStatus("Connected");
      console.log("WebSocket connected");
    };

    ws.onmessage = (event) => {
      let data;
      try {
        data = JSON.parse(event.data);
      } catch (e) {
        console.error("Error parsing WebSocket message:", e);
        return;
      }
      if (!Array.isArray(data) || data.length !== 2051) {
        setError("Invalid data received from WebSocket.");
        return;
      }
      // Extract PSD (first 2048) and detection info (last three)
      const psdData = data.slice(0, 2048);
      const detectionType = data[2048];
      const startFreq = data[2049];
      const endFreq = data[2050];

      let classificationStr = "Unknown";
      let tailInfo = null;
      let globalOffset = 0;
      if (detectionType === 1) {
        classificationStr = "WiFi";
        globalOffset = 30;
        tailInfo = { start: startFreq, end: endFreq, offset: 30 };
      } else if (detectionType === 2) {
        classificationStr = "Bluetooth";
        globalOffset = 30;
        tailInfo = { start: startFreq, end: endFreq, offset: 30 };
      }
      setClassification(classificationStr);

      if (detectionType !== 0) {
        const timestamp = new Date().toLocaleTimeString();
        setLogs((prevLogs) => [{ timestamp, classification: classificationStr }, ...prevLogs]);
      }

      drawSpectrogram(psdData, globalOffset, tailInfo);
      tickCountRef.current++;
      drawDetectionBoxes();
    };

    ws.onerror = (err) => {
      console.error("WebSocket error:", err);
      setError("WebSocket encountered an error.");
    };

    ws.onclose = () => {
      console.log("WebSocket closed");
      setConnectionStatus("Disconnected");
    };

    return () => ws.close();
  }, [simulate, drawSpectrogram, drawDetectionBoxes]);

  // Auto-scroll the log panel when new entries are added.
  useEffect(() => {
    if (logContainerRef.current) {
      logContainerRef.current.scrollTop = 0;
    }
  }, [logs]);

  // Clear the spectrogram + overlay
  const handleClear = () => {
    if (!canvasRef.current) return;
    const ctx = canvasRef.current.getContext("2d");
    ctx.fillStyle = "#000000";
    ctx.fillRect(0, 0, canvasRef.current.width, canvasRef.current.height);
  
    const overlayCanvas = overlayCanvasRef.current;
    if (overlayCanvas) {
      const overlayCtx = overlayCanvas.getContext("2d");
      overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
    }
  };  

  const sendMessage = (message) => {
    console.log("Message sent:", message);
    // Implement actual WebSocket send logic if needed.
  };

  // Returns a badge color based on the detection type.
  const getBadgeColor = () => {
    switch (classification) {
      case "Bluetooth":
        return "#2196F3";
      case "WiFi":
        return "#F44336";
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
      {/* Main container with spectrogram and log panel */}
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
        {/* Spectrogram and controls */}
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
            <canvas ref={canvasRef} width={800} height={400} style={{ width: "100%", height: "100%" }} />
            <canvas
              ref={overlayCanvasRef}
              width={800}
              height={400}
              style={{
                position: "absolute",
                top: 0,
                left: 0,
                width: "100%",
                height: "100%",
                pointerEvents: "none",
              }}
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
          <Box sx={{ display: "flex", gap: "10px", marginBottom: "20px" }}>
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
        {/* Log panel */}
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
