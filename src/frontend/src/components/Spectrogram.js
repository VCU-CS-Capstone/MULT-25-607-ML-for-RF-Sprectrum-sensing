import React, { useEffect, useRef, useState, useCallback, useMemo } from "react";
import { Box, Button, Typography, CircularProgress, Alert, Divider } from "@mui/material";

const Spectrogram = () => {
  // Refs for canvases and logs.
  const canvasRef = useRef(null);
  const overlayCanvasRef = useRef(null); // overlay for drawing detection boxes
  const logContainerRef = useRef(null);
  const tickCountRef = useRef(0); // counts how many rows have been drawn
  
  // Instead of a single detection event/box, we maintain arrays.
  const detectionEventsRef = useRef([]);  // active detection events (for adding bumps)
  const detectionBoxesRef = useRef([]);     // detection boxes to draw on the overlay
  
  const [classification, setClassification] = useState("Unknown");
  const [connectionStatus, setConnectionStatus] = useState("Idle");
  const [error, setError] = useState(null);
  const [logs, setLogs] = useState([]);
  const [simulate, setSimulate] = useState(false);
  
  const drawTimeout = useRef(null);
  const simulationIntervalRef = useRef(null);

  // Precompute a color map for 256 intensity levels.
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

  // Downsample the PSD (8192 points) to match the canvas width.
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
   * Draws one new row on the spectrogram.
   */
  const drawSpectrogram = useCallback(
    (psd, globalOffset = 0, tailInfo = null) => {
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
            // Shift the image up one row.
            const image = ctx.getImageData(0, 1, width, height - 1);
            ctx.putImageData(image, 0, 0);
          } catch (e) {
            console.error("Error shifting image data:", e);
          }
          // Draw the new row at the bottom.
          ctx.putImageData(imgData, 0, height - 1);
          drawTimeout.current = null;
        });
      } catch (error) {
        console.error("Error in drawSpectrogram:", error);
        setError("Error drawing spectrogram.");
      }
    },
    [colorMap]
  );

  /**
   * Draws all detection boxes on the overlay canvas.
   *
   * For each box (which was created when its event started) we compute:
   * - y_top = (canvas.height - 1) - (currentTick - startTick)
   * - y_bottom = y_top + totalTicks - 1  
   * This makes the box surround all rows drawn for that event.
   *
   * If a box is completely scrolled off (y_bottom < 0) it is removed.
   */
  const drawDetectionBoxes = useCallback(() => {
    if (!canvasRef.current || !overlayCanvasRef.current) return;
    const canvas = canvasRef.current;
    const overlayCanvas = overlayCanvasRef.current;
    const ctx = overlayCanvas.getContext("2d");
    const width = canvas.width;
    const height = canvas.height;
    // Clear the overlay.
    ctx.clearRect(0, 0, width, height);
    const newBoxes = [];
    detectionBoxesRef.current.forEach((box) => {
      const currentTick = tickCountRef.current;
      // Calculate the top edge where the event first appeared.
      const y_top = (height - 1) - (currentTick - box.startTick);
      // The bottom edge is below y_top by the total number of rows (event duration + tail).
      let y_bottom = y_top + box.totalTicks - 1;
      // Skip if the entire box has scrolled off the top.
      if (y_bottom < 0) return;
      // Clamp the visible part.
      const visibleTop = Math.max(0, y_top);
      const visibleBottom = Math.min(height - 1, y_bottom);
      const rectHeight = visibleBottom - visibleTop + 1;
      // Horizontal coordinates are computed from the PSD frequency data.
      const scaleX = width / 8192;
      let x = (box.center - 3 * box.sigma) * scaleX;
      let rectWidth = (6 * box.sigma) * scaleX;
      if (x < 0) {
        rectWidth += x;
        x = 0;
      }
      if (x + rectWidth > width) {
        rectWidth = width - x;
      }
      // Choose stroke color.
      let strokeColor;
      if (box.type === "Bluetooth") {
        strokeColor = "#2196F3"; // blue
      } else if (box.type === "WiFi") {
        strokeColor = "#F44336"; // red
      } else {
        return; // no box for Unknown
      }
      ctx.strokeStyle = strokeColor;
      ctx.lineWidth = 2;
      ctx.strokeRect(x, visibleTop, rectWidth, rectHeight);
      // Keep this box if any part is still visible.
      newBoxes.push(box);
    });
    detectionBoxesRef.current = newBoxes;
  }, []);

  /**
   * Simulation effect.
   *
   * Every 50ms we generate a new PSD row and:
   * - With a 5% chance, start a new detection event (WiFi or Bluetooth).
   *   For each new event, we add its Gaussian bump to the PSD and push a new box.
   * - For each active event, add its bump and decrement its duration.
   * - Draw the spectrogram row and then draw all detection boxes.
   */
  useEffect(() => {
    if (!simulate) return;
    setConnectionStatus("Simulating");
    simulationIntervalRef.current = setInterval(() => {
      // Generate a noisy baseline.
      const psd = Array.from({ length: 8192 }, () => 0.5 + (Math.random() - 0.5) * 0.4);
      let globalOffset = 0;
      let tailInfo = null; // (optional: add tail effects if desired)
      
      // Boost if noise exceeds threshold.
      const noiseThreshold = 0.65;
      const maxVal = Math.max(...psd);
      if (maxVal > noiseThreshold) {
        globalOffset = 30;
      }
      
      // With a 5% chance, start a new event.
      if (Math.random() < 0.05) {
        const simulatedClassification = Math.random() < 0.5 ? "WiFi" : "Bluetooth";
        let eventCenter, eventSigma, eventDuration;
        if (simulatedClassification === "WiFi") {
          eventCenter = 3600 + (Math.random() - 0.5) * 100;
          eventSigma = 60;
        } else {
          eventCenter = 1050 + (Math.random() - 0.5) * 40;
          eventSigma = 30;
        }
        eventDuration = Math.floor(Math.random() * 10) + 10; // duration (in ticks)
        // Add a new detection event.
        detectionEventsRef.current.push({
          type: simulatedClassification,
          center: eventCenter,
          sigma: eventSigma,
          duration: eventDuration,
        });
        // Create a corresponding detection box.
        detectionBoxesRef.current.push({
          type: simulatedClassification,
          center: eventCenter,
          sigma: eventSigma,
          startTick: tickCountRef.current,
          totalTicks: eventDuration + 10, // add extra tail rows if desired
        });
        // Update the classification badge and log the event.
        setClassification(simulatedClassification);
        const timestamp = new Date().toLocaleTimeString();
        setLogs((prevLogs) => [{ timestamp, classification: simulatedClassification }, ...prevLogs]);
      } else {
        // If no new event, set classification to Unknown.
        setClassification("Unknown");
      }
      
      // For every active detection event, add its Gaussian bump.
      detectionEventsRef.current.forEach((event) => {
        const amplitude = 0.7;
        for (let i = 0; i < 8192; i++) {
          const bump = amplitude * Math.exp(-Math.pow(i - event.center, 2) / (2 * Math.pow(event.sigma, 2)));
          psd[i] = Math.min(1, psd[i] + bump);
        }
        // Decrement event duration.
        event.duration--;
      });
      // Remove finished events.
      detectionEventsRef.current = detectionEventsRef.current.filter((event) => event.duration > 0);
      
      // Draw the spectrogram row.
      drawSpectrogram(psd, globalOffset, tailInfo);
      tickCountRef.current++;
      // Draw all detection boxes on the overlay.
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
    if (overlayCanvasRef.current) {
      const overlayCtx = overlayCanvasRef.current.getContext("2d");
      overlayCtx.clearRect(0, 0, overlayCanvasRef.current.width, overlayCanvasRef.current.height);
    }
  };

  const sendMessage = (message) => {
    console.log("Message sent:", message);
  };

  // Return badge colors matching the detection type.
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
