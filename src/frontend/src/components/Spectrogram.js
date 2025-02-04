import React, { useEffect, useRef, useState, useCallback } from "react";
import { Box, Button, Typography, CircularProgress, Alert } from "@mui/material";

const Spectrogram = () => {
  const canvasRef = useRef(null);
  const [classification, setClassification] = useState("Unknown");
  const [connectionStatus, setConnectionStatus] = useState("Connecting");
  const [error, setError] = useState(null);
  const socketRef = useRef(null);
  const drawTimeout = useRef(null);

  // Function to create a color map (jet-like)
  const createColorMap = () => {
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
    console.log("Color map created with", colMap.length, "entries.");
    return colMap;
  };

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

  // Spectrogram drawing logic with useCallback
  const drawSpectrogram = useCallback((psd) => {
    try {
      console.log("Starting to draw spectrogram");
      if (!canvasRef.current) {
        console.warn("Canvas reference is missing.");
        return;
      }

      if (
        !Array.isArray(psd) ||
        psd.length !== 8192 ||
        psd.some((val) => typeof val !== "number" || val < 0 || val > 1)
      ) {
        console.warn("Invalid PSD data received.");
        setError("Invalid PSD data format.");
        return;
      }

      const canvas = canvasRef.current;
      const ctx = canvas.getContext("2d");
      const width = canvas.width;
      const height = canvas.height;

      console.log("Drawing spectrogram with canvas size:", width, height);

      // Throttle drawing using requestAnimationFrame
      if (drawTimeout.current) return;

      drawTimeout.current = requestAnimationFrame(() => {
        // Downsample PSD data to match canvas width
        const downsampled = downsamplePsd(psd, width);
        console.log("Downsampled PSD length:", downsampled.length);
        console.log("Sample downsampled PSD values:", downsampled.slice(0, 10));

        // Create ImageData for one line
        const imgData = ctx.createImageData(width, 1);
        const colMap = createColorMap();

        for (let i = 0; i < width; i++) {
          const intensity = Math.min(255, Math.floor(downsampled[i] * 255));
          const [r, g, b, a] = colMap[intensity];
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

        console.log("Spectrogram line drawn.");

        // Reset timeout
        drawTimeout.current = null;
      });
    } catch (error) {
      console.error("Error in drawSpectrogram:", error);
      setError("Error drawing spectrogram.");
    }
  }, []);

  // WebSocket connection setup with useCallback dependencies
  useEffect(() => {
    console.log("Establishing WebSocket connection...");
    const ws = new WebSocket("ws://localhost:8000/ws");

    ws.onopen = () => {
      console.log("Connected to WebSocket server");
      setConnectionStatus("Connected");
      setError(null);
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        const { psd, classification } = data;

        console.log("Received PSD data length:", psd.length);
        console.log("Sample PSD values:", psd.slice(0, 10)); // Log first 10 values for brevity

        // Validate PSD data
        if (
          !Array.isArray(psd) ||
          psd.length !== 8192 ||
          psd.some((val) => typeof val !== "number" || val < 0 || val > 1)
        ) {
          console.warn("Invalid PSD data received.");
          setError("Invalid PSD data format.");
          return;
        }

        // Process and draw the PSD data
        drawSpectrogram(psd);

        // Update classification
        setClassification(classification);
      } catch (error) {
        console.error("Error parsing WebSocket message:", error);
        setError("Received malformed data.");
      }
    };

    ws.onerror = (error) => {
      console.error("WebSocket error:", error);
      setError("WebSocket encountered an error.");
    };

    ws.onclose = (event) => {
      if (event.wasClean) {
        console.log(
          `WebSocket connection closed cleanly, code=${event.code} reason=${event.reason}`
        );
        setConnectionStatus("Disconnected");
      } else {
        console.warn("WebSocket connection died unexpectedly");
        setError("WebSocket connection lost.");
      }
    };

    socketRef.current = ws; // Assign to ref

    return () => {
      if (socketRef.current) {
        socketRef.current.close();
        console.log("WebSocket connection closed.");
      }
    };
  }, [drawSpectrogram]);

  // Function to handle clearing the spectrogram
  const handleClear = () => {
    if (!canvasRef.current) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    ctx.fillStyle = "#000000";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    console.log("Spectrogram cleared.");
  };

  // Function to send messages via WebSocket
  const sendMessage = (message) => {
    if (socketRef.current && socketRef.current.readyState === WebSocket.OPEN) {
      socketRef.current.send(JSON.stringify(message));
      console.log("Message sent via WebSocket:", message);
    } else {
      console.log("WebSocket is not open.");
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
      <Box
        sx={{
          position: "relative",
          width: "90%",
          maxWidth: "800px",
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

        {connectionStatus === "Connecting" && (
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
              Connecting...
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
          onClick={() => drawSpectrogram(Array(8192).fill(0.5))}
        >
          Draw Mock Data
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
  );
};

export default Spectrogram;