import React, { useEffect, useRef, useState, useMemo } from "react";
import { createSpectrogramDrawer } from "./utils/drawingUtils";
import { generateColorMap } from "./utils/colorMap";
import { Box, Card, CardContent, Typography, Button, Slider } from '@mui/material';

const CANVAS_WIDTH = 2048;
const CANVAS_HEIGHT = 1024;
const WEBSOCKET_URL = "ws://localhost:3030";

const DetectionType = {
  NONE: 0,
  WIFI: 1,
  BLUETOOTH: 2,
};

const getDetectionLabel = (type) => {
  switch (type) {
    case DetectionType.WIFI:
      return "WiFi";
    case DetectionType.BLUETOOTH:
      return "Bluetooth";
    default:
      return "None";
  }
};

const Spectrogram = () => {
  // Refs
  const canvasRef = useRef(null);
  const spectrogramDrawerRef = useRef(null);
  const websocketRef = useRef(null);

  // State
  const [connectionStatus, setConnectionStatus] = useState("Idle");
  const [error, setError] = useState(null);
  const [isActive, setIsActive] = useState(false);
  const [isClearing, setIsClearing] = useState(false);
  const [isConnected, setIsConnected] = useState(false);
  const [currentDetection, setCurrentDetection] = useState("None");
  const [detectionRange, setDetectionRange] = useState({ start: 0, end: 0 });
  const lastUsedRangeRef = useRef([-110, -70]);
  const [rangeDbm, setRangeDbm] = useState(lastUsedRangeRef.current);
  const handleRangeChange = (_, newValue) => {
    setRangeDbm(newValue);
    lastUsedRangeRef.current = newValue; // Keep ref in sync
  };

  // Generate color map
  const colorMap = useMemo(() => generateColorMap(), []);

  // Initialize spectrogram drawer
  useEffect(() => {
    if (canvasRef.current && colorMap) {
      spectrogramDrawerRef.current = createSpectrogramDrawer(
        canvasRef.current,
        colorMap,
        setError
      );

      return () => {
        spectrogramDrawerRef.current?.cleanup();
      };
    }
  }, [colorMap]);

  useEffect(() => {
    if (spectrogramDrawerRef.current) {
      // Use the ref to prevent unnecessary updates
      spectrogramDrawerRef.current.updateRange(lastUsedRangeRef.current)
        .catch((error) => {
          console.error("Failed to update range:", error);
        });
    }
  }, [rangeDbm]);

  const handleWebSocketMessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      if (!Array.isArray(data) || data.length !== 2051) {
        throw new Error("Invalid data format");
      }

      const psdData = data.slice(0, 2048);
      const detectionType = data[2048];
      const startIndex = data[2049];
      const endIndex = data[2050];

      // Update detection state
      setCurrentDetection(getDetectionLabel(detectionType));
      setDetectionRange({ start: startIndex, end: endIndex });

      // Use the ref's value to ensure consistency
      spectrogramDrawerRef.current?.draw(
        psdData,
        lastUsedRangeRef.current, // Use ref instead of state to ensure latest value
        [startIndex, endIndex]
      ).catch((error) => {
        console.error("Failed to draw spectrogram:", error);
        setError("Drawing error occurred");
      });
    } catch (error) {
      console.error("Error processing WebSocket message:", error);
      setError("Invalid data received");
    }
  };

  // WebSocket connection management
  const initializeWebSocket = () => {
    if (websocketRef.current) {
      websocketRef.current.close();
    }

    const ws = new WebSocket(WEBSOCKET_URL);

    ws.onopen = () => {
      console.log("WebSocket Connected");
      setConnectionStatus("Connected");
      setIsConnected(true);
      setError(null);
    };

    ws.onclose = () => {
      console.log("WebSocket Disconnected");
      setConnectionStatus("Disconnected");
      setIsConnected(false);
      ws.close();
      if (isActive && !isClearing) {
        websocketRef.current.close();
      }
    };

    ws.onerror = (error) => {
      console.error("WebSocket Error:", error);
      setError("WebSocket connection error");
    };

    ws.onmessage = handleWebSocketMessage;
    websocketRef.current = ws;
  };

  // Handle connection state
  useEffect(() => {
    if (isActive && !isClearing) {
      initializeWebSocket();
    } else if (websocketRef.current) {
      websocketRef.current.close();
    }

    return () => {
      if (websocketRef.current) {
        websocketRef.current.close();
      }
    };
  }, [isActive, isClearing]);

  const handleClear = () => {
    setIsClearing(true);

    if (websocketRef.current) {
      websocketRef.current.close();
    }

    spectrogramDrawerRef.current?.clear();
    setError(null);
    setCurrentDetection("None");
    setDetectionRange({ start: 0, end: 0 });

    setTimeout(() => {
      setIsClearing(false);
    }, 100);
  };

  return (
    <Box
      sx={{
        display: "flex",
        justifyContent: "center",
        alignItems: "center",
        height: "100%",
      }}
    >
      <Card sx={{ width: CANVAS_WIDTH + 40, padding: 2 }}>
        <CardContent>
          <Box sx={{ display: "flex", flexDirection: "column", gap: 2 }}>
            {/* Spectrogram Canvas */}
            <Box sx={{ display: "flex", justifyContent: "center" }}>
              <canvas
                ref={canvasRef}
                width={CANVAS_WIDTH}
                height={CANVAS_HEIGHT}
                style={{ background: "black" }}
              />
            </Box>

            {/* Signal Range Control */}
            <Box sx={{ mt: 1 }}>
              <Typography variant="body2" gutterBottom>
                Signal Range (dBm):
              </Typography>
              <Box sx={{ display: "flex", alignItems: "center", gap: 2 }}>
                <Typography variant="caption">{rangeDbm[0]}</Typography>
                <Slider
                  value={rangeDbm}
                  onChange={handleRangeChange}
                  valueLabelDisplay="auto"
                  min={-200}
                  max={0}
                  sx={{ flex: 1 }}
                />
                <Typography variant="caption">{rangeDbm[1]}</Typography>
              </Box>
            </Box>

            {/* Connection Status */}
            <Box
              sx={{
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
              }}
            >
              <Typography variant="body2" color="textSecondary">
                Status: {isClearing ? "Clearing" : connectionStatus}
                <span
                  style={{
                    display: "inline-block",
                    width: 10,
                    height: 10,
                    borderRadius: "50%",
                    backgroundColor: isConnected ? "#4CAF50" : "#f44336",
                    marginLeft: 10,
                  }}
                />
              </Typography>
            </Box>

            {/* Control Buttons */}
            <Box sx={{ display: "flex", gap: 2, justifyContent: "center" }}>
              <Button
                variant="contained"
                color={isActive ? "error" : "success"}
                onClick={() => setIsActive(!isActive)}
                disabled={isClearing}
              >
                {isActive ? "Disconnect" : "Connect"}
              </Button>

              <Button
                variant="outlined"
                color="secondary"
                onClick={handleClear}
                disabled={isClearing}
              >
                Clear
              </Button>
            </Box>

            {/* Detection Information */}
            <Box sx={{ display: "flex", justifyContent: "space-between", mt: 1 }}>
              <Typography variant="body2">
                Detection: {currentDetection}
              </Typography>
              {currentDetection !== "None" && (
                <Typography variant="body2">
                  Range: {detectionRange.start} - {detectionRange.end}
                </Typography>
              )}
            </Box>

            {/* Error Messages */}
            {error && (
              <Typography variant="body2" color="error" sx={{ mt: 2 }}>
                Error: {error}
              </Typography>
            )}
          </Box>
        </CardContent>
      </Card>
    </Box>
  );
};

export default Spectrogram;
