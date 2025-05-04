import React, {
  useEffect,
  useRef,
  useState,
  useMemo,
  useCallback,
} from "react";
import {
  Card,
  CardContent,
  Box,
  IconButton,
  Tooltip,
  Paper,
} from "@mui/material";
import FullscreenIcon from "@mui/icons-material/Fullscreen";
import FullscreenExitIcon from "@mui/icons-material/FullscreenExit";
import { getDetectionLabel } from "./utils/classify";
import { SpectrogramDrawer } from "./utils/drawingUtils";
import { createWebSocketManager } from "./utils/websocketHandler";
import { generateColorMap } from "./utils/colorMap";
import SpectrogramCanvas from "./SpectrogramCanvas";
import SignalRangeSlider from "./SignalRangeSlider";
import PixelSizeSlider from "./PixelSizeSlider";
import ConnectionStatus from "./ConnectionStatus";
import ControlButtons from "./ControlButtons";
import DetectionInfo from "./DetectionInfo";
import ErrorModal from "./ErrorModal";
import useWindowSize from "./utils/useWindowSize";
import ServerAddressModal from "./ServerAddressModal";

const CARD_RADIUS = 4;
const MAX_CONTAINER_WIDTH = 1920;
const DEFAULT_SERVER_ADDRESS = "ws://localhost:3030";

function getLargest16x9Rect(maxWidth, maxHeight) {
  let width = maxWidth;
  let height = Math.round((width * 9) / 16);
  if (height > maxHeight) {
    height = maxHeight;
    width = Math.round((height * 16) / 9);
  }
  return { width, height };
}

export default function Spectrogram() {
  // Responsive window size
  const { width: windowWidth, height: windowHeight } = useWindowSize();

  // Server connection
  const [serverAddressModalOpen, setServerAddressModalOpen] = useState(true);
  const [serverAddress, setServerAddress] = useState(DEFAULT_SERVER_ADDRESS);

  // Fullscreen state
  const [isFullscreen, setIsFullscreen] = useState(false);
  const canvasContainerRef = useRef(null);

  // Compute canvas 16:9 size within 80% height or 1100px width
  const MAX_CONTAINER_HEIGHT = Math.floor(windowHeight * 0.7);
  const { width: canvasWidth, height: canvasHeight } = getLargest16x9Rect(
    Math.min(windowWidth - 32, MAX_CONTAINER_WIDTH),
    Math.min(windowHeight - 32, MAX_CONTAINER_HEIGHT),
  );

  // Refs
  const canvasRef = useRef(null);
  const spectrogramDrawerRef = useRef(null);
  const websocketManagerRef = useRef(null);

  // State
  const [connectionStatus, setConnectionStatus] = useState("Idle");
  const [error, setError] = useState(null);
  const [errorModalOpen, setErrorModalOpen] = useState(false);
  const [isActive, setIsActive] = useState(false);
  const [isConnecting, setIsConnecting] = useState(false);
  const [isClearing, setIsClearing] = useState(false);
  const [isConnected, setIsConnected] = useState(false);
  const [currentDetection, setCurrentDetection] = useState("None");
  const [detectionRange, setDetectionRange] = useState({ start: 0, end: 0 });
  const lastUsedRangeRef = useRef([-110, -70]);
  const [rangeDbm, setRangeDbm] = useState(lastUsedRangeRef.current);
  const [pixelSize, setPixelSize] = useState(4);

  // Color map (memoized)
  const colorMap = useMemo(() => generateColorMap(), []);

  // Handle server address save
  const handleServerAddressSave = (address) => {
    setServerAddress(address);
    setServerAddressModalOpen(false);
    // Reset connection state when server changes
    if (websocketManagerRef.current) {
      websocketManagerRef.current.disconnect();
    }
  };

  // ---- Handlers ----

  function handleRangeChange(_, newValue) {
    setRangeDbm(newValue);
    lastUsedRangeRef.current = newValue;
  }

  function handlePixelSizeChange(_, newValue) {
    setPixelSize(newValue);
  }

  // WebSocket message handler
  function handleWebSocketMessage(event) {
    try {
      const data = JSON.parse(event.data);
      if (!Array.isArray(data) || data.length !== 2051)
        throw new Error("Invalid data format");
      const psdData = data.slice(0, 2048);
      const detectionType = data[2048];
      const startIndex = data[2049];
      const endIndex = data[2050];

      setCurrentDetection(getDetectionLabel(detectionType));
      setDetectionRange({ start: startIndex, end: endIndex });

      spectrogramDrawerRef.current
        ?.draw(
          psdData,
          lastUsedRangeRef.current,
          [startIndex, endIndex],
          detectionType,
        )
        .catch((e) => setError("Drawing error: " + e.message));
    } catch (e) {
      setError("Invalid data received: " + e.message);
    }
  }

  function handleClear() {
    setIsClearing(true);
    websocketManagerRef.current?.disconnect();
    spectrogramDrawerRef.current?.clear();
    setError(null);
    setCurrentDetection("None");
    setDetectionRange({ start: 0, end: 0 });
    setTimeout(() => setIsClearing(false), 100);
  }

  const handleFullscreenToggle = useCallback(() => {
    const el = canvasContainerRef.current;
    if (!el) return;
    if (!document.fullscreenElement) el.requestFullscreen();
    else document.exitFullscreen();
  }, []);

  // ---- Effects ----

  // Initialize SpectrogramDrawer
  useEffect(() => {
    if (canvasRef.current && colorMap) {
      spectrogramDrawerRef.current = new SpectrogramDrawer(
        canvasRef.current,
        colorMap,
        setError,
        pixelSize,
      );
      return () => spectrogramDrawerRef.current?.cleanup();
    }
  }, [colorMap, pixelSize, canvasWidth, canvasHeight]);

  // Update PSD range on slider change
  useEffect(() => {
    spectrogramDrawerRef.current
      ?.updateRange(lastUsedRangeRef.current)
      .catch((e) => setError("Failed to update range: " + e.message));
  }, [rangeDbm]);

  // Show error modal if `error` changes
  useEffect(() => {
    if (error) setErrorModalOpen(true);
  }, [error]);

  // Manage WebSocket lifecycle
  useEffect(() => {
    if (!websocketManagerRef.current) {
      websocketManagerRef.current = createWebSocketManager(
        serverAddress, // Use the state variable instead of hardcoded address
        {
          onOpen: () => {
            setConnectionStatus("Connected");
            setIsConnected(true);
            setError(null);
            setIsConnecting(false);
          },
          onClose: () => {
            setConnectionStatus("Disconnected");
            setIsConnected(false);
            setIsConnecting(false);
          },
          onError: (evt) => {
            let msg = "WebSocket error";
            if (evt?.target?.readyState === WebSocket.CLOSED)
              msg = `Failed to connect to ${serverAddress}`;
            setError(msg);
            setIsActive(false);
            setIsConnecting(false);
          },
          onMessage: handleWebSocketMessage,
        },
      );
    }

    if (isActive && !isClearing) {
      setIsConnecting(true);
      websocketManagerRef.current.connect();
    } else {
      websocketManagerRef.current.disconnect();
      setIsConnecting(false);
    }

    return () => {
      websocketManagerRef.current.disconnect();
      setIsConnecting(false);
    };
  }, [isActive, isClearing, serverAddress]); // Add serverAddress as dependency

  // Listen for fullscreen changes
  useEffect(() => {
    const onFS = () => setIsFullscreen(!!document.fullscreenElement);
    document.addEventListener("fullscreenchange", onFS);
    return () => document.removeEventListener("fullscreenchange", onFS);
  }, []);

  // ---- Layout values ----

  const displayCanvasWidth = isFullscreen ? windowWidth : canvasWidth;
  const displayCanvasHeight = isFullscreen ? windowHeight : canvasHeight;
  const overlayControlWidth = 420;

  // ---- Render ----

  return (
    <Box
      sx={{
        minHeight: "100vh",
        width: "100vw",
        bgcolor: "background.default",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        p: 0,
        m: 0,
      }}
    >
      <ServerAddressModal
        open={serverAddressModalOpen}
        onSave={handleServerAddressSave}
        defaultAddress={DEFAULT_SERVER_ADDRESS}
      />
      <Box
        ref={canvasContainerRef}
        sx={{
          width: displayCanvasWidth,
          height: isFullscreen ? displayCanvasHeight : "auto",
          bgcolor: isFullscreen ? "black" : "background.paper",
          borderRadius: isFullscreen ? 0 : CARD_RADIUS,
          boxShadow: isFullscreen ? 0 : 3,
          p: isFullscreen ? 0 : { xs: 1, sm: 2 },
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          gap: isFullscreen ? 0 : 2,
          position: "relative",
          m: 0,
        }}
      >
        {/* Fullscreen toggle */}
        <Box sx={{ position: "absolute", top: 8, right: 8, zIndex: 10 }}>
          <Tooltip title={isFullscreen ? "Exit Fullscreen" : "Fullscreen"}>
            <IconButton
              size="small"
              onClick={handleFullscreenToggle}
              color="primary"
              sx={{
                bgcolor: "background.paper",
                borderRadius: "50%",
                opacity: 0.8,
                "&:hover": { opacity: 1 },
              }}
            >
              {isFullscreen ? <FullscreenExitIcon /> : <FullscreenIcon />}
            </IconButton>
          </Tooltip>
        </Box>

        {/* Spectrogram Canvas */}
        <SpectrogramCanvas
          canvasRef={canvasRef}
          width={displayCanvasWidth}
          height={displayCanvasHeight}
          sx={{
            borderRadius: isFullscreen ? 0 : CARD_RADIUS,
            boxShadow: isFullscreen ? 0 : 1,
            width: "100%",
            height: "100%",
            maxWidth: "100vw",
            maxHeight: "100vh",
            mb: isFullscreen ? 0 : 2,
            background: "black",
            p: 0,
            m: 0,
          }}
        />

        {/* Controls */}
        {isFullscreen ? (
          <Paper
            elevation={6}
            sx={{
              position: "fixed",
              left: 24,
              top: 24,
              width: overlayControlWidth,
              bgcolor: "rgba(30,30,30,0.92)",
              color: "white",
              borderRadius: CARD_RADIUS,
              p: 2,
              zIndex: 1201,
              boxShadow: 6,
            }}
          >
            <Box sx={{ display: "flex", flexDirection: "column", gap: 1 }}>
              <SignalRangeSlider
                rangeDbm={rangeDbm}
                onChange={handleRangeChange}
              />
              <PixelSizeSlider
                pixelSize={pixelSize}
                onChange={handlePixelSizeChange}
              />
              <ConnectionStatus
                isClearing={isClearing}
                connectionStatus={connectionStatus}
                isConnected={isConnected}
              />
              <ControlButtons
                isActive={isActive}
                setIsActive={setIsActive}
                isClearing={isClearing}
                handleClear={handleClear}
                isConnecting={isConnecting}
              />
              <DetectionInfo
                currentDetection={currentDetection}
                detectionRange={detectionRange}
              />
            </Box>
          </Paper>
        ) : (
          <Card
            sx={{
              width: "100%",
              bgcolor: "background.default",
              boxShadow: 2,
              borderRadius: CARD_RADIUS,
              p: 0,
            }}
          >
            <CardContent sx={{ p: 1 }}>
              <Box sx={{ display: "flex", flexDirection: "column", gap: 0.5 }}>
                <SignalRangeSlider
                  rangeDbm={rangeDbm}
                  onChange={handleRangeChange}
                />
                <PixelSizeSlider
                  pixelSize={pixelSize}
                  onChange={handlePixelSizeChange}
                />
                <ConnectionStatus
                  isClearing={isClearing}
                  connectionStatus={connectionStatus}
                  isConnected={isConnected}
                />
                <ControlButtons
                  isActive={isActive}
                  setIsActive={setIsActive}
                  isClearing={isClearing}
                  handleClear={handleClear}
                  isConnecting={isConnecting}
                />
                <DetectionInfo
                  currentDetection={currentDetection}
                  detectionRange={detectionRange}
                />
              </Box>
            </CardContent>
          </Card>
        )}

        {/* Error Modal */}
        <ErrorModal
          open={errorModalOpen}
          message={error}
          onClose={() => setErrorModalOpen(false)}
        />
      </Box>
    </Box>
  );
}
