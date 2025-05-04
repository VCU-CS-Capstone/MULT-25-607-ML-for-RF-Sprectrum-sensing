import React, { useState } from "react";
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogContentText,
  DialogActions,
  Button,
  TextField,
  Box,
} from "@mui/material";

function ServerAddressModal({
  open,
  onSave,
  defaultAddress = "ws://localhost:3030",
}) {
  const [serverAddress, setServerAddress] = useState(defaultAddress);
  const [error, setError] = useState("");

  const handleSubmit = () => {
    // Validate WebSocket URL
    if (!serverAddress.trim()) {
      setError("Server address cannot be empty");
      return;
    }

    if (
      !serverAddress.startsWith("ws://") &&
      !serverAddress.startsWith("wss://")
    ) {
      setError("Address must start with ws:// or wss://");
      return;
    }

    onSave(serverAddress);
  };

  return (
    <Dialog open={open} onClose={() => {}} maxWidth="sm" fullWidth>
      <DialogTitle>Enter WebSocket Server Address</DialogTitle>
      <DialogContent>
        <DialogContentText>
          Please enter the address of the WebSocket server to connect to for
          spectrogram data.
        </DialogContentText>
        <Box sx={{ mt: 2 }}>
          <TextField
            autoFocus
            label="Server Address"
            fullWidth
            value={serverAddress}
            onChange={(e) => {
              setServerAddress(e.target.value);
              setError("");
            }}
            error={!!error}
            helperText={error || "Example: ws://localhost:3030"}
            margin="normal"
            variant="outlined"
          />
        </Box>
      </DialogContent>
      <DialogActions>
        <Button onClick={handleSubmit} color="primary" variant="contained">
          Connect
        </Button>
      </DialogActions>
    </Dialog>
  );
}

export default ServerAddressModal;
