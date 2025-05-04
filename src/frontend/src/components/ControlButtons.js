import React from "react";
import { Box, Button, CircularProgress } from "@mui/material";

function ControlButtons({
  isActive,
  setIsActive,
  isClearing,
  handleClear,
  isConnecting,
}) {
  return (
    <Box
      sx={{
        display: "flex",
        gap: 1,
        justifyContent: "center",
        flexWrap: "wrap",
      }}
    >
      {isConnecting ? (
        <Button
          variant="contained"
          color="primary"
          disabled
          size="small"
          sx={{
            minWidth: 100,
            height: 36,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
          }}
        >
          <CircularProgress size={20} color="inherit" />
        </Button>
      ) : (
        <Button
          variant="contained"
          color={isActive ? "error" : "success"}
          onClick={() => setIsActive(!isActive)}
          disabled={isClearing}
          size="small"
          sx={{ minWidth: 100 }}
        >
          {isActive ? "Disconnect" : "Connect"}
        </Button>
      )}

      <Button
        variant="outlined"
        color="secondary"
        onClick={handleClear}
        disabled={isClearing || isConnecting}
        size="small"
        sx={{ minWidth: 100 }}
      >
        Clear
      </Button>
    </Box>
  );
}

export default ControlButtons;
