
import React from "react";
import { Box, Typography } from "@mui/material";

function ConnectionStatus({ isClearing, connectionStatus, isConnected }) {
  return (
    <Box
      sx={{
        display: "flex",
        justifyContent: "space-between",
        alignItems: "center",
      }}
    >
      <Typography variant="body1" color="textSecondary">
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
  );
}

export default ConnectionStatus;
