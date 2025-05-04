import React from "react";
import { Box } from "@mui/material";

function SpectrogramCanvas({ canvasRef, width, height, sx }) {
  return (
    <Box
      sx={{
        width: "100%",
        display: "flex",
        justifyContent: "center",
        ...sx,
      }}
    >
      <canvas
        ref={canvasRef}
        width={width}
        height={height}
        style={{
          background: "black",
          width: "100%",
          height: "auto",
          borderRadius: 4,
          display: "block",
        }}
      />
    </Box>
  );
}

export default SpectrogramCanvas;
