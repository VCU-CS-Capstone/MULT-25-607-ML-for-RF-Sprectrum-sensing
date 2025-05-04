import React from "react";
import { Box, Typography } from "@mui/material";

function DetectionInfo({ currentDetection, detectionRange }) {
  return (
    <Box sx={{ display: "flex", justifyContent: "space-between", mt: 1 }}>
      <Typography variant="body1">Detection: {currentDetection}</Typography>
      {currentDetection !== "None" && (
        <Typography variant="body1">
          Range: {detectionRange.start} - {detectionRange.end}
        </Typography>
      )}
    </Box>
  );
}

export default DetectionInfo;
