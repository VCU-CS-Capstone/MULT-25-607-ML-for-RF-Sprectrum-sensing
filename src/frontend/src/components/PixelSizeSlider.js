import React from "react";
import { Box, Typography, Slider } from "@mui/material";

function PixelSizeSlider({ pixelSize, onChange }) {
  return (
    <Box>
      <Typography variant="body1" gutterBottom>
        Pixel Size:
      </Typography>
      <Box sx={{ display: "flex", alignItems: "center", gap: 2 }}>
        <Typography variant="caption">{pixelSize}</Typography>
        <Slider
          value={pixelSize}
          onChange={onChange}
          valueLabelDisplay="auto"
          min={1}
          max={64}
          step={1}
          sx={{ flex: 1 }}
        />
      </Box>
    </Box>
  );
}

export default PixelSizeSlider;
