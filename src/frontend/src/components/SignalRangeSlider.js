import React from "react";
import { Box, Typography, Slider } from "@mui/material";

function SignalRangeSlider({ rangeDbm, onChange }) {
  return (
    <Box sx={{ mt: 0.5 }}>
      <Typography variant="body2" gutterBottom>
        Signal Range (dBm):
      </Typography>
      <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
        <Typography variant="body2">{rangeDbm[0]}</Typography>
        <Slider
          value={rangeDbm}
          onChange={onChange}
          valueLabelDisplay="auto"
          min={-200}
          max={0}
          sx={{ flex: 1, mx: 1 }}
        />
        <Typography variant="body2">{rangeDbm[1]}</Typography>
      </Box>
    </Box>
  );
}

export default SignalRangeSlider;
