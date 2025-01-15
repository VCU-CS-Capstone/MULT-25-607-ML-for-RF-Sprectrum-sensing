// frontend/src/components/Header.jsx
import React from "react";
import AppBar from "@mui/material/AppBar";
import Toolbar from "@mui/material/Toolbar";
import Typography from "@mui/material/Typography";

const Header = () => {
  return (
    <AppBar position="static" color="primary">
      <Toolbar>
        <Typography variant="h6" component="div">
          SDR Spectrogram Analyzer
        </Typography>
      </Toolbar>
    </AppBar>
  );
};

export default Header;
