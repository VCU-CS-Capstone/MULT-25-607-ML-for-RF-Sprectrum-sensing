// frontend/src/App.jsx
import React from "react";
import Header from "./components/Header";
import Spectrogram from "./components/Spectrogram";

const App = () => {
  return (
    <div>
      <Header />
      <Spectrogram />
    </div>
  );
};

export default App;