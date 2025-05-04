// classify.js
export function getDetectionLabel(type) {
  switch (type) {
    case 1: return "WiFi";
    case 2: return "Bluetooth";
    default: return "None";
  }
}
