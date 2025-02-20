// server.js
const WebSocket = require('ws');

// Create a WebSocket server listening on port 8080
const wss = new WebSocket.Server({ port: 8080 });

console.log('WebSocket server started on port 8080');

wss.on('connection', (ws) => {
  console.log('Client connected.');

  // Function to create a test message
  const sendTestMessage = () => {
    // Create an array of 2048 PSD points with random values between 0 and 1
    const psd = Array.from({ length: 2048 }, () => Math.random());

    // Determine detection type:
    // 0: no detection, 1: WiFi, 2: Bluetooth
    // Here we simulate a detection with a 10% chance.
    const detectionType = Math.random() < 0.1 ? (Math.random() < 0.5 ? 1 : 2) : 0;
    
    // For a detected event, pick a random frequency range; otherwise, use default 0 and 2047.
    let startFreq = 0;
    let endFreq = 2047;
    if (detectionType !== 0) {
      startFreq = Math.floor(Math.random() * 2048);
      endFreq = Math.floor(Math.random() * (2048 - startFreq)) + startFreq;
    }
    
    // Combine PSD data with detection info to form a 2051-length array
    const messageArray = [...psd, detectionType, startFreq, endFreq];
    
    // Send the message as a JSON string
    ws.send(JSON.stringify(messageArray));
  };

  // Send a test message every 100ms (adjust as needed)
  const interval = setInterval(sendTestMessage, 100);

  ws.on('close', () => {
    clearInterval(interval);
    console.log('Client disconnected.');
  });
});
