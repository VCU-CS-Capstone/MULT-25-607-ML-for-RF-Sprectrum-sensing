// websocketHandler.js

/**
 * Creates and manages a WebSocket connection.
 * @param {string} url
 * @param {object} handlers - { onOpen, onMessage, onError, onClose }
 * @returns {object} { connect, disconnect, isConnected }
 */
export function createWebSocketManager(url, handlers) {
  let ws = null;
  let connected = false;

  function connect() {
    if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) {
      return;
    }
    ws = new WebSocket(url);

    ws.onopen = (event) => {
      connected = true;
      handlers.onOpen && handlers.onOpen(event);
    };
    ws.onmessage = (event) => {
      handlers.onMessage && handlers.onMessage(event);
    };
    ws.onerror = (event) => {
      handlers.onError && handlers.onError(event);
    };
    ws.onclose = (event) => {
      connected = false;
      handlers.onClose && handlers.onClose(event);
    };
  }

  function disconnect() {
    if (ws) {
      ws.close();
      ws = null;
      connected = false;
    }
  }

  function isConnected() {
    return connected;
  }

  return {
    connect,
    disconnect,
    isConnected,
  };
}
