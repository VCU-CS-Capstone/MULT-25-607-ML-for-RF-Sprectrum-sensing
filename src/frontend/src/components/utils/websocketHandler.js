export const createWebSocketConnection = (url, handlers) => {
  const ws = new WebSocket(url);

  ws.onopen = handlers.onOpen;
  ws.onmessage = handlers.onMessage;
  ws.onerror = handlers.onError;
  ws.onclose = handlers.onClose;

  return ws;
};
