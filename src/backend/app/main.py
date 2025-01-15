import logging
import asyncio
import json
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from app.models.model_loader import load_model
from app.sdr.sdr_stream import SDRStream
from app.utils.websocket_manager import WebSocketManager
from app.configs.settings import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Log current simulation settings
logger.info(f"USE_MOCK_SDR: {settings.use_mock_sdr}")
logger.info(f"USE_FILE_SDR: {settings.use_file_sdr}")
logger.info(f"USE_PSD_SIMULATOR: {settings.use_psd_simulator}")

app = FastAPI()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize WebSocket Manager
ws_manager = WebSocketManager()

# Load the ML model
model = load_model(settings.model_path)

# Initialize SDR Stream
sdr_stream = SDRStream(model, ws_manager, settings)
asyncio.create_task(sdr_stream.start_stream())

@app.websocket(settings.websocket_endpoint)
async def websocket_endpoint(websocket: WebSocket):
    await ws_manager.connect(websocket)
    try:
        while True:
            # Optionally handle incoming messages from frontend
            data = await websocket.receive_text()
            # Process messages if necessary
    except Exception as e:
        logger.error(f"WebSocket connection error: {e}")
    finally:
        await ws_manager.disconnect(websocket)