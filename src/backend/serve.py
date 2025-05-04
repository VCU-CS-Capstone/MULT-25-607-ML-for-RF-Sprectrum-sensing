#!/usr/bin/env python

import argparse
import os
import sys
import joblib
import asyncio
from mlrf.classifier import RFClassifier
from mlrf.server import run_server
from mlrf import logger

def parse_args():
    parser = argparse.ArgumentParser(description="RF Signal Classification Server")
    parser.add_argument("--port", type=int, default=3030, help="WebSocket server port")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="WebSocket server host")
    parser.add_argument("--model", type=str, required=True, help="Path to saved model file")
    parser.add_argument("--hz", type=int, default=30, help="Target update frequency in Hz")
    parser.add_argument("--source", type=str, choices=["h5", "sdr"], default="h5", help="Data source type (h5 or sdr)")
    parser.add_argument("--data_path", type=str, help="Path to H5 file for h5 source")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    return parser.parse_args()

def main():
    args = parse_args()
    if args.debug:
        os.environ["DEBUG"] = "True"
        logger.setLevel("DEBUG")
    if args.source.lower() == "h5" and not args.data_path:
        logger.error("Error: --data_path is required when using h5 source")
        sys.exit(1)
    logger.info(f"Loading model from: {args.model}")
    try:
        loaded_obj = joblib.load(args.model)
        if isinstance(loaded_obj, RFClassifier):
            model = loaded_obj
            logger.info("Loaded RFClassifier model")
        else:
            model = RFClassifier(loaded_obj, feature_method="combined")
            logger.info("Loaded ML model and wrapped in RFClassifier")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        sys.exit(1)
    try:
        asyncio.run(
            run_server(
                args.host, args.port, model, args.hz, args.source, args.data_path
            )
        )
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received: shutting down.")

if __name__ == "__main__":
    main()
