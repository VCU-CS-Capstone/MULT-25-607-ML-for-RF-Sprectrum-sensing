"""
mlrf: Machine Learning for Radio Frequency

This package provides data sources, classifiers, and server logic for
RF signal classification (e.g., WiFi vs Bluetooth) using machine learning.

Logging is configured here and used throughout the package.
"""

import logging
import os

LOG_LEVEL = os.getenv("MLRF_LOG_LEVEL", "INFO").upper()

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

logger = logging.getLogger("mlrf")
