"""PERISIM — Perioperative Similarity (ver SPECS.md)."""

import warnings

__version__ = "0.3.0"

# Aviso inocuo de scikit-learn cuando LightGBM predice sobre arrays sin nombres de columna (miles por SAGE).
warnings.filterwarnings("ignore", message="X does not have valid feature names")
