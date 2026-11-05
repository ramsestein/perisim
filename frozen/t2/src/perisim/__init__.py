"""PERISIM — Perioperative Similarity (see SPECS.md)."""

import warnings

__version__ = "0.3.0"

# Harmless scikit-learn warning when LightGBM predicts on arrays without column names (thousands per SAGE run).
warnings.filterwarnings("ignore", message="X does not have valid feature names")
