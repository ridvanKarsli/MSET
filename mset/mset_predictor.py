# === mset_predictor.py ===
# mset/mset_predictor.py
import numpy as np
from typing import Any

def mset_predict(memory: np.ndarray, obs: np.ndarray) -> np.ndarray:
    """
    Predict the expected value for an observation using the MSET algorithm.
    Args:
        memory (np.ndarray): Memory bank of normal data.
        obs (np.ndarray): Observation to predict.
    Returns:
        np.ndarray: Predicted value.
    """
    distances = np.linalg.norm(memory - obs, axis=1)
    weights = 1 / (distances + 1e-6)
    weights /= weights.sum()
    return np.dot(weights, memory)