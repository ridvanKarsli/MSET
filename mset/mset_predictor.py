# === mset_predictor.py ===
# mset/mset_predictor.py
import numpy as np

def mset_predict(memory, obs):
    distances = np.linalg.norm(memory - obs, axis=1)
    weights = 1 / (distances + 1e-6)
    weights /= weights.sum()
    return np.dot(weights, memory)