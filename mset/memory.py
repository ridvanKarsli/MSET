# === memory.py ===
# mset/memory.py
import numpy as np

def create_memory(data, max_size=100):
    if len(data) > max_size:
        idx = np.random.choice(len(data), max_size, replace=False)
        return data[idx]
    return data.copy()
