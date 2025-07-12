# === memory.py ===
# mset/memory.py
import numpy as np
from typing import Any

def create_memory(data: np.ndarray, max_size: int = 100) -> np.ndarray:
    """
    Create a memory bank from the given data with a maximum size.
    If data is larger than max_size, randomly sample without replacement.
    Args:
        data (np.ndarray): Input data array.
        max_size (int): Maximum number of samples to keep.
    Returns:
        np.ndarray: Memory array.
    """
    if len(data) > max_size:
        idx = np.random.choice(len(data), max_size, replace=False)
        return data[idx]
    return data.copy()
