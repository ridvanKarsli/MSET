# === mset_predictor.py ===
# mset/mset_predictor.py
import numpy as np
from typing import Any

# MSET algoritması ile bir gözlem için beklenen değeri tahmin eder
# memory: Normal veri hafızası
# obs: Tahmin edilecek gözlem
def mset_predict(memory: np.ndarray, obs: np.ndarray) -> np.ndarray:
    """
    MSET algoritması ile bir gözlem için beklenen değeri tahmin eder.
    memory: Normal veri hafızası (numpy array)
    obs: Tahmin edilecek gözlem (numpy array)
    Geriye tahmini değer (numpy array) döndürür.
    """
    distances = np.linalg.norm(memory - obs, axis=1)  # Hafızadaki her örnek ile gözlem arasındaki mesafeler
    weights = 1 / (distances + 1e-6)  # Mesafeye göre ağırlıklar (bölme hatası için küçük sabit eklenir)
    weights /= weights.sum()  # Ağırlıklar normalize edilir
    return np.dot(weights, memory)  # Ağırlıklı ortalama ile tahmin döndürülür