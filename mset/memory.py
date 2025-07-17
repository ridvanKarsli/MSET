# === memory.py ===
# mset/memory.py
import numpy as np
from typing import Any

# Verilerden maksimum boyutta bir hafıza bankası oluşturur
# data: Girdi veri dizisi
# max_size: Hafızada tutulacak maksimum örnek sayısı
def create_memory(data: np.ndarray, max_size: int = 100) -> np.ndarray:
    """
    Verilen veri dizisinden maksimum boyutta bir hafıza bankası oluşturur.
    Eğer veri boyutu max_size'dan büyükse, rastgele örnekler seçilir.
    data: Girdi veri dizisi
    max_size: Hafızada tutulacak maksimum örnek sayısı
    Geriye hafıza dizisini döndürür.
    """
    if len(data) > max_size:
        idx = np.random.choice(len(data), max_size, replace=False)  # Rastgele örnek seçimi
        return data[idx]
    return data.copy()  # Veri boyutu uygunsa doğrudan kopyala
