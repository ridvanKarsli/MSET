# === scaler.py ===
# utils/scaler.py
from sklearn.preprocessing import StandardScaler
import pandas as pd
from typing import List, Tuple
import numpy as np

# Verileri ölçeklendirmek için kullanılan fonksiyon
# pre_df: Bakım öncesi veri seti
# post_df: Bakım sonrası veri seti
# features: Ölçeklenecek özelliklerin isim listesi
def scale_data(pre_df: pd.DataFrame, post_df: pd.DataFrame, features: List[str]) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    pre_df ve post_df veri setlerindeki belirtilen özellikleri, pre_df üzerinde eğitilen StandardScaler ile ölçeklendirir.
    pre_df: Bakım öncesi veriler
    post_df: Bakım sonrası veriler
    features: Ölçeklenecek sütun isimleri
    Geriye sırasıyla ölçeklenmiş pre_df, ölçeklenmiş post_df ve scaler nesnesini döndürür.
    """
    scaler = StandardScaler()  # StandardScaler nesnesi oluşturuluyor
    pre_scaled = scaler.fit_transform(pre_df[features])  # Bakım öncesi verilerle scaler eğitiliyor ve ölçekleniyor
    post_scaled = scaler.transform(post_df[features])    # Bakım sonrası veriler aynı scaler ile ölçekleniyor
    return pre_scaled, post_scaled, scaler  # Sonuçlar döndürülüyor