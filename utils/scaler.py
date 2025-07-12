# === scaler.py ===
# utils/scaler.py
from sklearn.preprocessing import StandardScaler
import pandas as pd
from typing import List, Tuple
import numpy as np

def scale_data(pre_df: pd.DataFrame, post_df: pd.DataFrame, features: List[str]) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Scale pre and post data using StandardScaler fit on pre_df.
    Args:
        pre_df (pd.DataFrame): Pre-maintenance data.
        post_df (pd.DataFrame): Post-maintenance data.
        features (List[str]): List of feature column names.
    Returns:
        Tuple[np.ndarray, np.ndarray, StandardScaler]: Scaled pre, scaled post, and scaler object.
    """
    scaler = StandardScaler()
    pre_scaled = scaler.fit_transform(pre_df[features])
    post_scaled = scaler.transform(post_df[features])
    return pre_scaled, post_scaled, scaler