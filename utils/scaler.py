# === scaler.py ===
# utils/scaler.py
from sklearn.preprocessing import StandardScaler

def scale_data(pre_df, post_df, features):
    scaler = StandardScaler()
    pre_scaled = scaler.fit_transform(pre_df[features])
    post_scaled = scaler.transform(post_df[features])
    return pre_scaled, post_scaled, scaler