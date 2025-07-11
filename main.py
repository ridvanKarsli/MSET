# === main.py ===
import pandas as pd
import numpy as np
from datetime import timedelta
from sklearn.metrics import mean_squared_error
from mset.mset_predictor import mset_predict
from utils.scaler import scale_data

# === AYARLAR ===
excel_path = 'data/Ünite 2 IDF A Bakım değerlendirme verileri copy.xlsx'
makine_adi = "ÜNİTE 2 IDF A"
rmse_esik = 2.0
dogruluk_esik = 70.0
zaman_araliklari_saat = [24, 12, 6, 3]  # saatlik analiz aralıkları

# === VERİ OKUMA ===
pre_df = pd.read_excel(excel_path, sheet_name='Bakım Öncesi IDF A Verileri', header=1)
post_df = pd.read_excel(excel_path, sheet_name='Bakım Sonrası IDF A Verileri', header=1)
pre_df.columns = pre_df.columns.str.strip()
post_df.columns = post_df.columns.str.strip()

features = [
    'Klepe Pozisyonu', 'Akım (amper)', 'X vibrasyonu', 'Y vibrasyonu',
    'Motor Fan Tarafı Yatak Sıcaklığı', 'Motor Arka Yatak Sıcaklığı', 'Yağ Tankı Sıcaklığı',
    'Fan Yatak Sıcaklığı 1', 'Fan Yatak Sıcaklığı 2', 'Fan Yatak Sıcaklığı 3',
    'Fan Yatak Sıcaklığı 4', 'Fan Yatak Sıcaklığı 5', 'Fan Yatak Sıcaklığı 6',
    'Fan Yatak Sıcaklığı 7', 'Fan Yatak Sıcaklığı 8', 'Fan Yatak Sıcaklığı 9',
    'Motor U-Phase Sargı Sıcaklığı 1', 'Motor U-Phase Sargı Sıcaklığı 2',
    'Motor V-Phase Sargı Sıcaklığı 1', 'Motor V-Phase Sargı Sıcaklığı 2',
    'Motor W-Phase Sargı Sıcaklığı 1', 'Motor W-Phase Sargı Sıcaklığı 2'
]

def convert_comma_to_dot(df):
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = pd.to_numeric(df[col].str.replace(',', '.').str.replace(' ', ''), errors='coerce')
    return df

pre_df['Tarih'] = pd.to_datetime(pre_df['Tarih'], errors='coerce')
post_df['Tarih'] = pd.to_datetime(post_df['Tarih'], errors='coerce')

pre_df = convert_comma_to_dot(pre_df)
post_df = convert_comma_to_dot(post_df)
pre_df.dropna(inplace=True)
post_df.dropna(inplace=True)

for f in features.copy():
    if pre_df[f].nunique() <= 1:
        print(f"[UYARI] Sabit sütun çıkarıldı: {f}")
        features.remove(f)

pre_scaled, _, scaler = scale_data(pre_df, post_df, features)
memory = pre_scaled.copy()

# === OTOMATİK BAKIM TARİHİ BELİRLEME ===
bakim_tarihi = post_df['Tarih'].max() - timedelta(minutes=1)  # son veri zamanından biraz öncesi
print(f"Analiz için referans bakım tarihi: {bakim_tarihi}")

# === ZAMANSAL ANALİZ (SAATLİK) ===
for saat in zaman_araliklari_saat:
    alt_sinir = bakim_tarihi - timedelta(hours=saat)
    dilim_df = post_df[(post_df['Tarih'] >= alt_sinir) & (post_df['Tarih'] < bakim_tarihi)]
    if dilim_df.empty:
        print(f"[{saat} SAAT ÖNCE] Veri bulunamadı.")
        continue

    _, post_scaled_segment, _ = scale_data(pre_df, dilim_df, features)
    valid_rows = ~np.isnan(post_scaled_segment).any(axis=1)
    segment_scaled = post_scaled_segment[valid_rows]

    predictions = np.array([mset_predict(memory, obs) for obs in segment_scaled])
    rmse = np.sqrt(mean_squared_error(segment_scaled, predictions))
    mean_val = np.mean(np.abs(segment_scaled))
    accuracy = 100 * (1 - rmse / mean_val) if mean_val > 1e-3 else None

    print(f"\n[{saat} SAAT ÖNCE]")
    print(f"RMSE: {rmse:.4f}")
    print(f"Doğruluk: {accuracy:.2f}%" if accuracy is not None else "Doğruluk hesaplanamadı.")

# === TAM ANALİZ ===
filtered_post_scaled = scale_data(pre_df, post_df, features)[1]
valid_rows = ~np.isnan(filtered_post_scaled).any(axis=1)
filtered_post_scaled = filtered_post_scaled[valid_rows]
predictions = np.array([mset_predict(memory, obs) for obs in filtered_post_scaled])
rmse = np.sqrt(mean_squared_error(filtered_post_scaled, predictions))
mean_val = np.mean(np.abs(filtered_post_scaled))
accuracy = 100 * (1 - rmse / mean_val) if mean_val > 1e-3 else None
feature_rmse = np.sqrt(((filtered_post_scaled - predictions) ** 2).mean(axis=0))

print(f"\n[GENEL ANALİZ] Sistem: {makine_adi}")
print(f"RMSE: {rmse:.4f}")
print(f"Doğruluk: {accuracy:.2f}%" if accuracy is not None else "Doğruluk hesaplanamadı.")

if accuracy is not None and accuracy < dogruluk_esik:
    print(f"[Uyarı] Anormal davranış tespit edildi. Bakım gerekebilir.")
else:
    print("Sistem normal çalışıyor.")

sorted_features = sorted(zip(features, feature_rmse), key=lambda x: x[1], reverse=True)
print("\nEn çok sapma gösteren sensörler:")
for f, e in sorted_features[:3]:
    print(f"- {f}: {e:.2f}")

print("\nTüm sensör farkları (RMSE):")
for f, e in sorted_features:
    print(f"{f}: {e:.4f}")