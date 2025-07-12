# === main.py ===
import pandas as pd
import numpy as np
from datetime import timedelta
from typing import List, Tuple, Optional
from sklearn.metrics import mean_squared_error
from mset.mset_predictor import mset_predict
from utils.scaler import scale_data

# === SABİTLER ===
EXCEL_PATH = 'data/Ünite 2 IDF A Bakım değerlendirme verileri copy.xlsx'  # Veri dosyasının yolu
STARTUP_SHUTDOWN_PATH = 'data/IDF Verileri.xlsx'  # Başlatma/kapatma verileri
MACHINE_NAME = "ÜNİTE 2 IDF A"  # Analiz yapılacak makine adı
RMSE_THRESHOLD = 2.0  # RMSE eşik değeri
ACCURACY_THRESHOLD = 70.0  # Doğruluk oranı eşiği
TIME_WINDOWS_HOURS = [24, 12, 6, 3]  # Saatlik analiz aralıkları

# Analizde kullanılacak sensör/sütun isimleri
FEATURES = [
    'Klepe Pozisyonu', 'Akım (amper)', 'X vibrasyonu', 'Y vibrasyonu',
    'Motor Fan Tarafı Yatak Sıcaklığı', 'Motor Arka Yatak Sıcaklığı', 'Yağ Tankı Sıcaklığı',
    'Fan Yatak Sıcaklığı 1', 'Fan Yatak Sıcaklığı 2', 'Fan Yatak Sıcaklığı 3',
    'Fan Yatak Sıcaklığı 4', 'Fan Yatak Sıcaklığı 5', 'Fan Yatak Sıcaklığı 6',
    'Fan Yatak Sıcaklığı 7', 'Fan Yatak Sıcaklığı 8', 'Fan Yatak Sıcaklığı 9',
    'Motor U-Phase Sargı Sıcaklığı 1', 'Motor U-Phase Sargı Sıcaklığı 2',
    'Motor V-Phase Sargı Sıcaklığı 1', 'Motor V-Phase Sargı Sıcaklığı 2',
    'Motor W-Phase Sargı Sıcaklığı 1', 'Motor W-Phase Sargı Sıcaklığı 2'
]

# Başlatma/kapatma verilerinde kullanılacak sensör isimleri
STARTUP_FEATURES = [
    'IDF A Amper', 'IDF B Amper', 'IDF A Klepe', 'IDF Klepe B', 
    'IDF A X Vib.', 'IDF A Y Vib.', 'IDF A X Vib.2', 'IDF B Y Vib.'
]

def convert_comma_to_dot(df: pd.DataFrame) -> pd.DataFrame:
    """
    DataFrame'deki ondalık virgülleri noktaya çevirir ve boşlukları kaldırır.
    Özellikle Türkçe Excel dosyalarında ondalık ayraç olarak virgül kullanıldığı için gereklidir.
    """
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = pd.to_numeric(df[col].str.replace(',', '.').str.replace(' ', ''), errors='coerce')
    return df

def read_startup_shutdown_data(excel_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Makine başlatma ve kapatma verilerini okur ve ayrıştırır.
    """
    df = pd.read_excel(excel_path)
    df['Tarih Saat'] = pd.to_datetime(df['Tarih Saat'], errors='coerce')
    df = convert_comma_to_dot(df)
    df.dropna(inplace=True)
    
    # İlk 50 veri başlatma, son 50 veri kapatma olarak kabul edilir
    startup_data = df.head(50)
    shutdown_data = df.tail(50)
    
    print(f"Başlatma verileri: {len(startup_data)} kayıt")
    print(f"Kapatma verileri: {len(shutdown_data)} kayıt")
    
    return startup_data, shutdown_data

def read_and_preprocess_data(excel_path: str, features: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Excel dosyasından bakım öncesi ve sonrası verileri okur, ön işler ve sabit sütunları eler.
    """
    pre_df = pd.read_excel(excel_path, sheet_name='Bakım Öncesi IDF A Verileri', header=1)
    post_df = pd.read_excel(excel_path, sheet_name='Bakım Sonrası IDF A Verileri', header=1)
    pre_df.columns = pre_df.columns.str.strip()
    post_df.columns = post_df.columns.str.strip()
    pre_df['Tarih'] = pd.to_datetime(pre_df['Tarih'], errors='coerce')
    post_df['Tarih'] = pd.to_datetime(post_df['Tarih'], errors='coerce')
    pre_df = convert_comma_to_dot(pre_df)
    post_df = convert_comma_to_dot(post_df)
    pre_df.dropna(inplace=True)
    post_df.dropna(inplace=True)
    # Sabit (değişmeyen) sütunları analizden çıkar
    features_clean = features.copy()
    for f in features.copy():
        if pre_df[f].nunique() <= 1:
            print(f"[UYARI] Sabit sütun çıkarıldı: {f}")
            features_clean.remove(f)
    return pre_df, post_df, features_clean

def create_enhanced_memory(pre_df: pd.DataFrame, startup_data: pd.DataFrame, shutdown_data: pd.DataFrame, features: List[str]) -> np.ndarray:
    """
    Normal çalışma, başlatma ve kapatma verilerini birleştirerek gelişmiş bir hafıza oluşturur.
    """
    # Normal çalışma verilerini ölçekle
    pre_scaled, _, scaler = scale_data(pre_df, pre_df, features)

    # Başlatma/kapatma verilerinin sütunlarını normal çalışma sütunlarına eşleştir
    # Eşleştirme sözlüğü (örnek)
    mapping = {
        'Akım (amper)': 'IDF A Amper',
        'Klepe Pozisyonu': 'IDF A Klepe',
        'X vibrasyonu': 'IDF A X Vib.',
        'Y vibrasyonu': 'IDF A Y Vib.',
        # Diğer sensörler için gerekirse eşleştirme eklenebilir
    }
    # Sadece eşleşen sütunlar üzerinden başlatma/kapatma verisi oluştur
    mapped_cols = {k: v for k, v in mapping.items() if k in features and v in startup_data.columns}
    if not mapped_cols:
        print("Başlatma/kapatma verileri ile normal veriler arasında eşleşen sütun yok!")
        return pre_scaled
    # DataFrame'i eşleşen sütunlara göre yeniden adlandır
    startup_aligned = startup_data[[v for v in mapped_cols.values()]].rename(columns={v: k for k, v in mapped_cols.items()})
    shutdown_aligned = shutdown_data[[v for v in mapped_cols.values()]].rename(columns={v: k for k, v in mapped_cols.items()})
    # Eksik sütunları sıfırla (diğer sensörler için)
    for col in features:
        if col not in startup_aligned.columns:
            startup_aligned[col] = 0.0
        if col not in shutdown_aligned.columns:
            shutdown_aligned[col] = 0.0
    # Sıralamayı garanti et
    startup_aligned = startup_aligned[features]
    shutdown_aligned = shutdown_aligned[features]
    # Aynı scaler ile ölçekle
    startup_scaled = scaler.transform(startup_aligned)
    shutdown_scaled = scaler.transform(shutdown_aligned)
    # Hepsini birleştir
    enhanced_memory = np.vstack([pre_scaled, startup_scaled, shutdown_scaled])
    print(f"Gelişmiş hafıza: {enhanced_memory.shape[0]} kayıt (normal+başlatma+kapatma)")
    return enhanced_memory

def analyze_time_windows(pre_df: pd.DataFrame, post_df: pd.DataFrame, features: List[str],
                        memory: np.ndarray, time_windows: List[int], accuracy_threshold: float) -> Tuple[Optional[int], Optional[float], Optional[float]]:
    """
    Bakım sonrası verileri farklı zaman aralıklarında analiz eder.
    İlk arıza tespit edilen zamanı ve ilgili RMSE/doğruluk değerlerini döndürür.
    """
    first_fault_time = None
    first_fault_rmse = None
    first_fault_acc = None
    for hour in time_windows:
        # Zaman aralığına göre veri dilimini seç
        lower_bound = post_df['Tarih'].max() - timedelta(hours=hour)
        upper_bound = post_df['Tarih'].max() - timedelta(minutes=1)
        window_df = post_df[(post_df['Tarih'] >= lower_bound) & (post_df['Tarih'] < upper_bound)]
        if window_df.empty:
            print(f"[{hour} SAAT ÖNCE] Veri bulunamadı.")
            continue
        # Seçilen dilimi ölçekle
        _, post_scaled_segment, _ = scale_data(pre_df, window_df, features)
        valid_rows = ~np.isnan(post_scaled_segment).any(axis=1)
        segment_scaled = post_scaled_segment[valid_rows]
        # MSET ile tahmin ve hata hesapla
        predictions = np.array([mset_predict(memory, obs) for obs in segment_scaled])
        rmse = np.sqrt(mean_squared_error(segment_scaled, predictions))
        mean_val = np.mean(np.abs(segment_scaled))
        accuracy = 100 * (1 - rmse / mean_val) if mean_val > 1e-3 else None
        print(f"\n[{hour} SAAT ÖNCE]")
        print(f"RMSE: {rmse:.4f}")
        print(f"Doğruluk: {accuracy:.2f}%" if accuracy is not None else "Doğruluk hesaplanamadı.")
        # İlk arıza tespit edilen zamanı kaydet
        if (accuracy is not None and accuracy < accuracy_threshold and first_fault_time is None):
            first_fault_time = hour
            first_fault_rmse = rmse
            first_fault_acc = accuracy
    return first_fault_time, first_fault_rmse, first_fault_acc

def full_analysis(pre_df: pd.DataFrame, post_df: pd.DataFrame, features: List[str], memory: np.ndarray) -> Tuple[float, Optional[float], np.ndarray]:
    """
    Tüm bakım sonrası veriler üzerinde genel analiz yapar.
    RMSE, doğruluk ve sensör bazlı hata değerlerini döndürür.
    """
    filtered_post_scaled = scale_data(pre_df, post_df, features)[1]
    valid_rows = ~np.isnan(filtered_post_scaled).any(axis=1)
    filtered_post_scaled = filtered_post_scaled[valid_rows]
    predictions = np.array([mset_predict(memory, obs) for obs in filtered_post_scaled])
    rmse = np.sqrt(mean_squared_error(filtered_post_scaled, predictions))
    mean_val = np.mean(np.abs(filtered_post_scaled))
    accuracy = 100 * (1 - rmse / mean_val) if mean_val > 1e-3 else None
    feature_rmse = np.sqrt(((filtered_post_scaled - predictions) ** 2).mean(axis=0))
    return rmse, accuracy, feature_rmse

def print_fault_summary(first_fault_time: Optional[int], first_fault_rmse: Optional[float], first_fault_acc: Optional[float]):
    """
    Erken arıza tespitini ekrana yazdırır.
    """
    if first_fault_time is not None:
        print(f"\n{'*'*40}")
        print(f"ERKEN ARIZA TESPİTİ")
        print(f"{'*'*40}")
        print(f"Arıza ilk olarak [{first_fault_time} SAAT ÖNCE] tespit edildi!")
        print(f"RMSE: {first_fault_rmse:.4f}")
        print(f"Doğruluk: {first_fault_acc:.2f}%\n")
        print(f"{'*'*40}\n")

def print_analysis_report(rmse: float, accuracy: Optional[float], feature_rmse: np.ndarray, features: List[str]):
    """
    Genel analiz sonuçlarını ekrana yazdırır.
    """
    print(f"\n{'='*40}\n[GENEL ANALİZ] Sistem: {MACHINE_NAME}\n{'='*40}")
    print(f"RMSE: {rmse:.4f}")
    if accuracy is not None:
        print(f"Doğruluk: {accuracy:.2f}%")
    else:
        print("Doğruluk hesaplanamadı.")

def print_warning_or_normal(accuracy: Optional[float]):
    """
    Sistem durumu hakkında uyarı veya normal çalışıyor mesajı verir.
    """
    if accuracy is not None and accuracy < ACCURACY_THRESHOLD:
        print(f"\n[UYARI] Anormal davranış tespit edildi. Bakım gerekebilir!\n")
    else:
        print("\nSistem normal çalışıyor.\n")

def print_feature_rmse(feature_rmse: np.ndarray, features: List[str]):
    """
    Sensör bazlı hata (RMSE) değerlerini ve en çok sapma gösteren sensörleri ekrana yazdırır.
    """
    sorted_features = sorted(zip(features, feature_rmse), key=lambda x: x[1], reverse=True)
    print(f"{'-'*40}\nEn Çok Sapma Gösteren İlk 3 Sensör\n{'-'*40}")
    for f, e in sorted_features[:3]:
        print(f"- {f:<35} : {e:>7.2f}")
    print(f"\n{'-'*40}\nTüm Sensör Farkları (RMSE)\n{'-'*40}")
    print(f"{'Sensör Adı':<40} | {'RMSE':>8}")
    print(f"{'-'*53}")
    for f, e in sorted_features:
        print(f"{f:<40} | {e:>8.4f}")

def main():
    print("=== GELİŞMİŞ MSET ARIZA TESPİT SİSTEMİ ===")
    print("Makine başlatma/kapatma verileri dahil edilerek analiz yapılıyor...\n")
    
    # Başlatma/kapatma verilerini oku
    startup_data, shutdown_data = read_startup_shutdown_data(STARTUP_SHUTDOWN_PATH)
    
    # Ana verileri oku ve ön işle
    pre_df, post_df, features_clean = read_and_preprocess_data(EXCEL_PATH, FEATURES)
    
    # Gelişmiş hafıza oluştur (normal + başlatma + kapatma verileri)
    enhanced_memory = create_enhanced_memory(pre_df, startup_data, shutdown_data, features_clean)
    
    print(f"Analiz için referans bakım tarihi: {post_df['Tarih'].max() - timedelta(minutes=1)}")
    
    # Zaman aralıklarında arıza tespiti yap
    first_fault_time, first_fault_rmse, first_fault_acc = analyze_time_windows(
        pre_df, post_df, features_clean, enhanced_memory, TIME_WINDOWS_HOURS, ACCURACY_THRESHOLD)
    
    # Genel analiz yap
    rmse, accuracy, feature_rmse = full_analysis(pre_df, post_df, features_clean, enhanced_memory)
    print_analysis_report(rmse, accuracy, feature_rmse, features_clean)
    print_fault_summary(first_fault_time, first_fault_rmse, first_fault_acc)
    print_warning_or_normal(accuracy)
    print_feature_rmse(feature_rmse, features_clean)

if __name__ == "__main__":
    main()