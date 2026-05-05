import os
import random
import torch
import shutil
from ultralytics import YOLO
import pandas as pd  # Eğer yüklü değilse: pip install pandas

# 1. YOLLAR VE AYARLAR
MODEL_PATH = 'runs/detect/Araba_Odevi/Guvenli_Egitim/weights/best.pt'
TRAIN_RESULTS_PATH = 'runs/detect/Araba_Odevi/Guvenli_Egitim'
TEST_DIR = 'archive/cars_test/cars_test'
FINAL_REPORT_DIR = 'PROJE_FINAL_RAPOR_VERILERI'

# Klasörü oluştur veya temizle
if os.path.exists(FINAL_REPORT_DIR):
    shutil.rmtree(FINAL_REPORT_DIR)
os.makedirs(FINAL_REPORT_DIR, exist_ok=True)

# 2. MODELİ YÜKLE
if not os.path.exists(MODEL_PATH):
    print(f"HATA: Model bulunamadı! {MODEL_PATH}")
    exit()

model = YOLO(MODEL_PATH)


def rapor_verilerini_hazirla():
    print("\n" + "=" * 60)
    print("Odev Gereksinimleri ve Teknik Analiz Baslatildi")
    print("=" * 60)

    # --- MADDESİ 1: CNN (Evrişimli Sinir Ağları) ---
    print("\n[MADDESİ: CNN] Model Mimarisi Özetleniyor...")
    with open(f"{FINAL_REPORT_DIR}/model_mimarisi.txt", "w") as f:
        f.write(str(model.info()))
    print(f"-> CNN katman bilgileri 'model_mimarisi.txt' olarak kaydedildi.")

    # --- MADDESİ 2: Feature Extraction (Öznitelik Çıkarımı) ---
    print("\n[MADDESİ: FEATURE EXTRACTION] Görüntü Analizi Yapılıyor...")
    test_images = [f for f in os.listdir(TEST_DIR) if f.endswith('.jpg')]
    sample_img = os.path.join(TEST_DIR, random.choice(test_images))

    # visualize=True ile katman katman özellik çıkarımını alıyoruz
    model.predict(source=sample_img, visualize=True, project=FINAL_REPORT_DIR, name='feature_maps', device='mps')
    print(f"-> Feature Extraction görselleri '{FINAL_REPORT_DIR}/feature_maps' içine kaydedildi.")

    # --- MADDESİ 3: Başarı Oranları ve Grafikler ---
    print("\n[MADDESİ: BASARI ANALIZI] Eğitim Grafikleri Taşınıyor...")
    # Eğitim sırasında oluşan grafiklerin en önemlilerini rapor klasörüne taşıyoruz
    files_to_copy = ['results.png', 'confusion_matrix.png', 'F1_curve.png', 'labels.jpg']
    for file in files_to_copy:
        src = os.path.join(TRAIN_RESULTS_PATH, file)
        if os.path.exists(src):
            shutil.copy(src, FINAL_REPORT_DIR)

    print(f"-> Basarı grafikleri (results.png vb.) '{FINAL_REPORT_DIR}' klasörüne toplandı.")

    # --- MADDESİ 4: Toplu Tahmin Testi ---
    print("\n[MADDESİ: TESPIT] Rastgele 10 Araç Üzerinde Demo Yapılıyor...")
    random_samples = random.sample(test_images, 10)
    model.predict(source=[os.path.join(TEST_DIR, img) for img in random_samples],
                  save=True, conf=0.15, project=FINAL_REPORT_DIR, name='tahmin_ornekleri', device='mps')

    print("\n" + "=" * 60)
    print(f"ISLEM TAMAM! Tüm veriler '{FINAL_REPORT_DIR}' klasöründe.")
    print("=" * 60)


if __name__ == "__main__":
    rapor_verilerini_hazirla()