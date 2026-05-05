import os
import scipy.io as sio
import cv2
import torch
import shutil
import random
from ultralytics import YOLO
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# --- 1. YOLLAR VE AYARLAR ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARCHIVE_DIR = os.path.join(BASE_DIR, 'archive')
IMG_SOURCE_DIR = os.path.join(ARCHIVE_DIR, 'cars_train/cars_train')
MAT_ANNOS = os.path.join(ARCHIVE_DIR, 'car_devkit/devkit/cars_train_annos.mat')
MAT_META = os.path.join(ARCHIVE_DIR, 'car_devkit/devkit/cars_meta.mat')

DATASET_ROOT_YENI = os.path.join(BASE_DIR, 'datasets_yeni')
YAML_PATH_YENI = 'dataset_yeni.yaml'
OUTPUT_REPORT_DIR = 'GELISMIS_RAPOR_GÖRSELLERI'

# YOLO'nun oluşturduğu gerçek (iç içe geçmiş) yol
BEST_MODEL_PATH = os.path.join(BASE_DIR, 'runs/detect/runs/detect/Araba_Odevi/Gelistirilmis_Model_100e/weights/best.pt')

def setup_directories_yeni():
    """Yeni klasör yapısını kurar (Eğer varsa silmez, üzerine devam eder)."""
    if not os.path.exists(DATASET_ROOT_YENI):
        for folder in ['images/train', 'images/val', 'labels/train', 'labels/val']:
            os.makedirs(os.path.join(DATASET_ROOT_YENI, folder), exist_ok=True)
    if not os.path.exists(OUTPUT_REPORT_DIR):
        os.makedirs(OUTPUT_REPORT_DIR, exist_ok=True)

def prepare_data_yeni():
    """Verileri YOLO formatına çevirir (Eğer zaten hazırsa atlar)."""
    if os.path.exists(YAML_PATH_YENI):
        print("--- [BİLGİ] Veri seti zaten hazır, hazırlama adımı atlanıyor. ---")
        return

    print("--- ADIM 1: Yeni Veri Seti Hazırlanıyor ---")
    annos = sio.loadmat(MAT_ANNOS)['annotations'][0]
    meta = sio.loadmat(MAT_META)['class_names'][0]
    class_names = [c[0] for c in meta]

    train_data, val_data = train_test_split(annos, test_size=0.1, random_state=42)

    def process(data_list, split_type):
        for anno in tqdm(data_list, desc=f"{split_type} kopyalanıyor"):
            img_name = anno['fname'][0]
            src_path = os.path.join(IMG_SOURCE_DIR, img_name)
            if not os.path.exists(src_path): continue

            img = cv2.imread(src_path)
            if img is None: continue
            h, w = img.shape[:2]

            x_center = ((float(anno['bbox_x1'][0][0]) + float(anno['bbox_x2'][0][0])) / 2) / w
            y_center = ((float(anno['bbox_y1'][0][0]) + float(anno['bbox_y2'][0][0])) / 2) / h
            width = (float(anno['bbox_x2'][0][0]) - float(anno['bbox_x1'][0][0])) / w
            height = (float(anno['bbox_y2'][0][0]) - float(anno['bbox_y1'][0][0])) / h

            shutil.copy(src_path, os.path.join(DATASET_ROOT_YENI, f'images/{split_type}', img_name))
            with open(os.path.join(DATASET_ROOT_YENI, f'labels/{split_type}', img_name.replace('.jpg', '.txt')), 'w') as f:
                f.write(f"{int(anno['class'][0][0]) - 1} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

    process(train_data, 'train')
    process(val_data, 'val')

    with open(YAML_PATH_YENI, 'w') as f:
        f.write(f"path: {DATASET_ROOT_YENI}\ntrain: images/train\nval: images/val\nnc: {len(class_names)}\nnames: {class_names}")

def run_comparative_analysis():
    """Analizi tamamlar."""
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'

    # Örnek resim seçimi
    test_images = [f for f in os.listdir(os.path.join(DATASET_ROOT_YENI, 'images/val')) if f.endswith('.jpg')]
    sample_img_path = os.path.join(DATASET_ROOT_YENI, 'images/val', test_images[0])

    # 1. EĞİTİM ÖNCESİ (Sadece klasör yoksa çalışır)
    if not os.path.exists(os.path.join(OUTPUT_REPORT_DIR, 'Egitim_Oncesi_Fark')):
        print("\n--- ANALİZ: Eğitim Öncesi Öznitelikler Çıkarılıyor... ---")
        model_raw = YOLO('yolov8n.pt')
        model_raw.predict(source=sample_img_path, visualize=True, project=OUTPUT_REPORT_DIR, name='Egitim_Oncesi_Fark')

    # 2. EĞİTİM KONTROLÜ
    if not os.path.exists(BEST_MODEL_PATH):
        print("\n--- UYARI: Model dosyası bulunamadı, eğitim tekrar başlatılıyor! ---")
        model = YOLO('yolov8n.pt')
        model.train(
            data=YAML_PATH_YENI,
            epochs=100,
            imgsz=416,
            batch=16,
            device=device,
            project='runs/detect/Araba_Odevi',
            name='Gelistirilmis_Model_100e'
        )
    else:
        print("\n--- [BAŞARI] Eğitilmiş model bulundu, analiz adımına geçiliyor. ---")

    # 3. EĞİTİM SONRASI ANALİZ
    print("\n--- ANALİZ: Eğitim Sonrası Öznitelikler Çıkarılıyor... ---")
    model_trained = YOLO(BEST_MODEL_PATH)
    model_trained.predict(source=sample_img_path, visualize=True, project=OUTPUT_REPORT_DIR, name='Egitim_Sonrasi_Fark')

    print(f"\nİŞLEM TAMAM! Karşılaştırmalı görseller '{OUTPUT_REPORT_DIR}' klasöründe.")

if __name__ == "__main__":
    setup_directories_yeni()
    prepare_data_yeni()
    run_comparative_analysis()