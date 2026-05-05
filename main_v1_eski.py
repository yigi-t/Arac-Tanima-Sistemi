import os
import scipy.io as sio
import cv2
import torch
import shutil
from ultralytics import YOLO
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# --- 1. YOLLAR VE AYARLAR ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARCHIVE_DIR = os.path.join(BASE_DIR, 'archive')
IMG_SOURCE_DIR = os.path.join(ARCHIVE_DIR, 'cars_train/cars_train')
MAT_ANNOS = os.path.join(ARCHIVE_DIR, 'car_devkit/devkit/cars_train_annos.mat')
MAT_META = os.path.join(ARCHIVE_DIR, 'car_devkit/devkit/cars_meta.mat')
DATASET_ROOT = os.path.join(BASE_DIR, 'datasets')


def setup_directories():
    """Gereksiz dosyaları temizler ve yeni yapıyı kurar."""
    if os.path.exists(DATASET_ROOT):
        shutil.rmtree(DATASET_ROOT)  # Eski hatalı klasörleri siler
    for folder in ['images/train', 'images/val', 'labels/train', 'labels/val']:
        os.makedirs(os.path.join(DATASET_ROOT, folder), exist_ok=True)


def prepare_data():
    """Verileri YOLO formatına çevirir (RAM dostu kopyalama)."""
    print("--- ADIM 1: Veriler Hazırlanıyor (Bilgisayarı yormadan) ---")
    annos = sio.loadmat(MAT_ANNOS)['annotations'][0]
    meta = sio.loadmat(MAT_META)['class_names'][0]
    class_names = [c[0] for c in meta]

    # %90 Train, %10 Val
    train_data, val_data = train_test_split(annos, test_size=0.1, random_state=42)

    def process(data_list, split_type):
        for anno in tqdm(data_list, desc=f"{split_type} kopyalanıyor"):
            img_name = anno['fname'][0]
            src_path = os.path.join(IMG_SOURCE_DIR, img_name)
            if not os.path.exists(src_path): continue

            # Resim boyutunu oku (Sadece header okur, RAM harcamaz)
            img = cv2.imread(src_path)
            h, w = img.shape[:2]

            # Etiket hesaplama
            x_center = ((float(anno['bbox_x1'][0][0]) + float(anno['bbox_x2'][0][0])) / 2) / w
            y_center = ((float(anno['bbox_y1'][0][0]) + float(anno['bbox_y2'][0][0])) / 2) / h
            width = (float(anno['bbox_x2'][0][0]) - float(anno['bbox_x1'][0][0])) / w
            height = (float(anno['bbox_y2'][0][0]) - float(anno['bbox_y1'][0][0])) / h

            # Dosyaları yerleştir
            shutil.copy(src_path, os.path.join(DATASET_ROOT, f'images/{split_type}', img_name))
            with open(os.path.join(DATASET_ROOT, f'labels/{split_type}', img_name.replace('.jpg', '.txt')), 'w') as f:
                f.write(f"{int(anno['class'][0][0]) - 1} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

    process(train_data, 'train')
    process(val_data, 'val')

    # YAML dosyasını oluştur
    with open('dataset.yaml', 'w') as f:
        f.write(
            f"path: {DATASET_ROOT}\ntrain: images/train\nval: images/val\nnc: {len(class_names)}\nnames: {class_names}")


def train_light():
    """M4 Pro'yu yormadan en hafif modelle eğitim."""
    print("--- ADIM 2: Güvenli Eğitim Başlıyor ---")

    # Apple Silicon GPU (MPS) Kullanımı
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'

    # En hafif model (Nano)
    model = YOLO('yolov8n.pt')

    model.train(
        data='dataset.yaml',
        epochs=20,  # 20 epoch ödev sunumu için yeterlidir
        imgsz=416,  # Düşük çözünürlük = Çok az RAM kullanımı
        batch=4,  # Aynı anda sadece 4 resim (RAM'i korur)
        workers=2,  # macOS için ideal işlem sayısı
        cache=False,  # RAM yerine diskten oku (Swap'ı engeller)
        device=device,
        project='Araba_Odevi',
        name='Guvenli_Egitim'
    )


if __name__ == "__main__":
    setup_directories()
    prepare_data()
    train_light()