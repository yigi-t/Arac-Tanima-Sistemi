# 🚗 Araç Tanıma ve Sınıflandırma Sistemi | Vehicle Recognition & Classification System

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![YOLOv8](https://img.shields.io/badge/Model-YOLOv8n-green)
![Framework](https://img.shields.io/badge/Framework-Flask-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

Bu proje, **YOLOv8** mimarisi kullanılarak geliştirilmiş, web tabanlı bir araç tanıma ve sınıflandırma sistemidir[cite: 1, 29]. [cite_start]Stanford Cars veri seti üzerinde eğitilen model, araçları sadece nesne olarak tespit etmekle kalmaz; marka, model ve üretim yılına kadar **ince taneli (fine-grained)** sınıflandırma yapabilmektedir.

---

## 🎯 Projenin Amacı ve Kapsamı

Projenin temel amacı, araçların görsel verilerinden faydalanarak marka ve model kategorilerini otomatik olarak tahmin eden yüksek performanslı bir sistem geliştirmektir. Bilgisayarlı görü (computer vision) alanının en zorlu problemlerinden biri olan **İnce Taneli Nesne Tanıma**, birbirine görsel olarak çok benzeyen sınıflar arasındaki mikroskobik farkların ayırt edilmesini hedefler.

## 📊 Veri Seti Özellikleri (Stanford Cars Dataset)

Çalışmada, bilgisayarlı görü alanında popüler bir referans olan **Stanford Cars** veri seti kullanılmıştır:
* **Sınıf Çeşitliliği:** Toplamda 196 farklı araç marka ve modeli sisteme dahil edilmiştir.
* **Veri Hacmi:** Veri seti, toplam 16.185 adet yüksek kaliteli araç görüntüsünden oluşmaktadır.
* **Akademik Temel:** Veri seti hazırlığında Krause ve ekibinin (2013) 3D nesne temsili çalışmaları baz alınmıştır.

Veri Seti: https://www.kaggle.com/datasets/eduardo4jesus/stanford-cars-dataset

---

## 🛠️ Kullanılan Teknolojiler ve Kütüphaneler

Yüksek performanslı nesne tespiti ve sınıflandırma hedeflerine ulaşmak için optimize edilmiş kütüphaneler kullanılmıştır:

| Kütüphane / Framework | Kullanım Amacı |
| :--- | :--- |
| **Ultralytics YOLO** | Nesne tespiti ve model yönetimi. |
| **PyTorch** | Modelin derin öğrenme altyapısı ve tensör işlemleri. |
| **Flask** | Web sunucusu, API yönetimi ve arayüz entegrasyonu. |
| **OpenCV** | Görüntü işleme operasyonları ve görselleştirme. |
| **NumPy & Pandas** | Veri manipülasyonu ve matris işlemleri. |
| **Matplotlib** | Performans grafiklerinin ve eğitim verilerinin görselleştirilmesi. |

---

## 🔬 Teknik Metodoloji ve Analiz

### 1. Hiyerarşik Öznitelik Çıkarımı (Feature Extraction)
Proje kapsamında modelin karar verme süreci `visualize=True` metodolojisi ile şeffaflaştırılmıştır:
* **Kenar ve Doku Tespiti (Stage 0-1):** Modelin ilk katmanlarında görsel iskelet, kenarlar ve kontrast farkları saptanır.
* **Semantik Derinlik (Stage 21):** Derin katmanlarda araçların marka ve modeline özgü karakteristik tasarım dillerine odaklanılır.

### 2. Karşılaştırmalı Model Analizi (Ablasyon Çalışması)
Eğitimin etkisini gözlemlemek adına iki farklı model yapısı kurgulanmıştır:
* **V1 Modeli:** 20 epoch'luk başlangıç eğitimi; genel kategorizasyonda başarılı ancak detaylarda zayıf.
* **V2 Modeli:** 100 epoch'luk (96'da durdurulmuş) optimize eğitim; ince taneli detaylarda yüksek başarı.

---

## 📈 Performans Sonuçları ve Başarı Metrikleri

Modelimiz, derinlemesine yapılan testler sonucunda şu değerlere ulaşmıştır:
* **mAP50 Skoru (%94.9):** Nesne bulma ve genel sınıflandırmadaki yüksek doğruluğu temsil eder.
* **mAP50-95 Skoru (%90.1):** Sınırlayıcı kutuların araç üzerine milimetrik hassasiyetle oturduğunu tesciller.
* **F1-Confidence Dengesi:** Model, **0.547** güven eşiğinde **0.86** F1 skoru ile en kararlı çalışma noktasındadır.

---

## ⚙️ Kurulum ve Çalıştırma Talimatları (Terminal)

Sistemi yerel makinenizde ayağa kaldırmak için aşağıdaki adımları takip ediniz:

### 1. Depoyu Klonlayın
```bash
git clone [https://github.com/yigi-t/Arac-Tanima-Sistemi.git](https://github.com/yigi-t/Arac-Tanima-Sistemi.git)
cd Arac-Tanima-Sistemi
```


### 2. Sanal Ortam Oluşturun (Önerilir)
```bash
# macOS/Linux
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### 3. Bağımlılıkları Yükleyin
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Uygulamayı Başlatın
```bash
python app.py
```
Uygulama hazır olduğunda tarayıcınızdan http://127.0.0.1:5000 adresine erişebilirsiniz.

---

## 📂 Proje Dosya Yapısı
```bash
Arac-Tanima-Sistemi/
├── 📂 static/               # Arayüz ve görsel analiz dosyaları 
│   ├── 📂 css/              # Stil dosyaları (style.css)
│   ├── 📂 js/               # Dinamik etkileşimler için JavaScript
│   ├── 📂 uploads/          # Kullanıcı tarafından yüklenen test resimleri
│   └── 📂 results/          # Aktivasyon haritası (Feature Extraction) çıktıları [cite: 59]
├── 📂 templates/            # Flask HTML şablonları 
│   ├── 📄 index.html        # Ana yükleme ve seçim ekranı [cite: 385]
│   └── 📄 result.html       # Analiz ve tahmin sonuç ekranı [cite: 399]
├── 📂 runs/                 # YOLOv8 eğitim ve test çıktıları [cite: 35]
│   └── 📂 detect/           
│       └── 📂 train/        # Model ağırlıkları (best.pt) ve performans grafikleri [cite: 64, 382]
├── 📄 app.py                # Sunucu yönetimi ve Flask API ana giriş dosyası 
├── 📄 train.py              # Model eğitimi ve hiperparametre optimizasyonu [cite: 61]
├── 📄 requirements.txt      # Gerekli tüm kütüphane ve bağımlılık listesi
└── 📄 README.md             # Proje detaylı dökümantasyonu
```

---

## 👨‍💻 Hazırlayanlar

* Yiğit Taş
* Nisa Örnek
