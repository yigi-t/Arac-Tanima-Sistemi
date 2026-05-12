# 🚗 Araç Tanıma ve Sınıflandırma Sistemi | Vehicle Recognition & Classification System

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![YOLOv8](https://img.shields.io/badge/Model-YOLOv8n-green)
![Framework](https://img.shields.io/badge/Framework-Flask-orange)

[cite_start]Bu proje, derin öğrenme yöntemlerini kullanarak araçların görsel verilerinden marka ve model kategorilerini otomatik olarak tahmin eden yüksek performanslı bir sistemdir[cite: 6]. [cite_start]Proje, sadece nesne tespiti yapmakla kalmayıp, birbirine görsel olarak çok benzeyen sınıflar arasındaki küçük farkların ayırt edilmesini hedefleyen **İnce Taneli Nesne Tanıma (Fine-grained Object Recognition)** problemine odaklanmaktadır[cite: 19, 20].

---

## 🎯 Projenin Amacı

[cite_start]Projenin temel amacı, bilgisayarlı görü tekniklerini kullanarak yüksek doğruluk oranına sahip bir araç sınıflandırma altyapısı kurmaktır[cite: 6]. [cite_start]Sistem, karmaşık görsel verileri analiz ederek araçların karakteristik tasarım detaylarını (ızgara yapısı, far geometrisi vb.) birer "dijital parmak izi" olarak işler[cite: 46, 49].

## 📊 Veri Seti (Stanford Cars Dataset)

[cite_start]Çalışmada, akademik alanda popülerliği kabul görmüş **Stanford Cars** veri seti kullanılmıştır[cite: 14, 15].
* [cite_start]**Kapsam:** Toplamda 196 farklı araç marka ve modeli sisteme dahil edilmiştir[cite: 16].
* [cite_start]**Veri Miktarı:** Veri seti, 16.185 adet yüksek kaliteli araç görüntüsünden oluşmaktadır[cite: 17, 18].
* **Akademik Referans:** Krause, J., Stark, M., Deng, J., & Fei-Fei, L. (2013) çalışması temel alınmıştır.

---

## 🔬 Teknik Metodoloji ve Mimari

[cite_start]Sistem, nesne tespiti dünyasının güncel ve optimize edilmiş algoritmaları üzerine inşa edilmiştir[cite: 22, 26].

### 1. Model Mimarisi
* [cite_start]**YOLOv8 Nano (YOLOv8n):** Hafif ve hızlı bir mimari tercih edilerek uç birimlerde (Edge AI) çalışma potansiyeli korunmuştur[cite: 29].
* [cite_start]**Derin Evrişimli Sinir Ağları (CNN):** Görsel öznitelikleri hiyerarşik bir yapıda çıkarmak için kullanılmıştır[cite: 30, 31].
* [cite_start]**Anchor-Free (Çapasız) Tespit:** Nesne sınırlarını daha esnek ve hassas belirlemek için uygulanmıştır[cite: 31].

### 2. Öznitelik Çıkarımı (Feature Extraction) Analizi
[cite_start]Projenin en güçlü yönlerinden biri, modelin kararlarını şeffaflaştıran `visualize=True` metodolojisidir[cite: 37].
* [cite_start]**Stage 0 & 1:** Modelin en sığ katmanlarında kenar ve doku tespiti (Edge & Texture Detection) gerçekleştirilir[cite: 51, 52, 53].
* [cite_start]**Stage 21 (C2f):** Derin katmanlarda semantik derinlik ve tasarım dili analizi yapılarak marka/model özgü detaylara odaklanılır[cite: 55, 57].

---

## 📈 Performans Sonuçları ve Başarı Metrikleri

[cite_start]Modelin başarısı, hem eğitim (train) hem de doğrulama (validation) süreçlerinde titizlikle ölçülmüştür[cite: 64].

* [cite_start]**mAP50 Skoru (%94.9):** Nesne bulma ve genel kategorizasyondaki yüksek başarıyı temsil eder[cite: 97, 118].
* [cite_start]**mAP50-95 Skoru (%90.1):** Sınırlayıcı kutuların araç üzerine milimetrik hassasiyetle oturduğunu kanıtlayan en zorlu başarı kriteridir[cite: 119, 136, 173].
* [cite_start]**F1-Confidence Dengesi:** Model, 0.547 güven eşiğinde (confidence) 0.86 F1 skoru ile en kararlı çalışma noktasına ulaşmıştır[cite: 221, 225].
* [cite_start]**Eğitim Stratejisi:** Model 100 epoch hedeflenmiş, ancak overfitting riskine karşı 96. epoch'ta "erken durdurma" (early stopping) ile en iyi ağırlıklar (best.pt) kaydedilmiştir[cite: 229, 382].

---

## 💻 Uygulama Arayüzü ve Kullanım

[cite_start]Proje, Flask tabanlı interaktif bir web arayüzü üzerinden sunulmaktadır[cite: 38, 385, 395].
* [cite_start]**V1 (20 Epoch):** Başlangıç aşamasındaki zayıf model analizi[cite: 227, 388, 390].
* [cite_start]**V2 (100 Epoch):** Derinleştirilmiş eğitimle optimize edilen güçlü model analizi[cite: 229, 392, 393].
* [cite_start]**Kullanım:** Kullanıcı dostu arayüz üzerinden sürükle-bırak yöntemiyle resim yüklenip "Tahmin Et" butonuyla anlık analiz sonuçlarına ulaşılabilir[cite: 396, 399].

---

## 🛠️ Kullanılan Teknolojiler

| Kütüphane / Framework | Kullanım Amacı |
| :--- | :--- |
| **Ultralytics YOLO** | [cite_start]Nesne tespiti ve model yönetimi [cite: 35] |
| **PyTorch** | [cite_start]Derin öğrenme altyapısı [cite: 36] |
| **Flask** | [cite_start]Web sunucusu ve API yönetimi [cite: 38] |
| **OpenCV** | [cite_start]Görüntü işleme operasyonları [cite: 39] |
| **NumPy & Pandas** | [cite_start]Veri manipülasyonu ve matris işlemleri [cite: 40] |
| **Matplotlib** | [cite_start]Performans grafiklerinin görselleştirilmesi [cite: 41] |

---

## 🏢 Endüstriyel Uygulama Alanları

Geliştirilen sistem şu alanlarda doğrudan entegrasyona uygundur:
* [cite_start]**Akıllı Ulaşım:** Trafik güvenliği ve plaka/marka doğrulama sistemleri[cite: 11].
* [cite_start]**Otopark Otomasyonu:** İki faktörlü doğrulama süreçleri[cite: 12].
* [cite_start]**Lojistik:** Otomotiv üretim bantlarında kalite kontrol ve envanter yönetimi[cite: 13].

---
