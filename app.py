import os
from flask import Flask, render_template, request
from ultralytics import YOLO

app = Flask(__name__)

# --- 1. YOLLAR VE KURULUM ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static/uploads')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# --- 2. GLOBAL GEÇMİŞ LİSTESİ ---
# Son analiz edilen 10 aracı bellekte tutar
analysis_history = []

# --- 3. MODELLER ---
# V2: Ana dizine kopyaladığın güçlü model
MODEL_V2_PATH = os.path.join(BASE_DIR, 'best.pt')
# V1: Proje klasöründeki ham model
MODEL_V1_PATH = os.path.join(BASE_DIR, 'yolov8n.pt')

# Modelleri M4 Pro RAM'ine yükle
model_v1 = YOLO(MODEL_V1_PATH)
model_v2 = YOLO(MODEL_V2_PATH)


# --- 4. YARDIMCI FONKSİYONLAR ---
def manage_storage(max_files=10):
    """Klasördeki dosya sayısını 10 ile sınırlar, eskileri siler."""
    files = [os.path.join(UPLOAD_FOLDER, f) for f in os.listdir(UPLOAD_FOLDER) if f != '.DS_Store']
    files.sort(key=os.path.getctime)  # Eskiden yeniye sırala

    while len(files) > max_files:
        oldest_file = files.pop(0)
        try:
            os.remove(oldest_file)
        except Exception as e:
            print(f"Temizlik hatası: {e}")


# --- 5. ANA ROTA ---
@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    image_url = None
    selected_model = "v2"

    if request.method == 'POST':
        file = request.files.get('file')
        selected_model = request.form.get('model_choice')

        if file and file.filename != '':
            # Dosyayı kaydet
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)

            # Depolama temizliği yap (max 10)
            manage_storage(10)

            # Seçilen modele göre analiz yap (M4 Pro GPU - MPS aktif)
            current_model = model_v2 if selected_model == "v2" else model_v1
            results = current_model.predict(source=file_path, conf=0.25, device='mps')

            # Sonuçları işle
            if len(results[0].boxes) > 0:
                box = results[0].boxes[0]
                label = current_model.names[int(box.cls)]
                conf = float(box.conf)
                prediction = f"[{selected_model.upper()}] {label.upper()} (%{conf * 100:.1f})"
            else:
                prediction = f"[{selected_model.upper()}] Tespit edilemedi."

            image_url = f"static/uploads/{file.filename}"

            # Analiz geçmişini güncelle (En başa ekle, 10'da dur)
            analysis_history.insert(0, {'image': image_url, 'label': prediction})
            if len(analysis_history) > 10:
                analysis_history.pop()

    return render_template('index.html',
                           prediction=prediction,
                           image_url=image_url,
                           selected_model=selected_model,
                           history=analysis_history)


if __name__ == '__main__':
    # Flask sunucusunu başlat
    app.run(debug=True, port=5000)