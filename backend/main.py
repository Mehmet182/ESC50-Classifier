import os
import sys
from contextlib import asynccontextmanager

# Proje kök dizinini ayarla
base_path = os.path.dirname(os.path.abspath(__file__))
if base_path not in sys.path:
    sys.path.insert(0, base_path)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # TF uyarılarını azalt
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import tensorflow as tf
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

# Lambda katmanlarindaki bytecode 'tf' global degiskenine referans veriyor.
import builtins
builtins.tf = tf

from utils.audio_utils import preprocess_audio_bytes, get_model_inputs


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        yamnet_path = os.path.join(base_path, "models", "yamnet_model")
        model_path = os.path.join(base_path, "models", "model.keras")

        print("⏳ Modeller yükleniyor...")
        models["yamnet"] = tf.saved_model.load(yamnet_path)

        # Lambda hatasını Custom Objects ile aşıyoruz
        models["esc50"] = tf.keras.models.load_model(
            model_path,
            safe_mode=False,
            custom_objects={'Lambda': tf.keras.layers.Lambda}
        )
        print("✅ Tüm modeller yerelden başarıyla yüklendi!")
    except Exception as e:
        print(f"❌ Modeller yüklenirken hata oluştu: {e}")
    yield
    models.clear()

models = {}
ESC50_CATEGORIES = ['dog', 'rooster', 'pig', 'cow', 'frog', 'cat', 'hen', 'insects', 'sheep', 'crow', 'rain',
                    'sea_waves', 'crackling_fire', 'crickets', 'chirping_birds', 'water_drops', 'wind', 'pouring_water',
                    'toilet_flush', 'thunderstorm', 'crying_baby', 'sneezing', 'clapping', 'breathing', 'coughing',
                    'footsteps', 'laughing', 'brushing_teeth', 'snoring', 'drinking_sipping', 'door_knock',
                    'mouse_click', 'keyboard_typing', 'door_slam', 'ticking_clock', 'can_opening', 'washing_machine',
                    'vacuum_cleaner', 'clock_alarm', 'church_bells', 'airplane', 'helicopter', 'fireworks', 'hand_saw',
                    'lawn_mower', 'car_horn', 'engine', 'train', 'chainsaw', 'siren']

app = FastAPI(title="ESC-50 Classifier", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Frontend static dosyalarını sun
frontend_path = os.path.join(os.path.dirname(base_path), "frontend")
if os.path.exists(frontend_path):
    app.mount("/static", StaticFiles(directory=frontend_path), name="static")


@app.get("/")
async def serve_frontend():
    index_path = os.path.join(frontend_path, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "ESC-50 Classifier API"}


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        content = await file.read()
        waveform = preprocess_audio_bytes(content)
        spec_in, emb_in = get_model_inputs(waveform, models["yamnet"])

        preds = models["esc50"].predict([spec_in, emb_in], verbose=0)
        probs = preds[0].tolist()
        idx = int(np.argmax(preds[0]))
        confidence = float(np.max(preds[0]))

        # Top 5 tahmin
        top5_idx = np.argsort(preds[0])[::-1][:5].tolist()
        top5 = [
            {"label": ESC50_CATEGORIES[i], "confidence": float(preds[0][i])}
            for i in top5_idx
        ]

        return {
            "prediction": ESC50_CATEGORIES[idx],
            "confidence": confidence,
            "top5": top5,
            "all_probabilities": {ESC50_CATEGORIES[i]: float(p) for i, p in enumerate(probs)}
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)