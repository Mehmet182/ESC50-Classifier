import os
import sys
from contextlib import asynccontextmanager

# Import sorununu kökten çözmek için ana dizini sisteme tanıtıyoruz
base_path = os.path.dirname(os.path.abspath(__file__))
if base_path not in sys.path:
    sys.path.insert(0, base_path)

from fastapi import FastAPI, UploadFile, File, HTTPException
import tensorflow as tf
import uvicorn
import numpy as np

# Lambda katmanlarindaki bytecode 'tf' global degiskenine referans veriyor.
import builtins
builtins.tf = tf


# Lambda katmanı hatasını çözmek için gerekli tanım
def lambda_output_fix(input_shape):
    return (input_shape[0], 1)


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


# Modülleri manuel olarak bağlamayı deneyelim (Hata alırsak sistem çökmesin diye)
try:
    import utils.audio_utils as audio_utils

    print("✅ Ses işleme modülleri (utils) başarıyla bağlandı.")
except Exception as e:
    print(f"❌ Kritik Hata: utils bulunamadı. Lütfen klasör yapısını kontrol et. Detay: {e}")

models = {}
ESC50_CATEGORIES = ['dog', 'rooster', 'pig', 'cow', 'frog', 'cat', 'hen', 'insects', 'sheep', 'crow', 'rain',
                    'sea_waves', 'crackling_fire', 'crickets', 'chirping_birds', 'water_drops', 'wind', 'pouring_water',
                    'toilet_flush', 'thunderstorm', 'crying_baby', 'sneezing', 'clapping', 'breathing', 'coughing',
                    'footsteps', 'laughing', 'brushing_teeth', 'snoring', 'drinking_sipping', 'door_knock',
                    'mouse_click', 'keyboard_typing', 'door_slam', 'ticking_clock', 'can_opening', 'washing_machine',
                    'vacuum_cleaner', 'clock_alarm', 'church_bells', 'airplane', 'helicopter', 'fireworks', 'hand_saw',
                    'lawn_mower', 'car_horn', 'engine', 'train', 'chainsaw', 'siren']

app = FastAPI(title="ESC-50 Classifier", lifespan=lifespan)


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        content = await file.read()
        # utils üzerinden çağırıyoruz
        waveform = audio_utils.preprocess_audio_bytes(content)
        spec_in, emb_in = audio_utils.get_model_inputs(waveform, models["yamnet"])

        preds = models["esc50"].predict([spec_in, emb_in])
        idx = np.argmax(preds[0])
        return {"tahmin": ESC50_CATEGORIES[idx], "oran": float(np.max(preds[0]))}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)