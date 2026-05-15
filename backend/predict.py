# -*- coding: utf-8 -*-
"""
ESC-50 Ses Siniflandirici - Predict Script
Kullanim: python predict.py <ses_dosyasi_yolu>
Ornek:    python predict.py test.wav
"""
import os
import sys
import numpy as np

# Proje kök dizinini ayarla
base_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, base_path)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # TF uyarılarını azalt
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf

# Lambda katmanlarindaki bytecode 'tf' global degiskenine referans veriyor.
# Model Jupyter'da egitildigi icin 'tf' orada globaldi. Burada da
# builtins'e enjekte ederek ayni erisimi sagliyoruz.
import builtins
builtins.tf = tf

from utils.audio_utils import preprocess_audio_file, get_model_inputs


ESC50_CATEGORIES = [
    'dog', 'rooster', 'pig', 'cow', 'frog',
    'cat', 'hen', 'insects', 'sheep', 'crow',
    'rain', 'sea_waves', 'crackling_fire', 'crickets', 'chirping_birds',
    'water_drops', 'wind', 'pouring_water', 'toilet_flush', 'thunderstorm',
    'crying_baby', 'sneezing', 'clapping', 'breathing', 'coughing',
    'footsteps', 'laughing', 'brushing_teeth', 'snoring', 'drinking_sipping',
    'door_knock', 'mouse_click', 'keyboard_typing', 'door_slam', 'ticking_clock',
    'can_opening', 'washing_machine', 'vacuum_cleaner', 'clock_alarm', 'church_bells',
    'airplane', 'helicopter', 'fireworks', 'hand_saw', 'lawn_mower',
    'car_horn', 'engine', 'train', 'chainsaw', 'siren'
]


def load_models():
    """YAMNet ve ESC-50 modellerini yükler."""
    yamnet_path = os.path.join(base_path, "models", "yamnet_model")
    model_path = os.path.join(base_path, "models", "model.keras")

    print("[*] YAMNet yukleniyor...")
    yamnet = tf.saved_model.load(yamnet_path)
    print("[OK] YAMNet yuklendi.")

    print("[*] ESC-50 modeli yukleniyor...")
    esc50_model = tf.keras.models.load_model(
        model_path,
        safe_mode=False,
        custom_objects={'Lambda': tf.keras.layers.Lambda}
    )
    print("[OK] ESC-50 modeli yuklendi.")

    return yamnet, esc50_model


def predict(audio_path, yamnet, esc50_model):
    """Ses dosyasini siniflandirir."""
    # 1. Ses dosyasini yukle ve on isle
    waveform = preprocess_audio_file(audio_path)
    print(f"[INFO] Ses uzunlugu: {len(waveform)} ornek ({len(waveform)/16000:.2f} saniye)")

    # 2. YAMNet ile özellik çıkar
    spec_input, emb_input = get_model_inputs(waveform, yamnet)
    print(f"[INFO] Spectrogram shape: {spec_input.shape}, Embedding shape: {emb_input.shape}")

    # 3. ESC-50 modeli ile tahmin
    preds = esc50_model.predict([spec_input, emb_input], verbose=0)
    idx = np.argmax(preds[0])
    confidence = float(np.max(preds[0]))

    return ESC50_CATEGORIES[idx], confidence, preds[0]


def main():
    if len(sys.argv) < 2:
        print("Kullanım: python predict.py <ses_dosyası_yolu>")
        print("Örnek:    python predict.py test.wav")
        sys.exit(1)

    audio_path = sys.argv[1]

    if not os.path.exists(audio_path):
        print(f"[HATA] Dosya bulunamadi: {audio_path}")
        sys.exit(1)

    # Modelleri yukle
    yamnet, esc50_model = load_models()

    # Tahmin yap
    print(f"\n[*] Dosya analiz ediliyor: {audio_path}")
    predicted_class, confidence, all_probs = predict(audio_path, yamnet, esc50_model)

    # Sonuclari goster
    print("\n" + "=" * 50)
    print(f">> Tahmin: {predicted_class}")
    print(f">> Guven:  {confidence * 100:.2f}%")
    print("=" * 50)

    # Top 5 tahmin
    top5_idx = np.argsort(all_probs)[::-1][:5]
    print("\nEn yuksek 5 tahmin:")
    for i, idx in enumerate(top5_idx, 1):
        print(f"   {i}. {ESC50_CATEGORIES[idx]:20s} -> %{all_probs[idx] * 100:.2f}")


if __name__ == "__main__":
    main()
