import io
import numpy as np
import librosa
import soundfile as sf


def preprocess_audio_bytes(audio_bytes, target_sr=16000):
    """
    API'ye gelen ham ses verisini YAMNet standartlarına (16kHz, Mono) getirir.
    """
    # 1. Bayt verisini ses sinyaline dönüştür
    audio_data, sr = sf.read(io.BytesIO(audio_bytes))

    # 2. Eğer stereo (2 kanal) ise mono'ya çevir
    if len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data, axis=1)

    # 3. Örnekleme hızını 16kHz'e çek (Resampling)
    if sr != target_sr:
        audio_data = librosa.resample(y=audio_data, orig_sr=sr, target_sr=target_sr)

    # 4. Veri tipini float32 yap (YAMNet zorunluluğu)
    return audio_data.astype(np.float32)


def preprocess_audio_file(file_path, target_sr=16000):
    """
    Dosya yolundan ses yükleyip YAMNet standartlarına (16kHz, Mono) getirir.
    """
    audio_data, sr = librosa.load(file_path, sr=target_sr, mono=True)
    return audio_data.astype(np.float32)


def get_model_inputs(waveform, yamnet_model, target_spec_frames=528, target_emb_frames=10):
    """
    Ses sinyalini YAMNet'e sokar ve modelin hibrit kollarını besleyecek
    Spectrogram ve Embedding verilerini hazırlar.
    
    Boyutları modelin beklediği sabit boyutlara pad/crop ile ayarlar:
    - Spectrogram: (1, 528, 64, 1)
    - Embedding:   (1, 10, 1024)
    """
    # YAMNet tahmini
    scores, embeddings, spectrogram = yamnet_model(waveform)

    # Numpy'a çevir
    spectrogram = spectrogram.numpy()  # (frames, 64)
    embeddings = embeddings.numpy()    # (frames, 1024)

    # --- Spectrogram boyut ayarlama (target_spec_frames, 64) ---
    spec_frames = spectrogram.shape[0]
    if spec_frames < target_spec_frames:
        # Eksik kısımları sıfırla doldur (padding)
        pad_width = target_spec_frames - spec_frames
        spectrogram = np.pad(spectrogram, ((0, pad_width), (0, 0)), mode='constant')
    elif spec_frames > target_spec_frames:
        # Fazla kısımları kes (crop)
        spectrogram = spectrogram[:target_spec_frames, :]

    # --- Embedding boyut ayarlama (target_emb_frames, 1024) ---
    emb_frames = embeddings.shape[0]
    if emb_frames < target_emb_frames:
        pad_width = target_emb_frames - emb_frames
        embeddings = np.pad(embeddings, ((0, pad_width), (0, 0)), mode='constant')
    elif emb_frames > target_emb_frames:
        embeddings = embeddings[:target_emb_frames, :]

    # Model giriş formatına çevir
    # CNN kolu: (1, 528, 64, 1)
    spec_input = np.expand_dims(spectrogram, axis=0)
    spec_input = np.expand_dims(spec_input, axis=-1)

    # LSTM kolu: (1, 10, 1024)
    emb_input = np.expand_dims(embeddings, axis=0)

    return spec_input, emb_input