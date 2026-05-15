import tensorflow_hub as hub
import tensorflow as tf

# Modeli indir
model = hub.load('https://tfhub.dev/google/yamnet/1')
# 'yamnet_model' adında bir klasöre kalıcı olarak kaydet
tf.saved_model.save(model, 'models/yamnet_model')
print("YAMNet yerel klasöre kaydedildi!")