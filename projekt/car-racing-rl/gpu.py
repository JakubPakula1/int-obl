import tensorflow as tf
print("GPU dostępne: ", tf.config.list_physical_devices('GPU'))
print("TensorFlow używa GPU: ", tf.test.is_gpu_available())
print("TensorFlow wersja: ", tf.__version__)