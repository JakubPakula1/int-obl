import tensorflow as tf
print("TensorFlow version:", tf.__version__)
print("Dostępne urządzenia:", tf.config.list_physical_devices())

# Test MPS
if len(tf.config.list_physical_devices('MPS')) > 0:
    print("MPS dostępne!")
    with tf.device('/device:MPS:0'):
        a = tf.constant([[1, 2], [3, 4]])
        b = tf.constant([[1, 2], [3, 4]])
        print(a * b)