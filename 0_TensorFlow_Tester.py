import tensorflow as tf
print("TensorFlow version:", tf.__version__)

tf.config.list_physical_devices('GPU')
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))